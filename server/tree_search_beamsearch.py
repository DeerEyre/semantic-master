import torch
from utils import utils_seq2seq
import requests
import argparse
import json
import requests
import numpy as np
import jieba.posseg as pseg
import math
from torch import nn
from config.project_config import *

from model.tokenization_unilm import UnilmTokenizer
from model.modeling_unilm import UnilmForSeq2SeqDecode, UnilmConfig
from sanic import Sanic
from sanic.response import json as sanic_json


def read_model():

    config = UnilmConfig.from_pretrained(model_config_path, max_position_embeddings=512)
    tokenizer_ = UnilmTokenizer.from_pretrained(tokenizer_path)

    mask_word_id, eos_word_ids, sos_word_id = tokenizer_.convert_tokens_to_ids(["[MASK]", "[SEP]", "[S2S_SOS]"])

    bi_uni_pipeline_ = list()
    bi_uni_pipeline_.append(utils_seq2seq.Preprocess4Seq2seqDecode(list(tokenizer_.vocab.keys()),
                                                                   tokenizer_.convert_tokens_to_ids,
                                                                   512, max_tgt_length=400))
    model_recover = torch.load(model_path)
    model = UnilmForSeq2SeqDecode.from_pretrained(model_path, state_dict=model_recover,
                                                  config=config,
                                                  mask_word_id=mask_word_id,
                                                  search_beam_size=4,
                                                  length_penalty=2,
                                                  eos_id=eos_word_ids,
                                                  sos_id=sos_word_id,
                                                  ngram_size=3)

    del model_recover

    device_ = torch.device(cuda)
    model.to(device_)
    torch.cuda.empty_cache()
    model.eval()

    return model, tokenizer_, bi_uni_pipeline_, device_


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def search_pos(text):
    new_pos = []
    for word, pos in pseg.lcut(text):
        word_token = tokenizer.tokenize(word)
        if pos in pos_tag:
            new_pos.extend([pos_tag[pos]] * len(word_token))

        else:
            new_pos.extend(["❤"] * len(word_token))

    return text + "➤" + "".join(new_pos)


def score_percentage(score):
    score = score if score > 3 else 3
    score = score if score < 10 else 10

    score = (1 - (score - 3) / 7) * 100

    return int(round(score, 0))


def first_expand(x, K):
    input_shape = list(x.size())
    expanded_shape = input_shape[:1] + [1] + input_shape[1:]
    x = torch.reshape(x, expanded_shape)
    repeat_count = [1, K] + [1] * (len(input_shape) - 1)
    x = x.repeat(*repeat_count)
    x = torch.reshape(x, [input_shape[0] * K] + input_shape[1:])
    return x


def select_beam_items(x, ids, K, batch_size):
    ids = ids.long()
    id_shape = list(ids.size())
    id_rank = len(id_shape)
    assert len(id_shape) == 2
    x_shape = list(x.size())
    x = torch.reshape(x, [batch_size, K] + x_shape[1:])
    x_rank = len(x_shape) + 1
    assert x_rank >= 2
    if id_rank < x_rank:
        ids = torch.reshape(
            ids, id_shape + [1] * (x_rank - id_rank))
        ids = ids.expand(id_shape + x_shape[1:])
    y = torch.gather(x, 1, ids)
    y = torch.reshape(y, x_shape)
    return y


def text_en_word(text):
    en_token = []
    for word, pos in pseg.lcut(text):
        if pos == "eng":
            en_token.extend(tokenizer.encode(word))

    en_token = list(set(en_token))
    return en_token


def generate_step(model, input_ids, token_type_ids, position_ids, attention_mask, en_token):
    input_shape = list(input_ids.size())
    batch_size = input_shape[0]
    input_length = input_shape[1]
    output_shape = list(token_type_ids.size())
    output_length = output_shape[1]

    prev_embedding = None
    prev_encoded_layers = None
    curr_ids = input_ids
    mask_ids = input_ids.new(batch_size, 1).fill_(mask_word_id)
    next_pos = input_length
    K = search_beam_size
    total_scores = []
    beam_masks = []
    step_ids = []
    step_back_ptrs = []
    forbid_word_mask = None
    is_sep = [0 for _ in range(batch_size * K)]

    generate_id = [[] for _ in range(batch_size * K)]
    generate_logits = [[] for _ in range(batch_size * K)]

    while next_pos < output_length:
        curr_length = list(curr_ids.size())[1]

        start_pos = next_pos - curr_length
        x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)

        curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
        curr_attention_mask = attention_mask[:, start_pos:next_pos + 1, :next_pos + 1]
        curr_position_ids = position_ids[:, start_pos:next_pos + 1]

        new_embedding, new_encoded_layers, _ = \
            model.bert(x_input_ids, curr_token_type_ids, curr_position_ids, curr_attention_mask,
                       output_all_encoded_layers=True, prev_embedding=prev_embedding,
                       prev_encoded_layers=prev_encoded_layers)

        last_hidden = new_encoded_layers[-1][:, -1:, :]
        prediction_scores = model.cls(last_hidden)
        if len(en_token) > 0:
            prediction_scores[:, :, en_token] *= 1.4

        prediction_scores_cp = torch.ones(prediction_scores.size())
        prediction_scores_cp.copy_(prediction_scores)

        for p in range(prediction_scores.size()[0]):

            last_generate_id = [str(i) for i in generate_id[p]]
            now_generate_id = KG.next_ones(last_generate_id)

            if len(now_generate_id) == 0:
                prediction_scores[p, :, 0] += 10000

            else:
                now_generate_id = [int(i) for i in now_generate_id]
                prediction_scores[p, :, now_generate_id] += 10000

            if len(generate_id[p]) == 0:
                prediction_scores[p, :, 102] -= 1000000000

            if len(generate_id[p]) > 0 and is_sep[p] == 1 and generate_id[p][-1] == 102:
                prediction_scores[p, :, 102] -= 1000000000

        prediction_scores[:, :, 100] = 10e-6

        log_scores = torch.nn.functional.log_softmax(
            prediction_scores, dim=-1)

        if forbid_word_mask is not None:
            log_scores += (forbid_word_mask * -10000.0)
        if min_len and (next_pos - input_length + 1 <= min_len):
            log_scores[:, :, eos_id].fill_(-10000.0)

        kk_scores, kk_ids = torch.topk(log_scores, k=K)

        if len(total_scores) == 0:
            k_ids = torch.reshape(kk_ids, [batch_size, K])
            back_ptrs = torch.zeros(batch_size, K, dtype=torch.long)
            k_scores = torch.reshape(kk_scores, [batch_size, K])
        else:
            last_eos = torch.reshape(
                beam_masks[-1], [batch_size * K, 1, 1])
            last_seq_scores = torch.reshape(
                total_scores[-1], [batch_size * K, 1, 1])
            kk_scores += last_eos * (-10000.0) + last_seq_scores
            kk_scores = torch.reshape(kk_scores, [batch_size, K * K])
            k_scores, k_ids = torch.topk(kk_scores, k=K)

            back_ptrs = torch.div(k_ids, K)

            kk_ids = torch.reshape(kk_ids, [batch_size, K * K])
            k_ids = torch.gather(kk_ids, 1, k_ids)

        step_back_ptrs.append(back_ptrs)
        step_ids.append(k_ids)

        beam_masks.append(torch.eq(k_ids, eos_id).float())
        total_scores.append(k_scores)

        is_first = (prev_embedding is None)

        if prev_embedding is None:
            prev_embedding = first_expand(new_embedding[:, :-1, :], K)
        else:
            prev_embedding = torch.cat((prev_embedding, new_embedding[:, :-1, :]), dim=1)
            prev_embedding = select_beam_items(prev_embedding, back_ptrs, K, batch_size)

        if prev_encoded_layers is None:
            prev_encoded_layers = [first_expand(x[:, :-1, :], K) for x in new_encoded_layers]
        else:
            prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                   for x in zip(prev_encoded_layers, new_encoded_layers)]
            prev_encoded_layers = [select_beam_items(x, back_ptrs, K, batch_size) for x in prev_encoded_layers]

        curr_ids = torch.reshape(k_ids, [batch_size * K, 1])

        if is_first:
            token_type_ids = first_expand(token_type_ids, K)
            position_ids = first_expand(position_ids, K)
            attention_mask = first_expand(attention_mask, K)
            mask_ids = first_expand(mask_ids, K)

        output = torch.reshape(curr_ids, (1, -1)).tolist()[0]
        generate_id_cp = generate_id[:]
        generate_logits_cp = generate_logits[:]

        for i in range(batch_size * K):
            if len(generate_id[i]) == 0:
                generate_id[i].append(output[i])
                generate_logits[i].append(prediction_scores_cp[0, :, output[i]])

            else:
                j = back_ptrs[-1].tolist()[i]
                generate_id[i] = generate_id_cp[int(j)] + [output[i]]

                generate_logits[i] = generate_logits_cp[int(j)] + [prediction_scores_cp[i, :, output[i]]]

            if output[i] == 0:
                is_sep[i] += 1
        if min(is_sep) == 1:
            break

        next_pos += 1

    min_score = 10e6
    for n, token_m in zip(generate_logits, generate_id):
        id_102 = 0
        length_n = 0
        for nt, t in enumerate(token_m):
            if t == 102:
                id_102 += 1
            if id_102 > 1:
                length_n = nt
                break
        length_n = len(n) if length_n == 0 else length_n
        score_n = torch.cat(n[:length_n], axis=-1).std().tolist()
        if score_n < min_score:
            min_score = score_n

    beam_score = score_percentage(min_score)

    total_scores = [x.tolist() for x in total_scores]
    step_ids = [x.tolist() for x in step_ids]
    step_back_ptrs = [x.tolist() for x in step_back_ptrs]

    traces = {'pred_seq': [], 'scores': [], 'wids': [], 'ptrs': []}
    for b in range(batch_size):
        scores = [x[b] for x in total_scores]
        wids_list = [x[b] for x in step_ids]
        ptrs = [x[b] for x in step_back_ptrs]
        traces['scores'].append(scores)
        traces['wids'].append(wids_list)
        traces['ptrs'].append(ptrs)
        last_frame_id = len(scores) - 1

        max_score = -math.inf
        frame_id = -1
        pos_in_frame = -1

        for fid in range(last_frame_id + 1):
            for i, wid in enumerate(wids_list[fid]):
                if wid == eos_id or fid == last_frame_id:
                    s = scores[fid][i]
                    if length_penalty > 0:
                        s /= math.pow((5 + fid + 1) / 6.0, length_penalty)
                    if s > max_score:
                        max_score = s
                        frame_id = fid
                        pos_in_frame = i

        if frame_id == -1:
            traces['pred_seq'].append([0])
        else:
            seq = [wids_list[frame_id][int(pos_in_frame)]]
            for fid in range(frame_id, 0, -1):
                pos_in_frame = ptrs[fid][int(pos_in_frame)]
                seq.append(wids_list[int(fid - 1)][int(pos_in_frame)])
            seq.reverse()
            traces['pred_seq'].append(seq)

    return traces, beam_score


def outputs(output_ids, buf):
    output_sequence = []
    for i in range(len(buf)):
        w_ids = output_ids[i]
        output_buf = tokenizer.convert_ids_to_tokens(w_ids)
        output_tokens = []
        count_sep = 0
        for t in output_buf:
            if t in ("[SEP]",):
                count_sep += 1
            if t in ("[PAD]",) or count_sep >= 2:
                break

            output_tokens.append(t)
        output_sequence.append(''.join(detokenize(output_tokens)))

    return output_sequence


def generate_all(model, titles, device_):
    en_token = text_en_word(titles[0])
    titles = [search_pos(titles[0])]

    max_src_length = 512 - 2 - 400

    input_lines = [tokenizer.tokenize(x)[:max_src_length] for x in titles]

    buf = input_lines
    max_a_len = max([len(x) for x in buf])
    instances = []
    for instance in [(x, max_a_len) for x in buf]:
        for proc in bi_uni_pipeline:
            instances.append(proc(instance))

    with torch.no_grad():
        batch = utils_seq2seq.batch_list_to_batch_tensors(instances)
        batch = [t.to(device_) if t is not None else None for t in batch]
        input_ids, token_type_ids, position_ids, input_mask = batch
        traces, beam_score = generate_step(model, input_ids, token_type_ids, position_ids, input_mask, en_token)

        out = outputs(traces['pred_seq'], buf)

    return out[0], beam_score


def tree_search_out(text, model, device_):

    output_text, beam_score = generate_all(model, [text], device_)
    entity, attributes = output_text.split("[SEP]")

    return {"entity": entity, "attributes": attributes, "score": beam_score, "code": 1}


app = Sanic(__name__)


@app.route("/tree", methods=['POST'])
async def reductions(request):
    jsonDic = request.json
    print("输入： %s" % jsonDic)

    try:
        text = jsonDic["text"]

    except Exception as ex:
        print(ex)
        return sanic_json({"error": "输入错误", "code": -1})

    output_json = tree_search_out(text, unilm_model, device)

    print("输出： %s" % output_json)

    return sanic_json(output_json)


if __name__ == '__main__':
    min_len = 1
    length_penalty = 2
    search_beam_size = 3

    eos_id = 0
    sos_id = sos_word_id

    print("加载模型中...")
    unilm_model, tokenizer, bi_uni_pipeline, device = load_model()
    print("加载模型完成")

    KG = Trie()
    print("加载数据中...")
    KG.load(trie_path)
    print("加载数据完成")

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=33366, type=int, help='服务端口')

    sh_args = parser.parse_args()

    app.run(host="0.0.0.0", port=sh_args.port, workers=1)
