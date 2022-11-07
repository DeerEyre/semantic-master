import logging
import logging.handlers

import torch
import re
import requests
import Levenshtein
import numpy as np
import jieba.posseg as pseg
import math
from torch import nn
from config.project_config import *
from utils import utils_seq2seq
from model.trie_model import Trie
from model.tokenization_unilm import UnilmTokenizer
from model.modeling_unilm import UnilmForSeq2SeqDecode, UnilmConfig

formatter = logging.Formatter(fmt='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
rf_handler = logging.handlers.TimedRotatingFileHandler(filename="logs/beam_search.log", when='D', interval=7,
                                                       backupCount=60)

logger_trid = logging.getLogger("beam_search")
logger_trid.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
rf_handler.setFormatter(formatter)

logger_trid.addHandler(ch)
logger_trid.addHandler(rf_handler)

logger_trid.info("start")

def get_word_vector(s1,s2):
    """
    :param s1: 句子1
    :param s2: 句子2
    :return: 返回句子的余弦相似度
    """
    # 分词
    cut1 = jieba.cut(s1)
    cut2 = jieba.cut(s2)
    list_word1 = (','.join(cut1)).split(',')
    list_word2 = (','.join(cut2)).split(',')
 
    # 列出所有的词,取并集
    key_word = list(set(list_word1 + list_word2))
    # 给定形状和类型的用0填充的矩阵存储向量
    word_vector1 = np.zeros(len(key_word))
    word_vector2 = np.zeros(len(key_word))
 
    # 计算词频
    # 依次确定向量的每个位置的值
    for i in range(len(key_word)):
        # 遍历key_word中每个词在句子中的出现次数
        for j in range(len(list_word1)):
            if key_word[i] == list_word1[j]:
                word_vector1[i] += 1
        for k in range(len(list_word2)):
            if key_word[i] == list_word2[k]:
                word_vector2[i] += 1
 
    # 输出向量
    return word_vector1, word_vector2

def cos_dist(vec1,vec2):
    """
    :param vec1: 向量1
    :param vec2: 向量2
    :return: 返回两个向量的余弦相似度
    """
    dist1=float(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
    return dist1

def read_model():
    """读取模型及相关配置"""

    config = UnilmConfig.from_pretrained(model_config_path, max_position_embeddings=512)
    tokenizer_ = UnilmTokenizer.from_pretrained(tokenizer_path)

    mask_word_id_, eos_word_ids_, sos_word_id_ = tokenizer_.convert_tokens_to_ids(["[MASK]", "[SEP]", "[S2S_SOS]"])

    bi_uni_pipeline_ = list()
    bi_uni_pipeline_.append(utils_seq2seq.Preprocess4Seq2seqDecode(list(tokenizer_.vocab.keys()),
                                                                   tokenizer_.convert_tokens_to_ids,
                                                                   512, max_tgt_length=400))
    model_recover = torch.load(model_path,map_location=torch.device('cpu'))
    model = UnilmForSeq2SeqDecode.from_pretrained(model_path, state_dict=model_recover,
                                                  config=config,
                                                  mask_word_id=mask_word_id_,
                                                  search_beam_size=4,
                                                  length_penalty=1,
                                                  eos_id=eos_word_ids_,
                                                  sos_id=sos_word_id_,
                                                  ngram_size=3)

    del model_recover

    device_ = torch.device('cpu')
    model.to(device_)
    torch.cuda.empty_cache()
    model.eval()

    return model, tokenizer_, bi_uni_pipeline_, device_, mask_word_id_, eos_word_ids_, sos_word_id_


def detokenize(tk_list):
    """去除##"""
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def search_pos(text):
    """搜索语句拼接词性，构建模型输入"""
    new_pos = []
    for word, pos in pseg.lcut(text):
        word_token = tokenizer.tokenize(word)
        if pos in pos_tag:
            new_pos.extend([pos_tag[pos]] * len(word_token))

        else:
            new_pos.extend(["❤"] * len(word_token))

    return text + "➤" + "".join(new_pos)


def score_percentage(score):
    """置信度转化为百分比"""
    score = score if score > 3 else 3
    score = score if score < 10 else 10

    score = (1 - (score - 3) / 7) * 100

    return int(round(score, 0))


def first_expand(x, K):
    """第一次预测饿拼接"""
    input_shape = list(x.size())
    expanded_shape = input_shape[:1] + [1] + input_shape[1:]
    x = torch.reshape(x, expanded_shape)
    repeat_count = [1, K] + [1] * (len(input_shape) - 1)
    x = x.repeat(*repeat_count)
    x = torch.reshape(x, [input_shape[0] * K] + input_shape[1:])
    return x


def select_beam_items(x, ids, K, batch_size):
    """beam search输出选择"""
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
    """英文分词"""
    en_token = []
    for word, pos in pseg.lcut(text):
        if pos == "eng":
            en_token.extend(tokenizer.encode(word))

    en_token = list(set(en_token))
    return en_token


def generate_step(model, input_ids, token_type_ids, position_ids, attention_mask, en_token):
    """预测步骤, 包括beam search"""
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

    forbid_duplicate_ngrams = False
    forbid_ignore_set = None

    ngram_size = 3
    partial_seqs = []
    buf_matrix = None
    total_scores = []
    output_ids = []

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
        is_first = (prev_embedding is None)
        if not is_first:
            seq_group = []
            total_scores_copy = [x.tolist() for x in total_scores]
            step_ids_copy = [x.tolist() for x in step_ids]
            step_back_ptrs_copy = [x.tolist() for x in step_back_ptrs]
            for b in range(batch_size):
                scores = [x[b] for x in total_scores_copy]
                wids_list = [x[b] for x in step_ids_copy]
                ptrs = [x[b] for x in step_back_ptrs_copy]
                last_frame_id = len(scores) - 1

                for i in range(K):
                    seq = [wids_list[last_frame_id][i]]
                    pos = i
                    for fid in range(last_frame_id, 0, -1):
                        pos = ptrs[int(fid)][int(pos)]
                        seq.append(wids_list[int(fid) - 1][int(pos)])
                    seq.reverse()
                    seq_group.append(seq)
        for i in range(prediction_scores.size()[0]):
            if is_first:
                cur_candidates = KG.data.keys()
                cur_ids = [int(j) for j in cur_candidates]
                tmp_score = torch.full_like(prediction_scores, -float('inf'))
                tmp_score[i, :, cur_ids] = prediction_scores[i, :, cur_ids]
                prediction_scores[i, :, :] = tmp_score[i, :, :]
            else:
                last_generate_id = [str(word) for word in seq_group[i]]
                cur_candidates = KG.next_ones(last_generate_id)
                tmp_score = torch.full_like(prediction_scores,-float('inf'))
                if len(cur_candidates) == 0:
                    tmp_score[i, :, 0] = 100000
                else:
                    cur_ids = [int(j) for j in cur_candidates]
                    tmp_score[i, :, cur_ids] = prediction_scores[i, :, cur_ids]
                if len(seq_group[i]) == 0:
                    tmp_score[i, :, 102] = -float('inf')
                if len(seq_group[i]) > 0 and is_sep[i] == 1 and seq_group[i][-1] == 102:
                    prediction_scores[i, :, 102] = -float('inf')
                prediction_scores[i, :, :] = tmp_score[i, :, :]
        prediction_scores[:, :, 100] = -float('inf')
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
        if forbid_duplicate_ngrams:
            wids = step_ids[-1].tolist()
            ptrs = step_back_ptrs[-1].tolist()
            if is_first:
                partial_seqs = []
                for b in range(batch_size):
                    for k in range(K):
                        partial_seqs.append([wids[int(b)][int(k)]])
            else:
                new_partial_seqs = []
                for b in range(batch_size):
                    for k in range(K):
                        new_partial_seqs.append(
                            partial_seqs[int(ptrs[int(b)][int(k)] + int(b * K))] + [wids[int(b)][int(k)]])
                partial_seqs = new_partial_seqs

            def get_dup_ngram_candidates(seq, n):
                cands = set()
                if len(seq) < n:
                    return []
                tail = seq[-(n - 1):]
                if forbid_ignore_set and any(tk in forbid_ignore_set for tk in tail):
                    return []
                for i in range(len(seq) - (n - 1)):
                    mismatch = False
                    for j in range(n - 1):
                        if tail[j] != seq[i + j]:
                            mismatch = True
                            break
                    if (not mismatch) and not (forbid_ignore_set and (seq[i + n - 1] in forbid_ignore_set)):
                        cands.add(seq[i + n - 1])
                return list(sorted(cands))

            if len(partial_seqs[0]) >= ngram_size:
                dup_cands = []
                for seq in partial_seqs:
                    dup_cands.append(
                        get_dup_ngram_candidates(seq, ngram_size))
                if max(len(x) for x in dup_cands) > 0:
                    if buf_matrix is None:
                        vocab_size = list(log_scores.size())[-1]
                        buf_matrix = np.zeros(
                            (batch_size * K, vocab_size), dtype=float)
                    else:
                        buf_matrix.fill(0)
                    for bk, cands in enumerate(dup_cands):
                        for i, wid in enumerate(cands):
                            buf_matrix[bk, wid] = 1.0
                    forbid_word_mask = torch.tensor(
                        buf_matrix, dtype=log_scores.dtype)
                    forbid_word_mask = torch.reshape(
                        forbid_word_mask, [batch_size * K, 1, vocab_size]).cuda()
                else:
                    forbid_word_mask = None
        output = torch.reshape(curr_ids, (1, -1)).tolist()[0]
        for i in range(batch_size * K):
            if output[i] == 0:
                is_sep[int(back_ptrs[-1][i])] += 1
        if min(is_sep) == 1:
            break
        next_pos += 1
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
        frame_group = []
        pos_group = []
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
                        frame_group.append(frame_id)
                        pos_group.append(pos_in_frame)
        if frame_id == -1:
            traces['pred_seq'].append([0])
        else:
            seq = [wids_list[frame_id][int(pos_in_frame)]]
            for fid in range(frame_id, 0, -1):
                pos_in_frame = ptrs[fid][int(pos_in_frame)]
                seq.append(wids_list[int(fid - 1)][int(pos_in_frame)])
            seq.reverse()
            traces['pred_seq'].append(seq)
        
        seq_group = []
        for i in range(len(frame_group)):
            idd = frame_group[i]
            pos = pos_group[i]
            seq = [wids_list[idd][int(pos)]]
            for fid in range(idd, 0, -1):
                pos = ptrs[fid][int(pos)]
                seq.append(wids_list[int(fid - 1)][int(pos)])
            seq.reverse()
            seq_group.append(seq)
    return traces


def outputs(output_ids, buf):
    """模型输出的解码结果"""
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
    """模型输入的组装"""

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
        traces = generate_step(model, input_ids, token_type_ids, position_ids, input_mask, en_token)

        out = outputs(traces['pred_seq'], buf)

    return out[0]


def tree_search_out(text, model, device_):
    """返回搜索的实体与属性"""

    output_text = generate_all(model, [text], device_)  
    entity, attributes = output_text.split("[SEP]")
    out_text = ''.join([entity, attributes])
    score = int(Levenshtein.jaro(text, out_text) * 100 + 0.5)
    return {"entity": entity, "attributes": attributes, 'score': score, "code": 1}


min_len = 1
length_penalty = 2
search_beam_size = 3

eos_id = 0

logger_trid.info("加载模型中...")
unilm_model, tokenizer, bi_uni_pipeline, device, mask_word_id, eos_word_ids, sos_word_id = read_model()
logger_trid.info("加载模型完成")

sos_id = sos_word_id

KG = Trie()
logger_trid.info("加载数据中...")
KG.load(trie_path)
logger_trid.info("加载数据完成")






if __name__ == '__main__':
    import pdb
    import jsonlines
    i = 0
    texts = []
    with open('../sentrans/train.jsonl', 'r+', encoding='utf8') as f:
        for item in jsonlines.Reader(f):
            texts.append(item['query'])
            i += 1
            if i > 50:
                break
    result = []
    f = jsonlines.open('result4.jsonl', 'w')
    for text in texts:
        out = tree_search_out(text, unilm_model, device)
        f.write({'query': text, 'entity': out['entity'], 'attributes': out['attributes'], 'score': out['score'] })
        result.append(out['score'])
    f.close()
    score = 0
    for item in result:
        score += item
    score /= len(result)
    print('平均分:', score)
    #output_json = tree_search_out(text, unilm_model, device)
    #print(output_json)
    torch.cuda.empty_cache()

