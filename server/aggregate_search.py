import logging
import logging.handlers

import json
import requests
import re
import time
from thefuzz import process
from urllib.parse import quote
from sanic import Sanic
from sanic.response import json as sanic_json

formatter = logging.Formatter(fmt='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
rf_handler = logging.handlers.TimedRotatingFileHandler(filename="logs/aggregate.log", when='D', interval=7,
                                                       backupCount=60)

logger_aggregate = logging.getLogger("aggregate_search")
logger_aggregate.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
rf_handler.setFormatter(formatter)

logger_aggregate.addHandler(ch)
logger_aggregate.addHandler(rf_handler)

logger_aggregate.info("start")


def sentence_classification(sentence, max_length=32):
    """输入句子的简单匹配"""

    sentence = sentence.replace("\n", "").replace(" ", "")
    if len(sentence) > max_length:
        return "fulltext_search"

    elif len(re.findall(r"[。！？…… …]", sentence)) > 0 and sentence[-1] not in "。！？…… …":
        return "fulltext_search"

    elif len(re.findall(r"[,，]", sentence)) > 1:
        return "fulltext_search"

    else:
        return "entity_search"


def fuzzy_search_main(word, word_one, top):
    """模糊匹配"""
    return process.extract(word, word_one, limit=top)


def score_search(search_result, text_input):
    """根据模糊匹配对搜索结果进行评分"""
    new_search_text = {text_input: {}}

    for key, values in search_result[text_input].items():
        search_text = text_input + key
        titles = set([v["title"] for v in values])
        titles_score = fuzzy_search_main(search_text, titles, len(titles))
        titles_score_dict = dict(zip([score[0] for score in titles_score], [score[1] for score in titles_score]))

        for v in values:
            v["score"] = titles_score_dict[v["title"]]

        search_result[text_input][key] = sorted(values, key=lambda x: x["score"], reverse=True)

        if len(key) > 1 and text_input not in key and len(search_result[text_input]) > 0 and \
                search_result[text_input][key][0]["score"] > 5:
            new_search_text[text_input][key] = search_result[text_input][key]

    return new_search_text


def containment_relationship(search_result, text_input):
    """寻找包含关系的标签"""

    nodes = [k for k, v in search_result[text_input].items()]

    abbreviation = {'研究现状': '现状',
                    '研究进展': '现状',
                    '进展': '现状',
                    '概述': '简介',
                    '介绍': '简介',
                    '概况': '简介',
                    '情况': '简介',
                    '理论': '简介',
                    '综述': '简介',
                    '描述': '简介',
                    '基本情况介绍': '简介',
                    '概念': '定义',
                    '涵义': '定义'}

    for i in nodes:
        for j in nodes:
            if i != j and i in j:
                abbreviation[j] = i

    return abbreviation


def search_abbreviation(search_result, text_input, attributes):
    """合并包含关系的标签"""

    abbreviation = containment_relationship(search_result, text_input)
    new_search_result = {}
    if attributes in abbreviation:
        del abbreviation[attributes]

    for key, values in search_result[text_input].items():
        if key in abbreviation:
            if abbreviation[key] in new_search_result:
                new_search_result[abbreviation[key]].extend(values)
            else:
                new_search_result[abbreviation[key]] = values

        elif key in new_search_result:
            new_search_result[key].extend(values)

        else:
            new_search_result[key] = values

    new_search_result = {text_input: new_search_result}

    return new_search_result


def sort_nodes(search_result, text_input, attributes):
    """属性排序"""
    nodes_score = {}

    for k, v in search_result[text_input].items():
        score_list = [v_["score"] for v_ in v]
        score_list.extend([90, 80])
        nodes_score[k] = sum(score_list[:3]) / 3

    if attributes in nodes_score:
        nodes_score[attributes] = 200

    nodes = sorted(nodes_score.items(), key=lambda x: x[1], reverse=True)
    nodes = [i[0] for i in nodes]

    return nodes


def get(word, attr, size=100):
    """根据词语和属性搜索"""
    url = "http://fulltext-search.k8s.laibokeji.com/search/enrityInfo/fulltext_v3?q=%s&size=%s&attr=%s" % (
        word, size, attr)
    response = requests.request("GET", url)
    try:
        response = response.json()
    except Exception as ex:
        logger_aggregate.error(ex)
    return response


def post(text, timeout=30):
    """前缀树搜索"""
    url = "http://127.0.0.1:33366/tree"

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=json.dumps({"text": text}), timeout=timeout)
    try:
        response = response.json()
    except Exception as ex:
        logger_aggregate.error(ex)
    return response


def search(text, num=10, version="fulltext_v1"):
    """片段搜索接口"""
    url = "http://fulltext-search.k8s.laibokeji.com/search/fulltext/%s?q=%s&size=%s" % (version, quote(text), num)

    result = requests.request("GET", url)
    result = result.json()

    docid_list = [data["id"] for data in result["content_list"]]

    result = [
        data["content"].replace("<whigh>", "").replace("</whigh>", "").replace("<phigh>", "<em>").replace("</phigh>",
                                                                                                          "</em>") for
        data in result["content_list"]]

    result = [{"content": content, "docid": docid} for content, docid in zip(result, docid_list)]

    return {"content_list": result}


def clear_content(search_result, entity, attributes):
    new_search_result = {entity: {}}
    for key, values in search_result[entity].items():
        new_values = []
        for value in values:
            entity_tmp = entity.lower()
            content_tmp = value["content"].lower().replace("\n", "").replace("", "")
            if entity_tmp in content_tmp:
                new_values.append(value)

        if len(new_values) > 0:
            new_search_result[entity][key] = new_values

    return new_search_result


def search_classification(sentence, way):
    if way == "fulltext_search":
        return {"fulltext_search": search(sentence)}

    elif way == "entity_search":
        result = post(sentence)
        entity, attributes, beam_score = result["entity"], result["attributes"], result["score"]
        search_result = get(entity, attributes)
        search_result = clear_content(search_result, entity, attributes)
        search_result = search_abbreviation(search_result, entity, attributes)
        search_result = score_search(search_result, entity)
        nodes = sort_nodes(search_result, entity, attributes)
        return {"entity_search": {"search_result": search_result, "nodes": nodes}}

    else:
        classification = sentence_classification(sentence)
        if classification == "entity_search":

            result = post(sentence)
            entity, attributes, beam_score = result["entity"], result["attributes"], result["score"]
            search_result = get(entity, attributes)
            search_result = clear_content(search_result, entity, attributes)
            search_result = search_abbreviation(search_result, entity, attributes)
            search_result = score_search(search_result, entity)
            nodes = sort_nodes(search_result, entity, attributes)

            if beam_score > -20 and len(nodes) > 0:
                all_score = sum(abb["score"] for abb in search_result[entity][nodes[0]])
                if all_score > 100:
                    return {"entity_search": {"search_result": search_result, "nodes": nodes}}
                else:
                    return {"fulltext_search": search(sentence)}
            else:
                return {"fulltext_search": search(sentence)}

        return {"fulltext_search": search(sentence)}


app = Sanic("agg")


@app.route("/agg", methods=['POST'])
async def reductions(request):
    jsonDic = request.json
    logger_aggregate.info("输入->" + str(jsonDic))

    try:
        sentence = jsonDic["sentence"]
    except Exception:
        return sanic_json({"code": -1, "msg": "输入有误", "data": {}, "time": int(time.time())})

    try:
        way = jsonDic["way"]
    except Exception:
        way = "aggregate_search"

    try:
        sentence = search_classification(sentence, way)
        sentence = {"code": 0, "msg": "success", "data": sentence, "time": int(time.time())}
    except Exception as ex:
        logger_aggregate.error(ex)
        sentence = {"code": 0, "msg": "null", "data": {}, "time": int(time.time())}

    logger_aggregate.info("输出->" + str(sentence))

    return sanic_json(sentence)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=51234, workers=4)


