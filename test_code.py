import argparse
import json
import os

import yaml
from attrdict import AttrDict
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from collections import defaultdict

from src.common import set_seed, ScoreManager, update_config

import ast
import numpy as np
from tqdm.auto import tqdm
import jsonlines


tx1 = {"doc_id": "0022",
       "sentences": ["今 年 想 换 安 卓 , 屏 幕 + 处 理 器 + 系 统 , 就 没 一 部 能 用 的 [ 打 脸 ]", "iqoo 9 [ doge ]",
                     "那 塑 料 支 架 都 吐 了 , 屏 幕 1080 [ 允 悲 ]",
                     "不 好 意 思 , 三 星 钻 排 就 算 是 1080 p 还 是 比 你 的 周 冬 雨 细 腻 呢 [ doge ]",
                     "连 影 像 都 缩 了 , 想 在 系 统 上 和 ios 拼 , 这 是 到 底 是 自 信 还 是 自 杀 ?",
                     "确 实 , 除 了 快 充 已 经 没 什 么 优 势 了",
                     "快 充 是 不 错 , 但 是 苹 果 今 年 的 续 航 冠 绝 机 圈 , 变 相 补 足 短 板 , 能 用 到 晚 上 不 亏 电 就 不 会 有 什 么 影 响 体 验 , 反 观 安 卓 这 边 , 续 航 一 个 比 一 个 拉 , 现 在 苹 果 真 有 大 师 兄 的 感 觉 , 外 卖 小 子 却 不 见 踪 影 。",
                     "magic v 屏 幕 挺 好 的 [ doge ]",
                     "只 看 屏 幕 的 画 , 能 选 的 也 不 少 , 是 因 为 多 项 兼 具 的 少"],
       "replies": [-1, 0, 1, 2, 0, 4, 5, 0, 7], "speakers": [0, 1, 2, 1, 3, 0, 3, 4, 5],
       "sentence_ids": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                        1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                        4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6,
                        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
       "triplets": [[4, 6, 7, 9, 18, 23, "neg", "安卓", "屏幕", "没一部能用"],
                    [4, 6, 10, 13, 18, 23, "neg", "安卓", "处理器", "没一部能用"],
                    [4, 6, 14, 16, 18, 23, "neg", "安卓", "系统", "没一部能用"],
                    [54, 56, 56, 58, 71, 73, "pos", "三星", "钻排", "细腻"],
                    [128, 130, 133, 135, 135, 139, "pos", "苹果", "续航", "冠绝机圈"],
                    [168, 170, 173, 175, 175, 181, "neg", "安卓", "续航", "一个比一个拉"],
                    [184, 186, -1, -1, 188, 191, "pos", "苹果", "", "大师兄"],
                    [205, 207, 207, 209, 209, 211, "pos", "magic v", "屏幕", "挺好"],
                    [4, 6, 78, 80, 81, 83, "neg", "安卓", "影像", "缩了"]],
       "targets": [[4, 6, "安卓"], [54, 56, "三星"], [128, 130, "苹果"], [168, 170, "安卓"], [184, 186, "苹果"],
                   [205, 207, "magic v"]],
       "aspects": [[10, 13, "处理器"], [133, 135, "续航"], [14, 16, "系统"], [56, 58, "钻排"], [217, 219, "屏幕"],
                   [78, 80, "影像"], [34, 38, "塑料支架"], [42, 44, "屏幕"], [123, 125, "不错"], [173, 175, "续航"],
                   [120, 122, "快充"], [86, 88, "系统"], [207, 209, "屏幕"], [7, 9, "屏幕"]],
       "opinions": [[175, 181, "一个比一个拉", "neg"], [209, 211, "挺好", "pos"], [81, 83, "缩了", "neg"],
                    [135, 139, "冠绝机圈", "pos"], [44, 45, "1080", "neg"], [39, 41, "吐了", "neg"],
                    [188, 191, "大师兄", "pos"], [71, 73, "细腻", "pos"], [18, 23, "没一部能用", "neg"]]}

tx2 = {"doc_id": "0062", "sentences": [
    "大 部 分 不 好 看 , 但 是 还 是 有 的 , 小 米 6 , 小 米 note 3 , 10 u , 11 u 都 好 看 ( 个 人 审 美 )",
    "小 米 11 u 那 个 大 摄 像 头 好 看 嘛 [ 允 悲 ]",
    "我 觉 得 挺 好 看 的 [ 允 悲 ] [ 允 悲 ] 有 特 色 , 白 陶 瓷 好 看 。 mate 40 系 列 也 好 看 , 我 不 太 喜 欢 千 篇 一 律 的 摄 像 头 , 有 特 色 并 且 不 丑 , 我 就 挺 喜 欢 的 [ doge ]",
    "我 感 觉 太 大 了 , 不 如 11 的 好 看 [ 允 悲 ] 白 色 陶 瓷 我 也 挺 喜 欢",
    "兄 弟 忘 了 小 米 note 2 了 ? 那 个 手 机 也 很 奈 斯 啊 [ doge ]",
    "手 感 还 行 , 配 色 也 还 行 , 但 是 后 置 摄 像 头 不 太 好 看 [ doge ] 所 以 我 也 没 说 mix 2",
    "手 持 10 u , 看 是 好 看 , 这 个 指 纹 真 的 受 不 了"], "replies": [-1, 0, 1, 2, 0, 4, 0],
       "speakers": [0, 1, 0, 1, 2, 0, 3],
       "sentence_ids": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                        4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                        5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
       "triplets": [[14, 17, -1, -1, 29, 31, "pos", "小米6", "", "好看"],
                    [18, 22, -1, -1, 29, 31, "pos", "小米note3", "", "好看"],
                    [23, 25, -1, -1, 29, 31, "pos", "10u", "", "好看"],
                    [26, 28, -1, -1, 29, 31, "pos", "11u", "", "好看"],
                    [37, 41, 44, 47, 47, 50, "doubt", "小米11u", "摄像头", "好看嘛"],
                    [37, 41, 44, 47, 57, 60, "pos", "小米11u", "摄像头", "挺好看"],
                    [37, 41, 44, 47, 69, 72, "pos", "小米11u", "摄像头", "有特色"],
                    [79, 83, -1, -1, 84, 86, "pos", "mate40系列", "", "好看"],
                    [79, 83, 97, 99, 101, 108, "pos", "mate40系列", "摄像", "有特色并且不丑"],
                    [37, 41, 44, 47, 121, 124, "neg", "小米11u", "摄像头", "太大了"],
                    [148, 152, 166, 168, 168, 170, "pos", "小米note2", "手感", "还行"],
                    [148, 152, 171, 173, 174, 176, "pos", "小米note2", "配色", "还行"],
                    [148, 152, -1, -1, 159, 162, "pos", "小米note2", "", "很奈斯"],
                    [148, 152, 179, 184, 184, 188, "neg", "小米note2", "后置摄像头", "不太好看"],
                    [201, 203, -1, -1, 206, 208, "pos", "10u", "", "好看"],
                    [201, 203, 211, 213, 213, 218, "neg", "10u", "指纹", "真的受不了"],
                    [37, 41, 44, 47, 125, 131, "neg", "小米11u", "摄像头", "不如11的好看"],
                    [127, 128, 44, 47, 125, 131, "pos", "11", "摄像头", "不如11的好看"]],
       "targets": [[14, 17, "小米6"], [18, 22, "小米note3"], [23, 25, "10u"], [26, 28, "11u"], [37, 41, "小米11u"],
                   [79, 83, "mate40系列"], [127, 128, "11"], [148, 152, "小米note2"], [197, 199, "mix2"],
                   [201, 203, "10u"]],
       "aspects": [[179, 184, "后置摄像头"], [211, 213, "指纹"], [97, 99, "摄像"], [44, 47, "摄像头"],
                   [166, 168, "手感"], [171, 173, "配色"]],
       "opinions": [[168, 170, "还行", "pos"], [121, 124, "太大了", "neg"], [125, 131, "不如11的好看", "pos"],
                    [206, 208, "好看", "pos"], [84, 86, "好看", "pos"], [57, 60, "挺好看", "pos"],
                    [174, 176, "还行", "pos"], [213, 218, "真的受不了", "neg"], [184, 188, "不太好看", "neg"],
                    [125, 131, "不如11的好看", "neg"], [69, 72, "有特色", "pos"], [47, 50, "好看嘛", "doubt"],
                    [159, 162, "很奈斯", "pos"], [101, 108, "有特色并且不丑", "pos"], [29, 31, "好看", "pos"]]}

bert_path = './chinese-roberta-wwm-ext'
tokenizer = AutoTokenizer.from_pretrained(bert_path)


def read_data(config, mode='valid'):
    """
    Read a JSON file, tokenize using BERT, and realign the indices of the original elements according to the tokenization results.
    """

    path = os.path.join(config.json_path, '{}.json'.format(mode))

    if not os.path.exists(path):
        raise FileNotFoundError('File {} not found! Please check your input and data path.'.format(path))

    content = json.load(open(path, 'r', encoding='utf-8'))
    res = []
    for line in tqdm(content, desc='Processing dialogues for {}'.format(mode)):
        # dialogue = line
        # sentences = dialogue['sentences']
        # new_sentences, pieces2words = align_index_with_list(sentences)
        # print(f'new_sentences: {new_sentences}')
        # print(f'pieces2words: {pieces2words}')
        new_dialog = parse_dialogue(line, mode)
        res.append(new_dialog)
        break
    return res


def parse_dialogue(dialogue, mode='valid'):
    # get the list of sentences in the dialogue
    sentences = dialogue['sentences']

    # align_index_with_list: align the index of the original elements according to the tokenization results
    # new_sentences : [[xx, xx, ...], [xx, xx, ...], ...]
    # pieces2words : [0, 1, 2, ...]
    new_sentences, pieces2words = align_index_with_list(sentences)
    print(f'pieces2words: {pieces2words}')
    word2pieces = defaultdict(list)
    for p, w in enumerate(pieces2words):
        word2pieces[w].append(p)
    print(f'word2pieces: {word2pieces}')

    dialogue['pieces2words'] = pieces2words
    dialogue['sentences'] = new_sentences

    targets, aspects, opinions = [dialogue[w] for w in ['targets', 'aspects', 'opinions']]
    for x, y, z in targets:
        a, b, c = word2pieces[x][0], word2pieces[y - 1][-1] + 1, z
        print(a, b, c)


def align_index_with_list(sentences):
    """_summary_
    align the index of the original elements according to the tokenization results
    Args:
        sentences (_type_): List<str>
        e.g., xiao mi 12x is my favorite
    """
    pieces2word = []
    word_num = 0
    all_pieces = []
    for sentence in sentences:
        sentence = sentence.split()
        tokens = [tokenizer.tokenize(w) for w in sentence]
        cur_line = []
        for token in tokens:
            for piece in token:
                pieces2word.append(word_num)
            word_num += 1
            cur_line += token
        all_pieces.append(cur_line)

    return all_pieces, pieces2word


def get_dict(config):
    entity_dict = {"O": 0, "ENT-T": 1, "ENT-A": 2, "ENT-O": 3}
    polarity_dict = config.polarity_dict
    print(f"polarity_dict: {polarity_dict}")

    aspect_dict = {}
    for w in config.bio_mode:
        aspect_dict['{}{}'.format(w, '' if w == 'O' else '-' + config.asp_type)] = len(aspect_dict)
    print(f'aspect_dict: {aspect_dict}')

    target_dict = {}
    for w in config.bio_mode:
        target_dict['{}{}'.format(w, '' if w == 'O' else '-' + config.tgt_type)] = len(target_dict)
    print(f'target_dict: {target_dict}')

    opinion_dict = {'O': 0}
    for p in polarity_dict:
        if p == 'O': continue
        for w in config.bio_mode[1:]:
            opinion_dict['{}-{}_{}'.format(w, config.opi_type, p)] = len(opinion_dict)
    print(f'opinion_dict: {opinion_dict}')

    relation_dict = {'O': 0, 'yes': 1}
    return polarity_dict, target_dict, aspect_dict, opinion_dict, entity_dict, relation_dict


def find_utterance_index(replies, sentence_lengths=None):
    """
    replies: [-1, 0, 1, 2, 0, 4, 0, 6]  sentence_length: [32, 23, 42, 50, 84, 24, 37, 25]
    """
    print(f'replies: {replies}')
    utterance_collections = [i for i, w in enumerate(replies) if w == 0]  # utterance_collections: [1, 4, 6]
    zero_index = utterance_collections[1]
    for i in range(len(replies)):
        if i < zero_index: continue
        if replies[i] == 0:
            zero_index = i
        replies[i] = (i - zero_index)
    print(f'replies: {replies}')


def read_jsonl(filepath):
    _jsonl = []
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            _jsonl.append(obj)
    return _jsonl


def read_text(filepath):
    txt_list = [line.strip() for line in open(filepath,'r',encoding='utf-8').readlines()]
    return txt_list


def get_triplets(ent_matrix, rel_matrix, pol_matrix, token2sents):
    ent_list = rel_matrix2list(ent_matrix)
    rel_list = rel_matrix2list(rel_matrix)
    pol_list = rel_matrix2list(pol_matrix)
    res, pair = decode_triplet(ent_list, rel_list, pol_list, token2sents, printf=False)
    return res, pair


def rel_matrix2list(rel_matrix):
    '''
    Convert a (512*512) matrix to a list of relations.
    '''
    rel_list = []
    nonzero = rel_matrix.nonzero()
    for x_index, y_index in zip(*nonzero):
        dic_key = int(rel_matrix[x_index][y_index].item())
        rel_elem = (x_index, y_index, dic_key)
        rel_list.append(rel_elem)
    return rel_list


def decode_triplet(ent_list, rel_list, pol_list, token2sents, printf=False):
    if printf:
        print(f'ent_list: {ent_list}\nrel_list: {rel_list}\npol_list: {pol_list}\ntoken2sents: {token2sents}')
    # Entity dictionary, with structure (head: [(tail, relation type)])
    entity_elem_dic = defaultdict(list)
    entity2type = {}
    for entity in ent_list:
        if token2sents[entity[0]] != token2sents[entity[1]]: continue
        entity_elem_dic[entity[0]].append((entity[1], entity[2]))
        entity2type[entity[:2]] = entity[2]
    if printf:
        print(f'entity_elem_dic: {entity_elem_dic}')
        print(f'entity2type: {entity2type}')

    # Decoding polarity matrix
    pol_entity_elem = defaultdict(list)
    for h2h_pol in pol_list:
        pol_entity_elem[h2h_pol[0]].append((h2h_pol[1], h2h_pol[2]))
    if printf:
        print(f'pol_entity_elem: {pol_entity_elem}')

    # (boundary,boundary -> polarity) set
    b2b_relation_set = {}
    for rel in pol_list:
        b2b_relation_set[rel[:2]] = rel[-1]
    if printf:
        print(f'b2b_relation_set: {b2b_relation_set}')

    # tail2tail set
    t2t_relation_set = set()
    for rel in rel_list:
        if rel[2] == rel_dic['t2t']:
            t2t_relation_set.add(rel[:2])
    if printf:
        print(f't2t_relation_set: {t2t_relation_set}')

    # head2head dictionary, with structure (head1: [(head2, relation type)])
    h2h_entity_elem = defaultdict(list)
    for h2h_rel in rel_list:
        # for each head-to-head relationship, mark its entity as 0
        if h2h_rel[2] != rel_dic['h2h']: continue
        h2h_entity_elem[h2h_rel[0]].append((h2h_rel[1], h2h_rel[2]))
    if printf:
        print(f'h2h_entity_elem: {h2h_entity_elem}')

    # for all head-to-head relations
    triplets = []
    for h1, values in tqdm(h2h_entity_elem.items(), desc='h2h_entity_elem'):
        if h1 not in entity_elem_dic: continue
        for h2, rel_tp in values:
            if h2 not in entity_elem_dic: continue
            for t1, ent1_tp in entity_elem_dic[h1]:
                for t2, ent2_tp in entity_elem_dic[h2]:
                    if (t1, t2) not in t2t_relation_set: continue
                    triplets.append((h1, t1, h2, t2))

    # if there is a (0,0,0,0) in triplets, remove it
    if (0, 0, 0, 0) in triplets:
        triplets.remove((0, 0, 0, 0))

    triplet_set = set(triplets)
    if printf:
        print(f'triplets: {triplets}\ntriplet_set: {triplet_set}')

    ele2list = defaultdict(list)
    for line in triplets:
        e0, e1 = line[:2], line[2:]
        ele2list[e0].append(e1)
    if printf:
        print(f'ele2list: {ele2list}')

    tetrad = []
    for subj, obj_list in tqdm(ele2list.items(), desc='ele2list'):
        for obj in obj_list:
            if obj not in ele2list: continue
            for third in ele2list[obj]:
                if (*subj, *third) not in triplet_set: continue
                tp0 = b2b_relation_set.get((subj[0], third[0]), -1)
                tp1 = b2b_relation_set.get((subj[1], third[1]), -1)
                if (tp0 == tp1 or tp0 == -1) and tp1 != -1:
                    tetrad.append((*subj, *obj, *third, tp1))
                elif tp0 != -1 and tp1 == -1:
                    tetrad.append((*subj, *obj, *third, tp0))
                else:
                    tetrad.append((*subj, *obj, *third, 1))
    if printf:
        print(f'tetrad: {tetrad}')

    pairs = {'ta': [], 'to': [], 'ao': []}
    for line in triplets:
        h1, t1, h2, t2 = line
        tp1 = entity2type[(h1, t1)]
        tp2 = entity2type[(h2, t2)]
        if tp1 == 1 and tp2 == 2:
            pairs['ta'].append(line)
        elif tp1 == 2 and tp2 == 3:
            pairs['ao'].append(line)
        elif tp1 == 1 and tp2 == 3:
            pairs['to'].append(line)
    if printf:
        print(f'pairs: {pairs}')
    return set(tetrad), pairs


def filter_entity(ent_list, new2old, pieces2words):
    res = []

    # If the entity is a sub-string of another entity, remove it
    # ent_list = sorted(ent_list, key=lambda x: (x[0], -x[1]))
    # ent_list = [w for i, w in enumerate(ent_list) if i == 0 or w[0] != ent_list[i-1][0]]

    for s, e, pol in ent_list:
        ns, ne = pieces2words[new2old[str(s)]], pieces2words[new2old[str(e)]]
        res.append([ns, ne, pol])
    return res


def trans2pair(pred_pairs, new2old, pieces2words):
    new_pairs = {}
    new_pos = lambda x: pieces2words[new2old[x]]
    for k, line in pred_pairs.items():
        new_line = []
        for s0, e0, s1, e1 in line:
            s0, e0, s1, e1 = new_pos(str(s0)), new_pos(str(e0)), new_pos(str(s1)), new_pos(str(e1))
            new_line.append([s0, e0, s1, e1])
        new_pairs[k] = new_line
    return new_pairs


def save2file(gold_file, pred_file):
    # pol_dict = {"O": 0, "pos": 1, "neg": 2, "other": 3}
    # pol_dict = config.polarity_dict
    # reverse_pol_dict = {v: k for k, v in pol_dict.items()}
    # reverse_ent_dict = {v: k for k, v in config.entity_dict.items()}

    gold_file = open(gold_file, 'r', encoding='utf-8')

    data = json.load(gold_file)

    res = []
    for line in data:
        doc_id, sentence = line['doc_id'], line['sentences']
        doc = ' '.join(sentence).split()
        print(doc_id, sentence)
        print(doc)
        break


def get_token_thread(dialogue):
    """
    sentences: ['这 手 机 不 怎 么 滴 但 是 对 比 iPhone 我 觉 得 除 了 处 理 器 其 他 都 比 iPhone 好 [ 笑 cry ]', 'iPhone 就 处 理 器 和 ios 出 色 , 其 他 真 的 多 年 被 安 卓 吊 打', '确 实 。 销 量 也 吊 打 安 卓 。 安 卓 厂 商 天 天 自 称 高 端 高 端 , 然 而 在 苹 果 面 前 就 是 个 孩 子 [ 允 悲 ]', '三 星 , 小 米 不 都 超 过 苹 果 ? 因 为 安 卓 系 统 选 择 太 多 , ios 只 有 一 家 , 如 果 安 卓 也 只 有 一 家 , 你 觉 得 结 果 是 什 么 ?', '随 你 说 咯 , 反 正 我 没 用 过 小 米 , 也 不 能 妄 加 评 论 。 只 是 外 出 旅 游 , 朋 友 小 米 手 机 从 来 没 见 过 拍 得 好 的 照 片 过 。 特 别 这 次 去 马 岭 河 瀑 布 , 必 须 要 抓 拍 , 兄 弟 的 米 11 张 张 糊 的 , 这 体 验 也 没 谁 了', '小 米 11 拍 照 确 实 不 行 啊 [ 黑 线 ] [ 黑 线 ] [ 黑 线 ]', '参 数 年 年 吊 打 , 体 验 年 年 勉 强 .. 就 这 么 回 事 儿 。 手 机 是 自 己 的 , 谁 用 谁 知 道 。', '就 算 是 体 验 也 比 iPhone 好 啊 我 可 受 不 了 iPhone 落 后 的 影 像 系 统']
    replies: [-1, 0, 1, 2, 0, 4, 0, 6]
    """
    sentences = dialogue['sentences']
    replies = dialogue['replies']
    print(f'sentences: {sentences}\nreplies: {replies}')

    sentence_ids = [[i] * len(w.split()) for i, w in enumerate(sentences)]
    sentence_ids = [w for sent in sentence_ids for w in sent]

    thread_list = [[0]]
    cur_thread = []
    for i, r in enumerate(replies):
        if i == 0: continue
        if r > replies[i - 1]:
            cur_thread.append(i)
        else:
            thread_list.append(cur_thread)
            cur_thread = [i]
    if len(cur_thread) > 0:
        thread_list.append(cur_thread)
    print(f'thread_list: {thread_list}')

    dis_matrix = np.zeros([len(replies), len(replies)], dtype=int)
    for i in range(len(thread_list)):
        first_list = thread_list[i]
        for ii in range(len(first_list)):
            for j in range(i, len(thread_list)):
                second_list = thread_list[j]
                for jj in range(len(second_list)):
                    if i == j:
                        dis_matrix[first_list[ii], second_list[jj]] = abs(ii - jj)
                        dis_matrix[second_list[jj], first_list[ii]] = abs(ii - jj)
                    elif i * j == 0:
                        dis_matrix[first_list[ii], second_list[jj]] = ii + jj + 1
                        dis_matrix[second_list[jj], first_list[ii]] = ii + jj + 1
                    else:
                        dis_matrix[first_list[ii], second_list[jj]] = ii + jj + 2
                        dis_matrix[second_list[jj], first_list[ii]] = ii + jj + 2

    return dis_matrix, sentence_ids


def read_data1(path, mode='pred'):
    with open(path, 'r', encoding='utf-8') as f:
        content = json.load(f)
        content = {w['doc_id']: w for w in content}
    if mode == 'pred': return content

    new_content = {}
    for k, line in content.items():
        triplets = line['triplets']
        ta = [tuple(w[:4]) for w in triplets if all(z != -1 for z in w[:4])]
        to = [tuple(w[0:2] + w[4:6]) for w in triplets if all(z != -1 for z in w[:2] + w[4:6])]
        ao = [tuple(w[2:6]) for w in triplets if all(z != -1 for z in w[2:6])]

        dis_matrix, sentence_ids = get_token_thread(line)
        raise TypeError('!!!')
        line.update({'ta': ta, 'to': to, 'ao': ao, 'dis_matrix': dis_matrix, 'sentence_ids': sentence_ids})
        new_content[k] = line

    return new_content


if __name__ == '__main__':
    config = AttrDict(yaml.load(open('src/config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader))
    # print(config)
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang', type=str, default='zh', choices=['zh', 'en'], help='language selection')
    parser.add_argument('-b', '--bert_lr', type=float, default=1e-5, help='learning rate for BERT layers')
    parser.add_argument('-c', '--cuda_index', type=int, default=0, help='CUDA index')
    parser.add_argument('-s', '--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()
    for k, v in vars(args).items():
        setattr(config, k, v)
    config = update_config(config)
    # # read_data(config, mode='valid')
    # # get_dict(config)
    # rep = [-1, 0, 0, 2, 0, 4]
    # find_utterance_index(rep,)

    # rel_dic = {"O": 0, "h2h": 1, "t2t": 2}
    # save_step = read_jsonl('./save_step.jsonl')
    # pred_ent_matrix = save_step[0]['pred_ent_matrix']
    # pred_rel_matrix = save_step[0]['pred_rel_matrix']
    # pred_pol_matrix = save_step[0]['pred_pol_matrix']
    # token2sents = save_step[0]['token2sents']
    # new2old = save_step[0]['new2old']
    # pieces2words = save_step[0]['pieces2words']
    # pred_rel_matrix = np.array(pred_rel_matrix)
    # pred_ent_matrix = np.array(pred_ent_matrix)
    # pred_pol_matrix = np.array(pred_pol_matrix)
    # print(f'new2old: {new2old}\npieces2words: {pieces2words}')
    # for i in range(len(pred_ent_matrix)):
    #     ent_matrix, rel_matrix, pol_matrix = pred_ent_matrix[i], pred_rel_matrix[i], pred_pol_matrix[i]
    #     pred_triplet, pred_pairs = get_triplets(ent_matrix, rel_matrix, pol_matrix, token2sents[i])
    #     pred_ents = rel_matrix2list(ent_matrix)
    #     print(f'pred_ents: {pred_ents}')
    #     pred_ents = filter_entity(pred_ents, new2old[i], pieces2words[i])
    #     print(f'pred_ents: {pred_ents}')
    #     pred_pairs = trans2pair(pred_pairs, new2old[i], pieces2words[i])
    #     print(f'pred_pairs: {pred_pairs}')
    #     break

    config.json_path = 'data/dataset/jsons_zh/'
    name = 'valid'
    args = AttrDict({
        'pred_file': os.path.join(config.target_dir, 'pred_{}_{}.json'.format(config.lang, name)),
        'gold_file': os.path.join(config.json_path, '{}.json'.format(name))
        # 'gold_file': os.path.join(self.config.json_path, '{}_gold.json'.format(name))
    })
    # save2file(args.gold_file, args.pred_file)
    read_data1(args.gold_file, mode='gold')




