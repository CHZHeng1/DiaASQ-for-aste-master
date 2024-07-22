from src.utils import WordPair
import os
import re
import json

import numpy as np

from collections import defaultdict
from itertools import accumulate
from transformers import AutoTokenizer
from typing import List, Dict
from loguru import logger
from tqdm import tqdm


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
        self.wordpair = WordPair()
        self.entity_dict = self.wordpair.entity_dic  # {"O": 0, "ENT-T": 1, "ENT-A": 2, "ENT-O": 3}

    def get_dict(self):
        self.polarity_dict = self.config.polarity_dict  # {'O': 0, 'pos': 1, 'neg': 2, 'other': 3}

        self.aspect_dict = {}  # {'O': 0, 'B-Aspect': 1, 'I-Aspect': 2, 'E-Aspect': 3, 'S-Aspect': 4}
        for w in self.config.bio_mode:
            self.aspect_dict['{}{}'.format(w, '' if w == 'O' else '-' + self.config.asp_type)] = len(self.aspect_dict)

        # self.target_dict = {}  # {'O': 0, 'B-Target': 1, 'I-Target': 2, 'E-Target': 3, 'S-Target': 4}
        # for w in self.config.bio_mode:
        #     self.target_dict['{}{}'.format(w, '' if w == 'O' else '-' + self.config.tgt_type)] = len(self.target_dict)

        # {'O': 0, 'B-Opinion_pos': 1, 'I-Opinion_pos': 2, 'E-Opinion_pos': 3, 'S-Opinion_pos': 4, 'B-Opinion_neg': 5,
        # 'I-Opinion_neg': 6, 'E-Opinion_neg': 7, 'S-Opinion_neg': 8, 'B-Opinion_other': 9, 'I-Opinion_other': 10,
        # 'E-Opinion_other': 11, 'S-Opinion_other': 12}
        self.opinion_dict = {'O': 0}
        for p in self.polarity_dict:
            if p == 'O': continue
            for w in self.config.bio_mode[1:]:
                self.opinion_dict['{}-{}_{}'.format(w, self.config.opi_type, p)] = len(self.opinion_dict)

        self.relation_dict = {'O': 0, 'yes': 1}
        return self.polarity_dict, self.aspect_dict, self.opinion_dict, self.entity_dict, self.relation_dict

    def get_neighbor(self, utterance_spans, replies, max_length, speaker_ids, thread_nums):
        """"
        utterance_spans: [(1, 30), (33, 53), (56, 95), (98, 145), (148, 229), (232, 253), (256, 290), (293, 315)]
        replies: [-1, 0, 1, 2, 0, 4, 0, 6]
        max_length: sum(sentence_length)    sentence_length: [32, 23, 42, 50, 84, 24, 37, 25]
        speaker_ids: [0, 1, 2, 1, 3, 0, 3, 0]
        thread_nums: [1, 3, 2, 2]
        """
        # utterance_mask = np.zeros([max_length, max_length], dtype=int)
        reply_mask = np.eye(max_length, dtype=int)  # max_length行max_length列的单位矩阵
        for i, w in enumerate(replies):
            s1, e1 = utterance_spans[i]  # 当前句
            s0, e0 = utterance_spans[w + (1 if w == -1 else 0)]  # 被回复的句子
            reply_mask[s0: e0 + 1, s1: e1 + 1] = 1  # 表示i回复w
            reply_mask[s1: e1 + 1, s0: e0 + 1] = 1  # 双向关联, 实际上主要反映了回复关系的对称性表示
            reply_mask[s0: e0 + 1, s0: e0 + 1] = 1  # 每个发言的每个字符都与自己相关联
            reply_mask[s1: e1 + 1, s1: e1 + 1] = 1  # 每个发言的每个字符都与自己相关联

        speaker_mask = np.zeros([max_length, max_length], dtype=int)
        for i, idx in enumerate(speaker_ids):
            # utterance_ids = [j for j, w in enumerate(speaker_ids) if w == idx]
            s0, e0 = utterance_spans[i]
            for j, idx1 in enumerate(speaker_ids):
                if idx != idx1: continue
                s1, e1 = utterance_spans[j]
                speaker_mask[s0: e0 + 1, s1: e1 + 1] = 1
                speaker_mask[s1: e1 + 1, s0: e0 + 1] = 1
                speaker_mask[s0: e0 + 1, s0: e0 + 1] = 1
                speaker_mask[s1: e1 + 1, s1: e1 + 1] = 1

        thread_mask = np.eye(max_length, dtype=int)
        thread_ends = accumulate(thread_nums)
        thread_spans = [(w - z, w) for w, z in zip(thread_ends, thread_nums)]
        for i, (s, e) in enumerate(thread_spans):
            if i == 0: continue
            head_start, head_end = utterance_spans[0]
            thread_mask[head_start: head_end + 1, head_start: head_end + 1] = 1
            for j in range(s, e):
                s0, e0 = utterance_spans[j]
                thread_mask[s0:e0 + 1, head_start:head_end + 1] = 1  # 线程内发言与对话首条发言的关联
                thread_mask[head_start:head_end + 1, s0:e0 + 1] = 1
                for k in range(s, e):
                    s1, e1 = utterance_spans[k]
                    thread_mask[s0 + 1: e0, s1 + 1: e1] = 1  # 线程内发言之间的子区间关联
                    thread_mask[s1 + 1: e1, s1 + 1: e1] = 1  # 线程内发言之间的相互关联
                    thread_mask[s0 + 1: e0, s0 + 1: e0] = 1
                    thread_mask[s1 + 1: e1, s0 + 1: e0] = 1

        return reply_mask.tolist(), speaker_mask.tolist(), thread_mask.tolist()

    def find_utterance_index(self, replies, sentence_lengths, printf=False):
        """
        根据回复关系和对话长度获取对话线索
        replies: [-1, 0, 1, 2, 0, 4, 0, 6]  sentence_length: [32, 23, 42, 50, 84, 24, 37, 25]
        """
        # 构建对话线索
        utterance_collections = [i for i, w in enumerate(replies) if w == 0]  # utterance_collections: [1, 4, 6] 找到所有"新对话"的起始位置
        if len(utterance_collections) > 1:  # 存在多个对话线索
            zero_index = utterance_collections[1]  # 设置初始的“新对话”起始位置
            for i in range(len(replies)):
                if i < zero_index: continue  # 跳过所有在当前“新对话”起始位置之前的索引
                if replies[i] == 0:
                    zero_index = i  # 更新“新对话”的起始位置
                replies[i] = (i - zero_index)  # 计算每个回复相对于最近的“新对话”起始位置的偏移量
        if printf:
            print(f'replies: {replies}')  # [-1, 0, 1, 2, 0, 1, 0, 1]

        sentence_index = [w + 1 for w in replies]  # 基于回复关系构建句子索引
        if printf:
            print(f'sentence_index: {sentence_index}')  # [0, 1, 2, 3, 1, 2, 1, 2]

        utterance_index = [[w] * z for w, z in zip(sentence_index, sentence_lengths)]  # 根据每个句子的长度复制对话索引
        if printf:
            print(f'utterance_index: {utterance_index}')
        utterance_index = [w for line in utterance_index for w in line]
        if printf:
            print(f'utterance_index: {utterance_index}')

        # 初始化词汇索引列表
        token_index = [list(range(sentence_lengths[0]))]  # [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]
        lens = len(token_index[0])  # 获取第一个句子的长度
        if printf:
            print(f'token_index: {token_index}\nlens: {lens}')
        for i, w in enumerate(sentence_lengths):
            if i == 0: continue  # 跳过第一个句子
            if sentence_index[i] == 1:  # 如果遇到新对话的开始
                distance = lens  # 更新起始距离
            token_index += [list(range(distance, distance + w))]  # 为当前句子生成词汇索引
            distance += w
        token_index = [w for line in token_index for w in line]
        if printf:
            print(f'token_index: {token_index}')

        utterance_collections = np.split(sentence_index, utterance_collections)
        if printf:
            print(f'utterance_collections: {utterance_collections}')  # [array([0]), array([1, 2, 3]), array([1, 2]), array([1, 2])]

        thread_nums = list(map(len, utterance_collections))  # [1, 3, 2, 2]
        thread_ranges = [0] + list(accumulate(thread_nums))  # [0, 1, 4, 6, 8]
        thread_lengths = [sum(sentence_lengths[thread_ranges[i]:thread_ranges[i + 1]]) for i in
                          range(len(thread_ranges) - 1)]  # [32, 115, 108, 62]

        return utterance_index, token_index, thread_lengths, thread_nums

    def get_pair(self, full_triplets):
        pairs = {'ao': set(), }
        # pairs = {'ta': set(), 'ao': set(), 'to': set()}
        for i in range(len(full_triplets)):
            sa, ea, so, eo, p = full_triplets[i][:5]
            # if st != -1 and sa != -1:
            #     pairs['ta'].add((st, et, sa, ea))
            #
            # if st != -1 and so != -1:
            #     pairs['to'].add((st, et, so, eo))

            if sa != -1 and eo != -1:
                pairs['ao'].add((sa, ea, so, eo))

        return pairs

    def transfer_polarity(self, pol):
        res = {'pos': 'pos', 'neg': 'neg'}
        return res.get(pol, 'other')

    def read_data(self, mode):
        """
        Read a JSON file, tokenize using BERT, and realign the indices of the original elements according to the tokenization results.
        """

        path = os.path.join(self.config.json_path, '{}.json'.format(mode))

        if not os.path.exists(path):
            raise FileNotFoundError('File {} not found! Please check your input and data path.'.format(path))

        content = json.load(open(path, 'r', encoding='utf-8'))
        res = []
        for line in tqdm(content, desc='Processing dialogues for {}'.format(mode)):
            new_dialog = self.parse_dialogue(line, mode)
            res.append(new_dialog)
        return res

    def check_text(self, tokenized_text, source_text):
        if self.config.bert_path in ['roberta-large', 'roberta-base']:
            t0 = tokenized_text.lower()
            roberta_chars = 'â ī ¥ Ġ ð ł ĺ ħ Ł ŀ į Ŀ Į ĵ © ĵ ĳ ¶ ã'.split()
            unused = [self.config.unk, '##']
            if self.config.bert_path in ['roberta-large', 'roberta-base']:
                unused += roberta_chars
            for u in unused:
                t0 = t0.replace(u.lower(), '')

            t1 = source_text.replace(' ', '').lower()
            for k in self.config.unkown_tokens:
                t1 = t1.replace(k, '')
            if self.config.bert_path in ['roberta-large', 'roberta-base']:
                t1 = t1.replace('×', '').replace('≥', '')
            if t0 != t1:
                logger.info(t1 + '||' + t1)
                logger.info(tokenized_text + '||' + source_text)
                t2 = t0
                for u in unused:
                    t2 = t2.replace(u, '')
                raise AssertionError("--{}-- != --{}--".format(t0, t1))
            return t0 == t1

        t0 = tokenized_text.replace('##', '').replace(self.config.unk, '').lower()
        t1 = source_text.replace(' ', '').lower()
        for k in self.config.unkown_tokens:
            t1 = t1.replace(k, '')
        if t0 != t1:
            logger.info(t1 + '||' + t1)
            logger.info(tokenized_text + '||' + source_text)
            raise AssertionError("{} != {}".format(t0, t1))
        return t0 == t1

    def parse_dialogue(self, dialogue, mode):
        # get the list of sentences in the dialogue
        sentences = dialogue['sentences']

        # align_index_with_list: align the index of the original elements according to the tokenization results
        new_sentences, pieces2words = self.align_index_with_list(sentences)
        # 统计位置偏移量 key对应实际位置, value对应索引列表 这是由于 BERT在进行分词时, '...'会被分成['.', '.', '.'], 但其对应的位置索引应该是一个
        word2pieces = defaultdict(list)
        for p, w in enumerate(pieces2words):
            word2pieces[w].append(p)

        dialogue['pieces2words'] = pieces2words
        dialogue['sentences'] = new_sentences

        # get target, aspect and opinion respectively, and align to the new index

        if mode != 'train':
            return dialogue
        aspects, opinions = [dialogue[w] for w in ['aspects', 'opinions']]
        # 更新位置索引, 更新后为经BERT分词后的索引位置
        # targets = [(word2pieces[x][0], word2pieces[y - 1][-1] + 1, z) for x, y, z in targets]
        # print(f'sentences: {sentences}\nword2pieces: {word2pieces}\naspects: {aspects}')
        aspects = [(word2pieces[x][0], word2pieces[y - 1][-1] + 1, z) for x, y, z in aspects]
        opinions = [(word2pieces[x][0], word2pieces[y - 1][-1] + 1, z, self.transfer_polarity(w)) for x, y, z, w in
                    opinions]

        # Put the elements into the dialogue object after converting the elements to the new index
        dialogue['aspects'], dialogue['opinions'] = aspects, opinions

        # Flatten the two-dimensional list and put the entire dialogue in a list
        news = [w for line in new_sentences for w in line]

        # Confirm the index again
        # for ts, te, t_t in targets:
        #     assert self.check_text(''.join(news[ts:te]), t_t)
        for ts, te, t_t in aspects:
            assert self.check_text(''.join(news[ts:te]), t_t)
        for ts, te, t_t, _ in opinions:
            assert self.check_text(''.join(news[ts:te]), t_t)

        triplets = []

        # polarity transfer and index transfer
        # [24, 25, 17, 20, 25, 26, "pos", "iPhone", "处理器", "好"]
        for a_s, a_e, o_s, o_e, polarity, a_t, o_t in dialogue['triplets']:
            polarity = self.transfer_polarity(polarity)
            nas, nos = [word2pieces[w][0] if w != -1 else -1 for w in [a_s, o_s]]
            nae, noe = [word2pieces[w - 1][-1] + 1 if w != -1 else -1 for w in [a_e, o_e]]
            # self.check_text(''.join(news[nts:nte]), t_t)
            self.check_text(''.join(news[nas:nae]), a_t) or nas == -1
            if not self.check_text(''.join(news[nos:noe]), o_t) and nos != -1:
                logger.info(''.join(news[nos:noe]) + '||' + o_t)
            self.check_text(''.join(news[nos:noe]), o_t) or nos == -1

            triplets.append((nas, nae, nos, noe, polarity, a_t, o_t))
        dialogue['triplets'] = triplets
        return dialogue

    def align_index_with_list(self, sentences):
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
            tokens = [self.tokenizer.tokenize(w) for w in sentence]
            cur_line = []
            for token in tokens:
                for piece in token:
                    pieces2word.append(word_num)
                word_num += 1
                cur_line += token
            all_pieces.append(cur_line)

        return all_pieces, pieces2word

    def align_index(self, sentences):
        res, char2token = [], {}
        source_lens, token_lens = 0, 0
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            if self.config.bert_path in ['roberta-large', 'bert-base-uncased']:
                c2t, tokens = self.alignment_roberta(sentence, tokens)
            else:
                c2t, tokens = self.alignment(sentence, tokens)
            res.append(tokens)
            for k, v in c2t.items():
                char2token[k + source_lens] = v + token_lens
            source_lens, token_lens = source_lens + len(sentence) + 1, token_lens + len(tokens)

        return res, char2token

    def alignment(self, source_sequence, tokenized_sequence: List[str], align_type: str = 'one2many') -> Dict:
        """[summary]
        # this is a function that to align sequcences  that before tokenized and after.
        Parameters
        ----------
        source_sequence : [type]
            this is the original sequence, whose type either can be str or list
        tokenized_sequence : List[str]
            this is the tokenized sequcen, which is a list of tokens.
        index_type : str, optional, default: str
            this indicate whether source_sequence is str or list, by default 'str'
        align_type : str, optional, default: one2many
            there may be several kinds of tokenizer style, 
            one2many: one word in source sequence can be split into multiple tokens 
            many2one: many word in source sequence will be merged into one token
            many2many: both contains one2many and many2one in a sequence, this is the most complicated situation.
        
        useage:
        source_sequence = "Here, we investigate the structure and dissociation process of interfacial water"
        tokenized_sequence = ['here', ',', 'we', 'investigate', 'the', 'structure', 'and', 'di', '##sso', '##ciation', 'process', 'of', 'inter', '##fa', '##cial', 'water']
        char2token = alignment(source_sequence, tokenized_sequence)
        print(char2token)
        for c, t in char2token.items():
            print(source_sequence[c], tokenized_sequence[t])
        """
        char2token = {}
        if isinstance(source_sequence, str) and align_type == 'one2many':
            source_sequence = source_sequence.lower()
            i, j = 0, 0
            while i < len(source_sequence) and j < len(tokenized_sequence):
                cur_token, length = tokenized_sequence[j], len(tokenized_sequence[j])
                if source_sequence[i] == ' ':
                    i += 1
                elif source_sequence[i: i + length] == cur_token:
                    for k in range(length):
                        char2token[i + k] = j
                    i, j = i + length, j + 1
                elif tokenized_sequence[j] == self.config.unk:
                    lens = 1
                    if j + 1 == len(tokenized_sequence):
                        lens = len(source_sequence) - i
                    else:
                        while i + lens < len(source_sequence):
                            if source_sequence[i + lens] == tokenized_sequence[j + 1].strip('#')[0] or \
                                    tokenized_sequence[j + 1] == self.config.unk:
                                break
                            lens += 1
                    new_token = self.repack_unknow(source_sequence[i:i + lens])
                    tokenized_sequence = tokenized_sequence[:j] + new_token + tokenized_sequence[j + 1:]
                    if tokenized_sequence[j] == self.config.unk:
                        char2token[i] = j
                        i += 1
                        j += 1
                else:
                    assert tokenized_sequence[j].startswith('#')
                    length = len(tokenized_sequence[j].lstrip('#'))
                    assert source_sequence[i: i + length] == tokenized_sequence[j].lstrip('#')
                    for k in range(length):
                        char2token[i + k] = j
                    i, j = i + length, j + 1
        return char2token, tokenized_sequence

    def alignment_roberta(self, source_sequence, tokenized_sequence: List[str]) -> Dict:
        # For English dataset
        char2token = {}
        if isinstance(source_sequence, str):
            source_sequence = source_sequence.lower()
            i, j = 0, 0
            while i < len(source_sequence) and j < len(tokenized_sequence):
                cur_token, length = tokenized_sequence[j], len(tokenized_sequence[j].strip('Ġ'))
                if source_sequence[i] == ' ':
                    i += 1
                elif source_sequence[i: i + length].lower() == cur_token.strip('Ġ').lower():
                    for k in range(length):
                        char2token[i + k] = j
                    i, j = i + length, j + 1
                elif tokenized_sequence[j] == self.config.unk:
                    lens = 1
                    if j + 1 == len(tokenized_sequence):
                        lens = len(source_sequence) - i
                    else:
                        while i + lens < len(source_sequence):
                            if source_sequence[i + lens] == tokenized_sequence[j + 1].strip('#')[0] or \
                                    tokenized_sequence[j + 1] == self.config.unk:
                                if tokenized_sequence[j + 1].strip('#')[0] == 'i' and j + 1 < len(
                                        tokenized_sequence) and len(tokenized_sequence[j + 1].strip()) > 1:
                                    if i + lens + 1 < len(source_sequence) and source_sequence[i + lens + 1] == \
                                            tokenized_sequence[j + 1].strip('#')[1]:
                                        break
                                else:
                                    break
                            lens += 1
                    new_token = self.repack_unknow(source_sequence[i:i + lens])
                    tokenized_sequence = tokenized_sequence[:j] + new_token + tokenized_sequence[j + 1:]
                    if tokenized_sequence[j] == self.config.unk:
                        char2token[i] = j
                        i += 1
                        j += 1
                else:
                    assert tokenized_sequence[j].startswith('#')
                    length = len(tokenized_sequence[j].lstrip('#'))
                    assert source_sequence[i: i + length] == tokenized_sequence[j].lstrip('#')
                    for k in range(length):
                        char2token[i + k] = j
                    i, j = i + length, j + 1
        return char2token, tokenized_sequence

    def repack_unknow(self, source_sequence):
        '''
        # sentence='🍎12💩', Bert can't recognize two contiguous emojis, so it recognizes the whole as '[UNK]'
        # We need to manually split it, recognize the words that are not in the bert vocabulary as UNK, 
        and let BERT re-segment the parts that can be recognized, such as numbers
        # The above example processing result is: ['[UNK]', '12', '[UNK]']
        '''
        lst = list(re.finditer('|'.join(self.config.unkown_tokens), source_sequence))
        start, i = 0, 0
        new_tokens = []
        while i < len(lst):
            s, e = lst[i].span()
            if start < s:
                token = self.tokenizer.tokenize(source_sequence[start:s])
                new_tokens += token
                start = s
            else:
                new_tokens.append(self.config.unk)
                start = e
            i += 1
        if start < len(source_sequence):
            token = self.tokenizer.tokenize(source_sequence[start:])
            new_tokens += token
        return new_tokens

    def transform2indices(self, dataset, mode='train', printf=False):
        res = []
        for document in dataset:
            sentences, speakers, replies, pieces2words = [document[w] for w in
                                                          ['sentences', 'speakers', 'replies', 'pieces2words']]
            if printf:
                print(f'sentences: {sentences}\nspeakers: {speakers}\nreplies: {replies}\npieces2words: {pieces2words}')

            if mode == 'train':
                triplets, aspects, opinions = [document[w] for w in ['triplets', 'aspects', 'opinions']]
            doc_id = document['doc_id']

            # sentence_length = list(map(lambda x : len(x) + 2, sentences))
            sentence_length = list(map(lambda x: len(x) + 2, sentences))  # 句子长度 加的2是 cls 和 sep
            if printf:
                print(f'sentence_length: {sentence_length}')

            # token2sentid = [[i] * len(w) for i, w in enumerate(sentences)]
            token2sentid = [[i] * len(w) for i, w in enumerate(sentences)]  # segment id
            token2sentid = [w for line in token2sentid for w in line]  # segment id flatten
            if printf:
                print(f'token2sentid: {token2sentid}')

            token2speaker = [[11] + [w] * len(z) + [10] for w, z in zip(speakers, sentences)]
            token2speaker = [w for line in token2speaker for w in line]  # speaker id 加上 cls 和 sep
            if printf:
                print(f'token2speaker: {token2speaker}')

            # New token indices (with CLS and SEP) to old token indices (without CLS and SEP)
            new2old = {}  # 新的token索引
            cur_len = 0
            for i in range(len(sentence_length)):
                for j in range(sentence_length[i]):
                    if j == 0 or j == sentence_length[i] - 1:
                        new2old[len(new2old)] = -1
                    else:
                        new2old[len(new2old)] = cur_len
                        cur_len += 1
            if printf:
                print(f'new2old: {new2old}')
            tokens = [[self.config.cls] + w + [self.config.sep] for w in sentences]
            if printf:
                print(f'tokens: {tokens}')

            # sentence_ids of each token (new token)
            nsentence_ids = [[i] * len(w) for i, w in enumerate(tokens)]
            nsentence_ids = [w for line in nsentence_ids for w in line]
            if printf:
                print(f'nsentence_ids: {nsentence_ids}')

            flatten_tokens = [w for line in tokens for w in line]
            sentence_end = [i - 1 for i, w in enumerate(flatten_tokens) if w == self.config.sep]  # 每个句子结束的位置  不含sep
            sentence_start = [i + 1 for i, w in enumerate(flatten_tokens) if w == self.config.cls]  # 每个句子开始的位置 不含cls
            if printf:
                print(f'flatten_tokens: {flatten_tokens}\nsentence_end: {sentence_end}\nsentence_start: {sentence_start}')

            utterance_spans = list(zip(sentence_start, sentence_end))
            if printf:
                print(f'utterance_spans: {utterance_spans}')
            utterance_index, token_index, thread_length, thread_nums = self.find_utterance_index(
                replies, sentence_length, printf=False
            )
            reply_mask, speaker_masks, thread_masks = self.get_neighbor(
                utterance_spans, replies, sum(sentence_length), speakers, thread_nums
            )

            input_ids = list(map(self.tokenizer.convert_tokens_to_ids, tokens))

            input_masks = [[1] * len(w) for w in input_ids]
            input_segments = [[0] * len(w) for w in input_ids]
            if printf:
                print(f'input_ids: {input_ids}\ninput_masks: {input_masks}\ninput_segments: {input_segments}')

            new_triplets, pairs, entity_lists, relation_lists, polarity_lists = [], [], [], [], []

            if mode == 'train':
                # 索引对齐, 句子id*2 加1是指当前句中的cls
                # targets = [(s + 2 * token2sentid[s] + 1, e + 2 * token2sentid[s]) for s, e, t in targets]
                aspects = [(s + 2 * token2sentid[s] + 1, e + 2 * token2sentid[s]) for s, e, t in aspects]
                opinions = [(s + 2 * token2sentid[s] + 1, e + 2 * token2sentid[s]) for s, e, t, p in opinions]
                opinions = list(set(opinions))

                full_triplets = []
                # t_s-> target_start, t_e-> target_end, etc.
                for a_s, a_e, o_s, o_e, polarity, a_t, o_t in triplets:
                    new_index = lambda start, end: (-1, -1) if start == -1 else (
                    start + 2 * token2sentid[start] + 1, end + 2 * token2sentid[start])
                    # t_s, t_e = new_index(t_s, t_e)
                    a_s, a_e = new_index(a_s, a_e)
                    o_s, o_e = new_index(o_s, o_e)
                    line = (a_s, a_e, o_s, o_e, self.polarity_dict[polarity])
                    full_triplets.append(line)
                    if all(w != -1 for w in [a_s, o_s]):
                        new_triplets.append(line)

                relation_lists = self.wordpair.encode_relation(full_triplets)  # 获取网格标记关系 tgt h2h
                pairs = self.get_pair(full_triplets)

                # target_lists = self.wordpair.encode_entity(targets, 'ENT-T')
                aspect_lists = self.wordpair.encode_entity(aspects, 'ENT-A')
                opinion_lists = self.wordpair.encode_entity(opinions, 'ENT-O')

                entity_lists = aspect_lists + opinion_lists
                polarity_lists = self.wordpair.encode_polarity(new_triplets)
                if printf:
                    print(f'entity_list: {entity_lists}\nrelation_lists:{relation_lists}\npolarity_list: {polarity_lists}')

            res.append((doc_id, input_ids, input_masks, input_segments, sentence_length, nsentence_ids, utterance_index,
                        token_index,
                        thread_length, token2speaker, reply_mask, speaker_masks, thread_masks, pieces2words, new2old,
                        new_triplets, pairs, entity_lists, relation_lists, polarity_lists))

        return res

    def forward(self):
        modes = 'train valid test'
        datasets = {}

        for mode in modes.split():
            data = self.read_data(mode)
            datasets[mode] = data

        label_dict = self.get_dict()

        res = {}
        for mode in modes.split():
            res[mode] = self.transform2indices(datasets[mode], mode, printf=False)

        res['label_dict'] = label_dict
        return res


if __name__ == '__main__':
    import yaml
    from attrdict import AttrDict
    import argparse
    from src.common import set_seed, ScoreManager, update_config

    config = AttrDict(yaml.load(open('./config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader))
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
    config.bert_path = "../chinese-roberta-wwm-ext"
    config.json_path = '../data/dataset/jsons_zh/'
    preprocessor = Preprocessor(config)
    data = preprocessor.forward()
