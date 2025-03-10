#!/usr/bin/env python


import os
import yaml
import argparse

import torch
import sys
import numpy as np

from attrdict import AttrDict
from torch.optim import AdamW
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

from src.utils import MyDataLoader, RelationMetric 
from src.model import BertWordPair
from src.common import set_seed, ScoreManager, update_config
from tqdm import tqdm
from loguru import logger

import jsonlines


def save_jsonl(filepath, content):
    with jsonlines.open(filepath, mode='a') as writer:
        writer.write(content)


def watch_pred_label(matrix):
    matrix_nonzero_list = []
    for idx in range(len(matrix)):
        nonzero_list = [col.item() for row in matrix[idx] for col in row if col != 0]
        matrix_nonzero_list.append(nonzero_list)
    return matrix_nonzero_list


def save_step(data, pred_ent_matrix, pred_rel_matrix, pred_pol_matrix):
    sentence_masks, full_masks = data['sentence_masks'], data['full_masks']
    pred_ent_matrix_ = pred_ent_matrix.argmax(-1) * data['sentence_masks']
    pred_rel_matrix_ = pred_rel_matrix.argmax(-1) * data['full_masks']
    pred_pol_matrix_ = pred_pol_matrix.argmax(-1) * data['full_masks']
    token2sents = data['token2sents'].tolist()
    new2old = data['new2old']
    pieces2words = data['pieces2words']
    doc_id = data['doc_id']

    pred_ent_matrix_nonzero_list = watch_pred_label(pred_ent_matrix_)
    pred_rel_matrix_nonzero_list = watch_pred_label(pred_rel_matrix_)
    pred_pol_matrix_nonzero_list = watch_pred_label(pred_pol_matrix_)

    sample = {
        'pred_ent_matrix': pred_ent_matrix.tolist(), 'pred_rel_matrix': pred_rel_matrix.tolist(),
        'pred_pol_matrix': pred_pol_matrix.tolist(), 'sentence_masks': sentence_masks.tolist(),
        'full_masks': full_masks.tolist(), 'token2sents': token2sents, 'new2old': new2old, 'pieces2words': pieces2words,
        'doc_id': doc_id
    }

    sample_ = {
        'pred_ent_matrix_nonzero_list': pred_ent_matrix_nonzero_list,
        'pred_rel_matrix_nonzero_list': pred_rel_matrix_nonzero_list,
        'pred_pol_matrix_nonzero_list': pred_pol_matrix_nonzero_list
    }
    save_jsonl('./save_step0712_01.jsonl', sample)


class Main:
    def __init__(self, args):
        config = AttrDict(yaml.load(open('src/config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader))

        for k, v in vars(args).items():
            setattr(config, k, v)
        
        config = update_config(config)

        set_seed(config.seed)
        if not os.path.exists(config.target_dir):
            os.makedirs(config.target_dir)

        config.device = torch.device('cuda:{}'.format(config.cuda_index) if torch.cuda.is_available() else 'cpu')
        self.config = config
    
    def train_iter(self):
        self.model.train()
        train_data = tqdm(self.trainLoader, total=self.trainLoader.__len__(), file=sys.stdout)
        losses = []
        for i, data in enumerate(train_data):
            loss, _ = self.model(**data)
            losses.append([w.tolist() for w in loss])
            sum(loss).backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()

            description = "Epoch {}, entity loss:{:.4f}, rel loss: {:.4f}, pol loss: {:.4f}".format(self.global_epoch, *np.mean(losses, 0))
            train_data.set_description(description)

    def evaluate_iter(self, dataLoader=None):

        self.model.eval()
        dataLoader = self.validLoader if dataLoader is None else dataLoader
        dataiter = tqdm(dataLoader, total=dataLoader.__len__(), file=sys.stdout)
        for i, data in enumerate(dataiter):
            with torch.no_grad():
                _, (pred_ent_matrix, pred_rel_matrix, pred_pol_matrix) = self.model(**data)
                self.relation_metric.add_instance(data, pred_ent_matrix, pred_rel_matrix, pred_pol_matrix)
                # if i == 0:
                #     save_step(data, pred_ent_matrix, pred_rel_matrix, pred_pol_matrix)

    def test(self):
        PATH = os.path.join(self.config.target_dir, "{}_{}.pth.tar").format(self.config.lang, self.best_iter)
        self.model.load_state_dict(torch.load(PATH, map_location=self.config.device)['model'])
        self.model.eval()

        self.evaluate_iter(self.testLoader)
        result = self.relation_metric.compute('test')

        score, res = result
        logger.info("Evaluate on test set, micro-F1 score: {:.4f}%".format(score * 100))
        print(res)

    def train(self):
        best_score, best_iter = 0, 0
        for epoch in range(self.config.epoch_size):
            self.global_epoch = epoch
            self.train_iter()
            self.evaluate_iter()

            score, res = self.relation_metric.compute()

            self.score_manager.add_instance(score, res)
            logger.info("Epoch {}, micro-F1 score: {:.4f}%".format(epoch, score * 100))
            print(res)
            # raise TypeError('!!! Not Implemented !!!')

            if score > best_score:
                best_score, best_iter = score, epoch

                torch.save({'epoch': epoch, 'model': self.model.cpu().state_dict(), 'best_score': best_score},
                           os.path.join(self.config.target_dir,  "{}_{}.pth.tar".format(self.config.lang, best_iter)))
                self.model.to(self.config.device)
            elif epoch - best_iter > self.config.patience:
                print("Not upgrade for {} steps, early stopping...".format(self.config.patience))
                break
            self.model.to(self.config.device)
        
        self.best_iter = best_iter
    
    def load_param(self):
        
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.config.weight_decay}, 
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0}]

        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=float(self.config.bert_lr),
                               eps=float(self.config.adam_epsilon))
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config.warmup_steps,
                                                         num_training_steps=self.config.epoch_size * self.trainLoader.__len__())
   
    def forward(self):
        self.config.bert_path = 'chinese-roberta-wwm-ext'
        self.config.json_path = 'data/dataset/jsons_zh/'
        self.config.batch_size = 2

        self.trainLoader, self.validLoader, self.testLoader, config = MyDataLoader(self.config).getdata()
        self.model = BertWordPair(self.config).to(config.device)

        self.score_manager = ScoreManager()
        self.relation_metric = RelationMetric(self.config)
        self.load_param()

        logger.info("Start training...")
        # self.best_iter = 7
        self.train()
        logger.info("Training finished..., best epoch is {}...".format(self.best_iter))
        self.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang', type=str, default='zh', choices=['zh', 'en'], help='language selection')
    parser.add_argument('-b', '--bert_lr', type=float, default=1e-5, help='learning rate for BERT layers')
    parser.add_argument('-c', '--cuda_index', type=int, default=0, help='CUDA index')
    parser.add_argument('-s', '--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()
    main = Main(args)
    main.forward()
