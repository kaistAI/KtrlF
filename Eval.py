import sys
import logging
import pickle
from tqdm import tqdm 
import json
from typing import Dict, Type, List, Callable, Iterable, Tuple,Optional, Union

from functools import partial
import random
import re
from functools import reduce
import copy
from collections import Counter, defaultdict 
import warnings
import string 
import difflib
import numpy as np
from collections import Counter 
import warnings
import re 
import string 
import difflib
import glob
import ast 
import types


import os
import bz2
from typing import List, Set
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json 
import pickle 
import argparse 

class Metric:
    """
    Interface for a metric.
    """
    def compute_one(self, pred, gold):
        """
        Computes metrics for one example.
        You must implement this.
        Args:
            pred: single prediction.
            gold: corresponding ground truth.
        """
        raise NotImplementedError()

    def __call__(self, pred, gold):
        return self.forward(pred, gold)

    def forward(self, pred: list, gold: list) -> dict:
        """
        Computes metric over list of predictions and ground truths and returns a dictionary of scores.
        Args:
            pred: list of predictions.
            gold: corresponding ground truths.
        """
        metrics = defaultdict(list)
        for pi, gi in zip(pred, gold):
            m = self.compute_one(pi, gi)
            for k, v in m.items():
                metrics[k].append(v)
        return {k: sum(v)/len(v) for k, v in metrics.items()}

    def lmap(self, f, *x):
        """list(map(f, x))"""
        return list(map(f, *x))

    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        def rid_of_specials(text):
            text = text.replace("<extra_id_0>", "")
            text = text.replace("<extra_id_1>", "")
            return text

        return rid_of_specials(white_space_fix(
            remove_articles(remove_punc(lower(s)))))

class MultiSpan(Metric):
    # (set -> list version)
    def list_multi_span_evaluate(self,preds: List[List], golds: List[ List]):
        assert len(preds) == len(golds)
        
        preds = [self.lmap(lambda x: self.normalize_answer(x), pred) for pred in preds] 
        golds =  [self.lmap(lambda x: self.normalize_answer(x), gold) for gold in golds] 
        
        # Evaluate
        em_p, em_r, em_f = self.list_compute_scores(golds, preds, eval_type='em')  # type: ignore
        overlap_p, overlap_r, overlap_f = self.list_compute_scores(golds, preds, eval_type='overlap')  # type: ignore
        result = {'list_exact_match_precision': 100 * em_p,
                'list_exact_match_recall': 100 * em_r,
                'list_exact_match_f1': 100 * em_f,
                'list_overlap_precision': 100 * overlap_p,
                'list_overlap_recall': 100 * overlap_r,
                'list_overlap_f1': 100 * overlap_f}
        return result

    def multi_span_evaluate(self,preds: List[List], golds: List[ List]):
        assert len(preds) == len(golds)
        
        preds = [set(self.lmap(lambda x: self.normalize_answer(x), pred)) for pred in preds] 
        golds =  [set(self.lmap(lambda x: self.normalize_answer(x), gold)) for gold in golds] 
        
        # Evaluate
        em_p, em_r, em_f = self.compute_scores(golds, preds, eval_type='em')  # type: ignore
        overlap_p, overlap_r, overlap_f = self.compute_scores(golds, preds, eval_type='overlap')  # type: ignore
        result = {'exact_match_precision': 100 * em_p,
                'exact_match_recall': 100 * em_r,
                'exact_match_f1': 100 * em_f,
                'overlap_precision': 100 * overlap_p,
                'overlap_recall': 100 * overlap_r,
                'overlap_f1': 100 * overlap_f}
        return result


    # (set -> list version)
    def list_compute_scores(self,golds: List[List], preds: List[List], eval_type: str = 'em', average: str = 'micro'):
        """Compute precision, recall and exact match (or f1) metrics.

        :param golds: dictionary of gold XX
        :param preds: dictionary of predictions
        :param eval_type: Evaluation type. Can be either "em" or "overlap".
        """
        nb_gold = 0
        nb_pred = 0
        nb_correct = 0
        nb_correct_p = 0
        nb_correct_r = 0
        
        for gold, pred in zip(golds, preds):
            nb_gold += max(len(gold), 1)
            nb_pred += max(len(pred), 1)
            if eval_type == 'em':
                if len(gold) == 0 and len(pred) == 0:
                    # Exact match no answer case
                    nb_correct += 1
                else:
                    # Exact match comparison between two sets
                    common = Counter(gold) & Counter(pred)
                    nb_correct = sum(common.values())
                    
            else:
                p_score, r_score = self.count_overlap(gold, pred)
                nb_correct_p += p_score
                nb_correct_r += r_score

        if eval_type == 'em':
            p = nb_correct / nb_pred if nb_pred > 0 else 0
            r = nb_correct / nb_gold if nb_gold > 0 else 0
        else:
            p = nb_correct_p / nb_pred if nb_pred > 0 else 0
            r = nb_correct_r / nb_gold if nb_gold > 0 else 0

        f = 2 * p * r / (p + r) if p + r > 0 else 0

        return p, r, f

    def compute_scores(self,golds: List[Set], preds: List[Set], eval_type: str = 'em', average: str = 'micro'):
        """Compute precision, recall and exact match (or f1) metrics.

        :param golds: dictionary of gold XX
        :param preds: dictionary of predictions
        :param eval_type: Evaluation type. Can be either "em" or "overlap".
        """
        nb_gold = 0
        nb_pred = 0
        nb_correct = 0
        nb_correct_p = 0
        nb_correct_r = 0
        
        for gold, pred in zip(golds, preds):
            nb_gold += max(len(gold), 1)
            nb_pred += max(len(pred), 1)
            if eval_type == 'em':
                if len(gold) == 0 and len(pred) == 0:
                    # Exact match no answer case
                    nb_correct += 1
                else:
                    # Exact match comparison between two sets
                    nb_correct += len(gold.intersection(pred))
            else:
                p_score, r_score = self.count_overlap(gold, pred)
                nb_correct_p += p_score
                nb_correct_r += r_score

        if eval_type == 'em':
            p = nb_correct / nb_pred if nb_pred > 0 else 0
            r = nb_correct / nb_gold if nb_gold > 0 else 0
        else:
            p = nb_correct_p / nb_pred if nb_pred > 0 else 0
            r = nb_correct_r / nb_gold if nb_gold > 0 else 0

        f = 2 * p * r / (p + r) if p + r > 0 else 0

        return p, r, f

    def count_overlap(self,gold: set, pred: set):
        """Count the overlap of the gold answer and the predicted answer.

        :param gold: Set of gold answers
        :param pred: Set of predicted answers
        """
        # Correct no answer prediction
        if len(gold) == 0 and (len(pred) == 0 or pred == {""}):
            return 1, 1

        # Incorrect no answer prediction
        elif len(gold) == 0 or (len(pred) == 0 or pred == {""}):
            return 0, 0

        # NOTE: Since it is possible to return multiple spans it is not clear which spans from pred should be compared to
        #       each span in gold. So all are compared and the highest precision and recall are taken.
        p_scores = np.zeros((len(gold), len(pred)))
        r_scores = np.zeros((len(gold), len(pred)))
        for i, gold_str in enumerate(gold):
            for j, pred_str in enumerate(pred):
                seq_matcher = difflib.SequenceMatcher(None, gold_str, pred_str)
                _, _, longest_len = seq_matcher.find_longest_match(0, len(gold_str), 0, len(pred_str))
                p_scores[i][j] = longest_len/len(pred_str) if longest_len > 0 else 0
                r_scores[i][j] = longest_len/len(gold_str) if longest_len > 0 else 0

        sort_index = r_scores.argsort()
        prev_idx_list= []
        score=[]
        for row, pair in enumerate(sort_index[:,::-1]): # best score per index 
            for idx in pair:
                if idx not in prev_idx_list:
                    score.append(r_scores[row,idx])
                    prev_idx_list.append(idx)
                    break

        r_score = sum(score)

        p_score = sum(np.max(p_scores, axis=0)) 
        
        return p_score, r_score


def evaluate(evaluator,predicted_data):
    score_key = ['em_top1', 'em_top100', 'f1_top1', 'f1_top100', 'rd_topk'] # 'se_pos', 

    total_score_dict=defaultdict(lambda: defaultdict(list))

    outputs=defaultdict(list)
    list_outputs=defaultdict(list)
    metrics=defaultdict(list)

    for key,pred_list in predicted_data.items():
        group_score_dict=defaultdict(list)
        list_group_score_dict=defaultdict(list)
        
        mins = defaultdict(lambda: defaultdict(list))
        avgs = defaultdict(lambda: defaultdict(list))

        for v in pred_list:       
            answer= v['answer']
        
            prediction= v['prediction']            
            results = evaluator.multi_span_evaluate([prediction], [answer])
            outputs['all'].append(results)

            list_results = evaluator.list_multi_span_evaluate([prediction], [answer])
            list_outputs['all'].append(list_results)
            
            group_score_dict['all'].append(results)
            list_group_score_dict['all'].append(list_results)

        '''
        calculate per group
        '''
        temp_dict=defaultdict(list)
        for topk, pairs in group_score_dict.items(): # key: topk, value: List(dict)
            for pair in pairs:
                for name,instance_score in pair.items():
                    total_score_dict[name][topk].append(instance_score) # key: score_name, value: Dict(key:topk, value: List(score))
                    temp_dict[name].append(instance_score)
            
            for name, v in temp_dict.items():
                metrics['cluster_{}_min_{}'.format(name, topk)].append(min(v))
                metrics['cluster_{}_avg_{}'.format(name, topk)].append(sum(v) / len(v))

        temp_dict=defaultdict(list)
        for topk, pairs in list_group_score_dict.items(): # key: topk, value: List(dict)
            for pair in pairs:
                for name,instance_score in pair.items():
                    name = 'list_' + name
                    total_score_dict[name][topk].append(instance_score) # key: score_name, value: Dict(key:topk, value: List(score))
                    temp_dict[name].append(instance_score)
            
            for name, v in temp_dict.items():
                metrics['cluster_{}_min_{}'.format(name, topk)].append(min(v))
                metrics['cluster_{}_avg_{}'.format(name, topk)].append(sum(v) / len(v))


    print('len(outputs):',len(outputs),len(outputs['all']))
    print('len(list_outputs):',len(list_outputs),len(list_outputs['all']))


    avg_em_prec = {topk: np.mean([pair['exact_match_precision'] for pair in x]) for topk, x in outputs.items()}
    avg_em_recall = {topk: np.mean([pair['exact_match_recall'] for pair in x]) for topk, x in outputs.items()}
    avg_em_f1 = {topk: np.mean([pair['exact_match_f1'] for pair in x]) for topk, x in outputs.items()}

    avg_overlap_prec = {topk: np.mean([pair['overlap_precision'] for pair in x]) for topk, x in outputs.items()}
    avg_overlap_recall = {topk: np.mean([pair['overlap_recall'] for pair in x]) for topk, x in outputs.items()}
    avg_overlap_f1 = {topk: np.mean([pair['overlap_f1'] for pair in x]) for topk, x in outputs.items()}

    print(f"batch_em_precison:",avg_em_prec)
    print(f"batch_em_recall:",avg_em_recall)
    print(f"batch_em_f1:",avg_em_f1)
    print(f"batch_overlap_prec:",avg_overlap_prec)
    print(f"batch_overlap_recall:",avg_overlap_recall)
    print(f"batch_overlap_f1:",avg_overlap_f1)


    avg_em_prec = {topk: np.mean([pair['list_exact_match_precision'] for pair in x]) for topk, x in list_outputs.items()}
    avg_em_recall = {topk: np.mean([pair['list_exact_match_recall'] for pair in x]) for topk, x in list_outputs.items()}
    avg_em_f1 = {topk: np.mean([pair['list_exact_match_f1'] for pair in x]) for topk, x in list_outputs.items()}

    avg_overlap_prec = {topk: np.mean([pair['list_overlap_precision'] for pair in x]) for topk, x in list_outputs.items()}
    avg_overlap_recall = {topk: np.mean([pair['list_overlap_recall'] for pair in x]) for topk, x in list_outputs.items()}
    avg_overlap_f1 = {topk: np.mean([pair['list_overlap_f1'] for pair in x]) for topk, x in list_outputs.items()}

    print("##### list version:\n")
    print(f"batch_em_precison:",avg_em_prec)
    print(f"batch_em_recall:",avg_em_recall)
    print(f"batch_em_f1:",avg_em_f1)
    print(f"batch_overlap_prec:",avg_overlap_prec)
    print(f"batch_overlap_recall:",avg_overlap_recall)
    print(f"batch_overlap_f1:",avg_overlap_f1)

    # Robustness score
    print("##### Robustness score")
    for k,v in metrics.items():
        print(f"{k}: {sum(v) / len(v)}")

    return {
        'avg_em_f1':avg_em_f1,
        'avg_overlap_f1':avg_overlap_f1,
        'cluster_list_exact_match_f1_min_all':metrics['cluster_list_exact_match_f1_min_all'],
        'cluster_list_overlap_f1_min_all':metrics['cluster_list_overlap_f1_min_all']
    }
