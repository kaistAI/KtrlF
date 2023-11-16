import collections
import argparse
from tqdm.auto import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import numpy as np
import pickle
import copy
from collections import Counter
import string
import re
from ktrlf_2_1_target_selection_openai import clear_doc_with_emtpy_query


def preprocess(qs, texts, q_text_pair_ids):
    _inputs = tokenizer(
        qs, 
        texts, 
        max_length=512, 
        stride=256,
        truncation="only_second",
        padding="max_length",
        return_overflowing_tokens=True,
        return_offsets_mapping=True
    )

    sample_map = _inputs.pop("overflow_to_sample_mapping")
    example_q_text_pair_ids = []
    exampe_texts = []
    for i in range(len(_inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_q_text_pair_ids.append(q_text_pair_ids[sample_idx])
        exampe_texts.append(texts[sample_idx])

        sequence_ids = _inputs.sequence_ids(i)
        offset = _inputs["offset_mapping"][i]
        _inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    _inputs["example_q_text_pair_id"] = example_q_text_pair_ids
    _inputs['example_text'] = exampe_texts
    return _inputs


def postprocess(_inputs, q_text_pair_ids, n_best=20, max_answer_length=30):
    example_to_features = collections.defaultdict(list)
    for idx, q_text_pair_id in enumerate(_inputs['example_q_text_pair_id']):
        example_to_features[q_text_pair_id].append(idx)

    _model_inputs = {k:torch.tensor(v, device=device) for k,v in _inputs.items() if k not in ['offset_mapping', 'example_q_text_pair_id', 'example_text']}
    with torch.no_grad():
        outputs = model(**_model_inputs)

    preds = []
    for q_text_pair_id in q_text_pair_ids:
        preds_per_q = []
        for feature_index in example_to_features[q_text_pair_id]:
            offsets = _inputs['offset_mapping'][feature_index]
            text = _inputs['example_text'][feature_index]
            
            start_logit = outputs.start_logits.cpu().numpy()[feature_index]
            end_logit = outputs.end_logits.cpu().numpy()[feature_index]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index:
                        continue
                    if offsets[end_index][1] - offsets[start_index][0] + 1 > max_answer_length:
                        continue

                    preds_per_q.append(
                        {
                            "text": text[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                    )

        preds_per_q = sorted(preds_per_q, key=lambda dic: -dic['logit_score'])
        preds_per_q = list(dict.fromkeys([ans['text'].strip() for ans in preds_per_q]))[:n_best]
        preds.append(preds_per_q)
        
    return preds

def scoring(all_task, answer_key, topk=5):
    def determine_correct(p, a, threshold=0.9):
        '''
        We set the threshold by referring to the human f1 performance mentioned in (Rajpurkar & Jia et al. 18).
        '''

        ''' Official evaluation snippet for v1.1 of the SQuAD dataset. '''
        def normalize_answer(s):
            """Lower text and remove punctuation, articles and extra whitespace."""
            def remove_articles(text):
                return re.sub(r'\b(a|an|the)\b', ' ', text)

            def white_space_fix(text):
                return ' '.join(text.split())

            def remove_punc(text):
                exclude = set(string.punctuation)
                return ''.join(ch for ch in text if ch not in exclude)

            def lower(text):
                return text.lower()

            return white_space_fix(remove_articles(remove_punc(lower(s))))

        def f1_score(prediction, ground_truth):
            prediction_tokens = normalize_answer(prediction).split()
            ground_truth_tokens = normalize_answer(ground_truth).split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                return 0
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1
        
        return f1_score(p, a) >= threshold
        
    _all_task = copy.deepcopy(all_task)
    for dic in _all_task:
        qa_pairs = dic['data']['qa_pairs']
        for qa_dic in qa_pairs:
            p_list = qa_dic['deberta_preds']
            a_list = qa_dic[answer_key]
        
            recall = 0
            for a in a_list:
                hit = sum([int(determine_correct(p,a)) for p in p_list]) > 0
                recall += 1 if hit else 0
            recall /= max([len(a_list),1])
        
            qa_dic[f'deberta_recall'] = recall
    return _all_task

def remove_easy_case(all_task, threshold=0.0):
    """ remove sample if any of the answers are predicted """
    _all_task = copy.deepcopy(all_task)
    for dic in _all_task:
        qa_pairs = dic['data']['qa_pairs']
        dic['data']['qa_pairs'] = [qa_dic for qa_dic in qa_pairs if qa_dic[f'deberta_recall'] <= threshold]
    return _all_task

def remove_false_answer_case(all_task, answer_key):
    _all_task = copy.deepcopy(all_task)
    for dic in _all_task:
        qa_pairs = dic['data']['qa_pairs']
        
        filtered_qa_pairs = []
        for qa_dic in qa_pairs:
            if len(qa_dic[answer_key])>0 and 'none' in qa_dic[answer_key][0].lower():
                continue
            filtered_qa_pairs.append(qa_dic)
            
        dic['data']['qa_pairs'] = filtered_qa_pairs
    return _all_task

def print_number_of_qa_pairs(all_task):
    return sum([len(dic['data']['qa_pairs']) for dic in all_task])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default='./dump/2_target_selection_merged.pickle')
    parser.add_argument("--output_path", type=str, default='./dump/3_query_filtering_merged.pickle')
    args = parser.parse_args()

    answer_key = 'final_preds'

    device = 'cuda'
    model_name = "deepset/deberta-v3-large-squad2"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(args.input_path, 'rb') as f:
        all_task = pickle.load(f)
    print(f"[Num. of Queries] Before: {print_number_of_qa_pairs(all_task)}")

    for dic in tqdm(all_task):
        doc_id = dic['id']
        data = dic['data']

        _qs = [qa_dic['question'] for qa_dic in data['qa_pairs']]
        _texts = [data["target_text"] for _ in range(len(data['qa_pairs']))]
        _q_text_pair_ids = [f"{doc_id}_{i}" for i in range(len(data['qa_pairs']))]

        _inputs = preprocess(_qs, _texts, _q_text_pair_ids)
        preds = postprocess(_inputs, _q_text_pair_ids)
        
        for i, p in enumerate(preds):
            data['qa_pairs'][i][f'deberta_preds'] = p

    all_task = scoring(all_task, answer_key)
    all_task = remove_easy_case(all_task)
    all_task = clear_doc_with_emtpy_query(all_task)
    print(f"[Num. of Queries] After removing easy case: {print_number_of_qa_pairs(all_task)}")
    all_task = remove_false_answer_case(all_task, answer_key)
    all_task = clear_doc_with_emtpy_query(all_task)
    print(f"[Num. of Queries] After removing false answer: {print_number_of_qa_pairs(all_task)}")

    with open(args.output_path, 'wb') as f:
        pickle.dump(all_task, f)
