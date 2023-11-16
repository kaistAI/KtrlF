import pandas as pd
from collections import defaultdict


def print_number_of_targets(all_task, answer_key):
    num_of_targets_per_query = []
    for dic in all_task:
        num_of_targets_per_query += [len(qa[answer_key]) for qa in dic['data']['qa_pairs']]
    print()
    print(f"\tper query: {pd.DataFrame(num_of_targets_per_query).describe()}")
    print(f"\tTotal: {sum(num_of_targets_per_query)}")

def print_number_of_mentions(all_task, answer_key):
    num_mentions_per_query = []
    for dic in all_task:
        mention_counter = defaultdict(int)
        for entity_dic in dic['data']['entity_info']:
            mention_counter[entity_dic['entity']] += 1
        
        for qa in dic['data']['qa_pairs']:
            _answer_mentions = sum([mention_counter[pred] for pred in qa[answer_key]])
            num_mentions_per_query.append(_answer_mentions)
    print()
    print(f"\tper query: {pd.DataFrame(num_mentions_per_query).describe()}")
    print(f"\tTotal: {sum(num_mentions_per_query)}")