import pickle
import copy
import argparse
from collections import Counter
from ktrlf_2_1_target_selection_openai import clear_query_with_empty_target, clear_doc_with_emtpy_query
from utils.statistics import print_number_of_targets, print_number_of_mentions


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt4_input_path", type=str, default='./dump/2_target_selection_gpt4.pickle')
    parser.add_argument("--chatgpt_input_path", type=str, default='./dump/2_target_selection_chatgpt.pickle')
    parser.add_argument("--output_path", type=str, default='./dump/2_target_selection_merged.pickle')
    args = parser.parse_args()

    final_answer_key = 'final_preds'

    with open(args.gpt4_input_path, 'rb') as f:
        gpt4_all_task = pickle.load(f)
    print(f"[Num. of Targets] GPT4:")
    print(f"{print_number_of_targets(gpt4_all_task, 'gpt4_preds')}")
    print(f"[Num. of Mentions] GPT4:")
    print(f"{print_number_of_mentions(gpt4_all_task, 'gpt4_preds')}")

    
    with open(args.chatgpt_input_path, 'rb') as f:
        chatgpt_all_task = pickle.load(f)
    print(f"[Num. of Targets] ChatGPT:")
    print(f"{print_number_of_targets(chatgpt_all_task, 'chatgpt_preds')}")
    print(f"[Num. of Mentions] ChatGPT:")
    print(f"{print_number_of_mentions(chatgpt_all_task, 'chatgpt_preds')}")


    assert len(gpt4_all_task)==len(chatgpt_all_task), f"{len(gpt4_all_task)}, {len(chatgpt_all_task)}"
    for to_dic, from_dic in zip(chatgpt_all_task, gpt4_all_task):
        assert len(to_dic['data']['qa_pairs'])==len(from_dic['data']['qa_pairs']), f"{to_dic['data']['qa_pairs']}, {from_dic['data']['qa_pairs']}"
        for to_pair, from_pair in zip(to_dic['data']['qa_pairs'], from_dic['data']['qa_pairs']):
            to_pair['gpt4_preds'] = copy.deepcopy(from_pair['gpt4_preds'])
            _all_entity_set_in_doc = set([entity_dic['entity'] for entity_dic in to_dic['data']['entity_info']])

            llama2_preds = [ent for ent in to_pair['_llama2_query_generation_preds'] if ent in _all_entity_set_in_doc]
            to_pair['_llama2_query_generation_preds'] = llama2_preds
            
            gpt4_preds = [ent for ent in from_pair['gpt4_preds'] if ent in _all_entity_set_in_doc]
            to_pair['gpt4_preds'] = gpt4_preds

            to_pair[final_answer_key] = [k for k,v in Counter(to_pair['chatgpt_preds']+gpt4_preds+llama2_preds).items() if v>=2]
    

    print(f"[Num. of Targets] Final:")
    print(f"{print_number_of_targets(chatgpt_all_task, final_answer_key)}")

    chatgpt_all_task = clear_query_with_empty_target(chatgpt_all_task, final_answer_key)
    chatgpt_all_task = clear_doc_with_emtpy_query(chatgpt_all_task)
    print(f"[Num. of Targets] After:")
    print(f"{print_number_of_targets(chatgpt_all_task, final_answer_key)}")
    print(f"[Num. of Mentions] After:")
    print(f"{print_number_of_mentions(chatgpt_all_task, final_answer_key)}")

    with open(args.output_path, 'wb') as f:
        pickle.dump(chatgpt_all_task, f)
