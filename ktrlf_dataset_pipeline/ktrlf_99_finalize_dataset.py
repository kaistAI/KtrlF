import pickle
import copy
import json
import argparse


def main(input_path, output_path):
    with open(input_path, 'rb') as f:
        all_task = pickle.load(f)

    final_dataset_form = []
    for dic in all_task:
        new_dic = copy.deepcopy(dic)
        new_dic['data']['qa_pairs'] = [{'question': pair['question'], 'target_entities': pair['final_preds']} for pair in dic['data']['qa_pairs']]
        final_dataset_form.append(new_dic)
    
    with open(output_path, 'w') as f:
        for dic in final_dataset_form:
            f.write(json.dumps(dic) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default='./dump/3_query_filtering_merged.pickle')
    parser.add_argument("--output_path", type=str, default='./dump/ktrlf_dataset.jsonl')
    args = parser.parse_args()

    main(args.input_path, args.output_path)
