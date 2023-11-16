import re
import time
import json
import pickle
import argparse
import datasets
from pathlib import Path
from vllm import LLM, SamplingParams
from utils.model_prompt_templates import prompt_template
from utils.statistics import print_number_of_mentions, print_number_of_targets


#### Get Answers
def _run_query_generation(llm, sampling_params, entity_set_in_all_c4, output_filepath, num_queries=10):
    system_prompt = f"""
You are an expert of query generation for entity search.

You must follow this requirements.
Requirements:
- Your task is to generate queries that retrieve entities in a given list.
- The generated query must be able to list the multiple entities.
- The answers must be countable.
- The answers have to be entities.
- Make sure your questions are unambiguous and based on facts rather than temporal information.
- Do not specify the number in a query.
- Do not start with 'What' in a query.
- Do not start with 'Which' in a query.
- Do not include an expression in your query that tells it to find from a given list.

The example is as below.

Generate 4 queries from the following list and extract subset list.
Candidate List:
[Apple, Microsoft, Samsung Electronics, Alphabet, AT&T, Amazon, Verizon Communications, China Mobile, Walt Disney, Facebook, Alibaba, Intel, Softbank, IBM, Tencent Holdings, Nippon Telegraph & Tel, Cisco Systems, Oracle, Deutsche Telkom, Taiwan Semiconductor, KDDi , HP, Legend Holding, Lenovo Group, ebay]

Query:
1. IT companies in Computer Hardware industry
=> Apple, HP, Legend Holding, Lenovo Group
2. Find all IT companies that have software as main business.
=> Microsoft, Oracle
3. Companies that is known for retail service
=> Amazon, Alibaba, ebay
4. Name all IT companies that have license in USA
=> Apple, Microsoft, Alphabet, AT&T, Amazon, Verizon Communications, Walt Disney, Facebook, Intel, IBM, Cisco Systems, Oracle, HP, ebay
"""

    prompts = [
        prompt_template(
            model_symbol='llama2',
            system_prompt=system_prompt.strip(),
            user_message=f"Generate {num_queries} queries from the following list and extract subset list.\nCandidate List: {entity_set}\n\nQuery:"
        ) for entity_set in entity_set_in_all_c4
    ]
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    with open(output_filepath, 'w') as f:
        for output, input_ in zip(outputs, prompts):
            f.write(json.dumps([
                input_, {'prompt': output.prompt, 'output': output.outputs[0].text}
            ])+'\n')

def _parse_gpt_output(ds, entity_info_dic, output_filepath, guarantee_num_sents=10):
    generated_output_list = []
    with open(output_filepath) as f:
        for line in f:
            input_, output_dic = json.loads(line)
            generated_output_list.append(output_dic['output'])

    _skip_count = 0

    assert len(ds)==len(generated_output_list), f'{len(ds)}, {len(generated_output_list)}'

    all_task = []
    for doc_id, target_text, qa_str in zip(ds['url'], ds['text'], generated_output_list):
        questions = []; answers = []
        _entity_set = set([dic['entity'] for dic in entity_info_dic[doc_id]])

        # Get N qa pairs
        for line in qa_str.split('\n'):
            line = line.strip()
            match = re.search(r'^\d+\.(.*)', line) # question
            if match:
                questions.append(match.group(1).strip())
            elif line[:2] == "=>":
                a_str = line[2:]
                _answers = [a.strip() for a in list(filter(None,a_str.split(',')))]
                _answers = [a[:-1] if a.endswith('.') and a[:-1] in _entity_set else a for a in _answers]
                answers.append(_answers)

        if len(questions) < guarantee_num_sents:
            print(f"[Error {_skip_count}] doc_idx: {doc_id}, num_q: {len(questions)}, num_ans: {len(answers)}, q: {questions}, ans: {answers}")
            _skip_count += 1
            continue
        
        guarantee_questions = questions[:guarantee_num_sents]
        guarantee_answers = answers[:guarantee_num_sents] + [[] for _ in range(guarantee_num_sents-len(answers))]

        data = {'qa_pairs': []}
        for qa_i, (question,_answers) in enumerate(zip(guarantee_questions,guarantee_answers)):
            data['qa_pairs'].append({
                'question': question,
                '_llama2_query_generation_preds': _answers
            })

        data['target_text'] = target_text
        data['entity_info'] = entity_info_dic[doc_id]

        all_task.append({
            'id':doc_id, 
            'data':data
        })
    
    return all_task

def generate_query(llm, sampling_params, ds, entity_info_dic):
    Path("./dump/.gpt_format").mkdir(parents=True, exist_ok=True)

    _timestamp = str(int(time.time()))
    output_format_path = f'./dump/.gpt_format/_llama2_query_generation_{_timestamp}_output_format.jsonl'

    entity_set_in_all_c4 = [set([dic['entity'] for dic in entity_info_dic[url]]) for url in ds['url']]

    _run_query_generation(llm, sampling_params, entity_set_in_all_c4, output_format_path)
    all_task = _parse_gpt_output(ds, entity_info_dic, output_format_path)
    return all_task

def heuristic_normalize(str):
    str = str.strip()
    str = str[:-1].strip() if str.endswith('.') or str.endswith(':') else str
    return str


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampled_c4_path", type=str, default="./dump/0_sampled_c4")
    parser.add_argument("--entity_path", type=str, default="./dump/c4_realnewslike_entity_info_list.pickle")
    parser.add_argument("--output_path", type=str, default="./dump/1_query_generation.pickle")
    args = parser.parse_args()
    
    ds = datasets.load_from_disk(args.sampled_c4_path)
    with open(args.entity_path, 'rb') as f:
        entity_info_dic = pickle.load(f)

    sampling_params = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=1024)
    llm = LLM(model="meta-llama/Llama-2-70b-chat-hf", tensor_parallel_size=4)

    print('load finish')
    all_task = generate_query(llm, sampling_params, ds, entity_info_dic)

    for dic in all_task:
        for qa in dic['data']['qa_pairs']:
            qa['question'] = heuristic_normalize(qa['question'])

    print(f"[Num. of Targets]")
    print(f"{print_number_of_targets(all_task, '_llama2_query_generation_preds')}")
    print(f"[Num. of Mentions]")
    print(f"{print_number_of_mentions(all_task, '_llama2_query_generation_preds')}")

    with open(args.output_path, 'wb') as f:
        pickle.dump(all_task, f)
