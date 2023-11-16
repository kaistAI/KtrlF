# Ktrl+F Dataset construction pipeline
This is official codebase for Ktrl+F Dataset construction pipeline.

## Step 1. Select Real News Articles
## Step 2. Generate Pairs of (Query, Targets)
## Step 3-1. Target Filtering
- GPT-3.5
```sh
python ktrlf_2_1_target_selection_openai.py
      --entity_evidence_cache_path ./dump/entity_evidence_dict.pickle
      --input_path ./dump/1_query_generation.pickle
      --openai_model_name gpt-3.5-turbo-0613
      --openai_request_url https://api.openai.com/v1/chat/completions
      --openai_api_key <openai-api-key>
      --output_path ./dump/2_target_selection_chatgpt.pickle
```
- GPT-4
```sh
python ktrlf_2_1_target_selection_openai.py
      --entity_evidence_cache_path ./dump/entity_evidence_dict.pickle
      --input_path ./dump/2_target_selection_chatgpt.pickle
      --openai_model_name gpt-4-0613
      --openai_request_url https://api.openai.com/v1/chat/completions
      --openai_api_key <openai-api-key>
      --output_path ./dump/2_target_selection_gpt4.pickle
```

- Merge all results

## Step 3-2. Query Filtering