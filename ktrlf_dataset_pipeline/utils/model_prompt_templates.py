def _llama2_prompt_template(system_prompt, user_message):
    return f"""
<s>[INST] <<SYS>>
{ system_prompt }
<</SYS>>

{ user_message } [/INST]""".strip()

def _vicuna_prompt_template(system_prompt, user_message):
    return f"""{system_prompt} USER: {user_message} ASSISTANT: """.strip()

def prompt_template(model_symbol, system_prompt, user_message):
    if 'llama2' in model_symbol.lower():
        return _llama2_prompt_template(system_prompt, user_message)
    elif 'vicuna' in model_symbol.lower():
        return _vicuna_prompt_template(system_prompt, user_message)
