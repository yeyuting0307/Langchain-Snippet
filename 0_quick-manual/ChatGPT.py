#%%

import openai
import os
from dotenv import load_dotenv
load_dotenv()

# openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

def StreamChatGPT(question):

    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", None)
    ASSISTANT_PROMPT = os.getenv("ASSISTANT_PRE_PROMPT", None)
    ASSISTANT_POST_PROMPT = os.getenv("ASSISTANT_POST_PROMPT", None)
    INNER_PRE_PROMPT = os.getenv("INNER_PRE_PROMPT", None)
    INNER_POST_PROMPT = os.getenv("INNER_POST_PROMPT", None)

    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": f"{SYSTEM_PROMPT}"})
    if ASSISTANT_PROMPT:
        messages.append({"role": "assistant", "content": f"{ASSISTANT_PROMPT}"})

    if INNER_PRE_PROMPT:    
        question = f'''
        {INNER_PRE_PROMPT} 
        {question}
        '''

    if INNER_POST_PROMPT:
        question = f'''
        {question} 
        {INNER_POST_PROMPT}
        '''

    messages.append( {"role": "user", "content": f"{question}"})

    if ASSISTANT_POST_PROMPT:
        messages.append({"role": "assistant", "content": f"{ASSISTANT_POST_PROMPT}"})

    completion = openai.ChatCompletion.create(
        model=os.getenv("model", "gpt-3.5-turbo-16k") ,
        messages = messages,
        temperature = float(os.getenv("temperature", 0)),
        max_tokens = int(os.getenv("max_tokens", 8000)),
        stream = True
    )
    for wp in completion:
        content = wp.choices[0].delta.get("content")
        finish = wp.choices[0].finish_reason
        if content:
            yield content
        elif finish == "stop":
            break
        else:
            continue
# %%
# question = "3個廣告文案"
# answer = ""
# for wp in StreamChatGPT(question):
#     answer += wp
#     print(answer, end = "\r")
# %%
