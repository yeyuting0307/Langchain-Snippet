#%%
import os
from dotenv import load_dotenv
load_dotenv('../.env') # OPENAI_API_KEY

# %%
# LLM : text-davinci-003
# ChatModel : gpt-3.5-turbo
from langchain.llms import OpenAI
llm = OpenAI(
    model_name = "text-davinci-003", 
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    # streaming=True ## Cannot stream results with multiple prompts.
)
# %%
# llm('Tell me 5 joke step by step')
llm.predict('Tell me 5 joke step by step')

# %%
llm_result = llm.generate(["Tell me a joke", "Tell me a poem"]*3)
len(llm_result.generations)
# %%
llm_result.generations[0]
# %%
llm_result.llm_output
# %%
