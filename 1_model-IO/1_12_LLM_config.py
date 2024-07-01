#%%

from langchain.llms import OpenAI
from langchain.llms.loading import load_llm
# %%
llm = load_llm("1_12_llm.json")
# %%

llm = OpenAI(model='text-davinci-002')
llm.save('1_12_saved_llm.json')

# %%
