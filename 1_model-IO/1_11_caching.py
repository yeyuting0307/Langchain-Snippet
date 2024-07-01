#%%
import os
from dotenv import load_dotenv
load_dotenv('../.env') # OPENAI_API_KEY

import langchain
from langchain.llms import OpenAI

# To make the caching really obvious, lets use a slower model.
llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)
# %%
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()

from pandas import to_datetime
# The first time, it is not yet in cache, so it should take longer
S = to_datetime('today')
llm.predict("Tell me a joke")
E = to_datetime('today')
print('first time =>',E-S)
# %%
# The second time it is, so it goes faster
S2 = to_datetime('today')
llm.predict("Tell me a joke")
E2 = to_datetime('today')
print('second time =>', E2-S2)
# %%

# %%
# =================== SQLiteCache ===================
# We can do the same thing with a SQLite cache
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
S = to_datetime('today')
llm.predict("Tell me a joke")
E = to_datetime('today')
print('first time =>',E-S)
# %%
# The second time it is, so it goes faster
S2 = to_datetime('today')
llm.predict("Tell me a joke")
E2 = to_datetime('today')
print('second time =>', E2-S2)
# %%

llm = OpenAI(model_name="text-davinci-002")
no_cache_llm = OpenAI(model_name="text-davinci-002", cache=False)


# %%
# pip install pypdf
from langchain.document_loaders import PyPDFLoader
from langchain.chains.mapreduce import MapReduceChain


loader = PyPDFLoader("../data/東杰常見問題&專業QA.pdf")
docs = loader.load_and_split()

print(docs, '\n pages:', len(docs))
# %%
from langchain.chains.summarize import load_summarize_chain
# map use cache, reduce not use cache # TODO:
chain = load_summarize_chain(llm, chain_type="map_reduce", reduce_llm=no_cache_llm)
chain.run(docs)
# %%
