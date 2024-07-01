#%%
import json
from pathlib import Path
from pprint import pprint

file_path='./data/fb_chat.json'
data = json.loads(Path(file_path).read_text())
data
# %%
# !pip install jq
from langchain.document_loaders import JSONLoader

loader = JSONLoader(
    file_path='./data/fb_chat.json',
    jq_schema='.messages[].content',
    text_content=False)

data = loader.load()

data
