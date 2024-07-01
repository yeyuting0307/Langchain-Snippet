#%%
import langchain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
import pandas as pd
df = pd.read_excel('car_info.xlsx')
data = df.drop(df.columns[0], axis=1).to_dict(orient='index')
texts = [",".join([f"{k}:{v}" for (k, v) in d.items()]) for _, d in data.items()]
texts[:3]

# %%

faiss_index = FAISS.from_texts(texts, OpenAIEmbeddings())
faiss_index
#%%
query = "我的volvo c30，想要換遠光燈，要買哪一款" 
k = 2 # number of documents to retrieve

sim_docs = faiss_index.similarity_search(query, k=k)
sim_reference = ""
for doc in sim_docs:
    sim_reference += doc.page_content 
    sim_reference += "\n"
print(sim_reference)
# %%
from ChatGPT import StreamChatGPT
prompt = f''' 根據以下資訊，請幫我釐清該問題想問的車種、車系，如果遇到```參考資料```的車種重複時，
如果"排氣量"或"年份"不為"NaN"、空值、"不分"，可以先繼續詢問是哪一種排氣量或年份：

範例1：
用戶的問題：我買的是TOYOTA的Sienta，要買哪一種車燈
參考資料
```
車種:豐田 TOYOTA,車種別名:豐田,車系:SIENTA,排氣量:1.8,年份:2017之後,近光燈:LED / H11,遠光燈:LED / 9005,霧燈:H16
車種:豐田 TOYOTA,車種別名:豐田,車系:SIENNA  休旅車,排氣量:3.5,年份:2009之後,近光燈:D2S/H11,遠光燈:9005,霧燈:H11
車種:豐田 TOYOTA,車種別名:豐田,車系:SIENNA  休旅車,排氣量:3.3,年份:2005之後,近光燈:D2R/9006,遠光燈:9005,霧燈:H3
```
你的回答：「請問您的TOYOTA車種SIENTA系列是否為修旅車？排氣量是1.8, 3.5, 3.3？購買年份是2017之後、2009之後、或者2005之後」

身為專業的汽車小幫手，可以先詢問車種、車系、排氣量、年份，釐清車輛資訊，再篩選出符合的近光燈、遠光燈、霧燈型號，並回覆用戶。


用戶的問題：{query}
參考資料：{sim_reference}
你的回答：「請問您的VOLVO車種C30系列的排氣量是2.0/2.4嗎？購買年份是2010之後還是2011之後？」
用戶的問題："我的2.4，2015年買的"
你的回答：
'''

#%%
answer = ""
for wp in StreamChatGPT(prompt):
    answer += wp
    print(answer, end = "\r")


