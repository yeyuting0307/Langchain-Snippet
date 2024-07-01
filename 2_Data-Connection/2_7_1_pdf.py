#%%
# pip install pypdf
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/PISA2022.pdf")
pages = loader.load_and_split()
pages
# %%
import os
import getpass
from dotenv import load_dotenv
load_dotenv()

# %%
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
docs = faiss_index.similarity_search("PISA是什麼", k=1)
for doc in docs:
    print(str(doc.metadata["page"]) + ":", doc.page_content[:300])
