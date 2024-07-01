#%%
from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

list_of_documents = [
    Document(page_content="foo", metadata=dict(page=1)),
    Document(page_content="bar", metadata=dict(page=1)),
    Document(page_content="foo", metadata=dict(page=2)),
    Document(page_content="barbar", metadata=dict(page=2)),
    Document(page_content="foo", metadata=dict(page=3)),
    Document(page_content="bar burr", metadata=dict(page=3)),
    Document(page_content="foo", metadata=dict(page=4)),
    Document(page_content="bar bruh", metadata=dict(page=4))
]
db = FAISS.from_documents(list_of_documents, embeddings)
#%%
# Get oprigin Docs
origin_docs = []
for i, doc_id in db.index_to_docstore_id.items():
    origin_docs.append(db.docstore.search(doc_id))
origin_docs

#%%
# Vectors
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
# %%
db = FAISS.load_local("index", OpenAIEmbeddings())
db
# %%
vectors = []
for i, doc_id in db.index_to_docstore_id.items():
    vectors.append(db.index.reconstruct(i))
vectors

# %%
import numpy as np
v = np.random.random(768)
db.similarity_search_by_vector(v, 1)

#%%
results_with_scores = db.similarity_search_with_score("foo")
for doc, score in results_with_scores:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
# %%
## Add filter with page
results_with_scores = db.similarity_search_with_score("foo", filter=dict(page=1))
for doc, score in results_with_scores:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
# %%

results = db.max_marginal_relevance_search("foo", filter=dict(page=1))
for doc in results:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}")
# %%

results = db.similarity_search("foo", filter=dict(page=1), k=2, fetch_k=4)
for doc in results:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}")
# %%
