#%%
from langchain.embeddings import VertexAIEmbeddings

embeddings = VertexAIEmbeddings()
text = "This is a test document."


query_result = embeddings.embed_query(text)
doc_result = embeddings.embed_documents([text])

doc_result
