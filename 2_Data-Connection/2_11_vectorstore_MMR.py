#%%
# FIXME: http://localhost:6333
# pip install qdrant-client
from langchain.vectorstores import Qdrant

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

raw_documents = TextLoader('./data/FLOP.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
documents
#%%
embeddings = OpenAIEmbeddings()
# db = await Qdrant.afrom_documents(documents, embeddings, "http://localhost:6333")
db = Qdrant.from_documents(documents, embeddings)

#%%
query = "What did the president say about Ketanji Brown Jackson"
docs = await db.asimilarity_search(query)
print(docs[0].page_content)


embedding_vector = embeddings.embed_query(query)
docs = await db.asimilarity_search_by_vector(embedding_vector)

query = "What did the president say about Ketanji Brown Jackson"
found_docs = await qdrant.amax_marginal_relevance_search(query, k=2, fetch_k=10)
for i, doc in enumerate(found_docs):
    print(f"{i + 1}.", doc.page_content, "\n")

# %%
