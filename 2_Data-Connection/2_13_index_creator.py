#%%
# pip install chromadb
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings


# %%
from langchain.document_loaders import TextLoader
loader = TextLoader('./data/FLOP.txt', encoding='utf8')

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, 
    chunk_overlap=0
)

# docs = text_splitter.split_documents(loader.load())
# docs
# %%
from langchain.indexes import VectorstoreIndexCreator
embedding_model = OpenAIEmbeddings(
    model = "text-embedding-ada-002"
)

llm_model = ChatOpenAI(model_name='gpt-3.5-turbo-16k')
# llm_model = OpenAI(
#     model="text-davinci-003",
#     temperature=0
# )

index = VectorstoreIndexCreator(
    embedding = embedding_model,
    text_splitter = text_splitter,
).from_loaders([loader])

# %%
query = "公式呢"
index.query(query, llm=llm_model)

# %%
query = "公式呢"
index.query_with_sources(query, llm=llm_model)
# {'question': '公式呢',
#  'answer': '...略...',
#  'sources': './data/FLOP.txt'}

# %%
