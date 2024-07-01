#%%
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader('./data/PISA2022.pdf')
pdf = loader.load_and_split()
pdf
#%%
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = 500,
    chunk_overlap  = 200,
    length_function = len
)

docs = text_splitter.split_documents(pdf)
docs
# %%
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()


# %%
from langchain.vectorstores import Chroma
db = Chroma.from_documents(docs, embeddings)
retriever = db.as_retriever()
retriever

# %%
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k')
llm

# %%
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever
)

# %%
query = 'PISA能吃嗎'
qa.run(query)

# %%
# VectorstoreIndexCreator等價於上述邏輯
from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=embeddings,
    text_splitter=text_splitter,
).from_loaders([loader])

query = 'PISA能吃嗎'
index.query(query, llm=llm)
# %%

