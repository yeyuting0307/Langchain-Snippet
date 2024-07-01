#%%
# pip install lark pinecone-client
import os
import pinecone
from dotenv import load_dotenv
load_dotenv()

pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"])
try:
    pinecone.create_index("pinecone-index", dimension=1536)
    pinecone.list_indexes()
except Exception as e:
    print(e)


#%%
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import Pinecone, FAISS


docs = [
    Document(page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose", metadata={"year": 1993, "rating": 7.7, "genre": ["action", "science fiction"]}),
    Document(page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...", metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2}),
    Document(page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea", metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6}),
    Document(page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them", metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3}),
    Document(page_content="Toys come alive and have a blast doing so", metadata={"year": 1995, "genre": "animated"}),
    Document(page_content="Three men walk into the Zone, three men walk out of the Zone", metadata={"year": 1979, "rating": 9.9, "director": "Andrei Tarkovsky", "genre": ["science fiction", "thriller"], "rating": 9.9})
]

embeddings = OpenAIEmbeddings()

# FIXME: SelfQueryRetriever 尚不支援FAISS
vectorstore = Pinecone.from_documents(
    docs, embeddings, index_name="pinecone-index"
)

vectorstore
# %%
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info=[
    AttributeInfo(
        name="genre",
        description="The genre of the movie", 
        type="string or list[string]", 
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released", 
        type="integer", 
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director", 
        type="string", 
    ),
    AttributeInfo(
        name="rating",
        description="A 1-10 rating for the movie",
        type="float"
    ),
]

document_content_description = "Brief summary of a movie"
llm = ChatOpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm, 
    vectorstore, 
    document_content_description, 
    metadata_field_info, 
    verbose=True
)
# %%
retriever.get_relevant_documents("What are some movies about dinosaurs")


# %%

# This example only specifies a filter
retriever.get_relevant_documents("I want to watch a movie rated higher than 8.5")
# %%
retriever = SelfQueryRetriever.from_llm(
    llm, 
    vectorstore, 
    document_content_description, 
    metadata_field_info, 
    enable_limit=True,
    verbose=True
)
# semms to get limit in query
retriever.get_relevant_documents("What are two movies about dinosaurs")

# %%



