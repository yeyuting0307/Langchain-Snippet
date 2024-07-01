#%%
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Load blog post
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()
data
#%%
# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)
splits, len(splits)
#%%
# VectorDB
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=splits, embedding=embedding)
vectordb
# %%
# LLM
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0)
llm
#%%
# Retriver
from langchain.retrievers.multi_query import MultiQueryRetriever
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm
)

retriever_from_llm
#%%
# Retriver - use genereated-multi-queries to get relevant_documents, if repeated, unique them
question = "What are the approaches to Task Decomposition?"
uniq_relev_docs = retriever_from_llm.get_relevant_documents(query=question)
uniq_relev_docs, len(uniq_relev_docs)

#%%
# ======= Supplying your own prompt =======

from typing import List
from langchain import LLMChain
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

# Output parser will split the LLM result into a list of queries

class LineList(BaseModel):
    lines: List[str] = Field(description="Lines of text")

class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    # override parse method
    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)

output_parser = LineListOutputParser()
output_parser
# %%

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions seperated by newlines.
    Original question: {question}""",
)

# Chain
llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)

# Other inputs
question = "What are the approaches to Task Decomposition?"


#%%
# Run
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    retriever = MultiQueryRetriever(
        retriever=vectordb.as_retriever(), llm_chain=llm_chain, parser_key="lines"
    )  # "lines" is the key (attribute name) of the parsed output

    # Results
    unique_docs = retriever.get_relevant_documents(
        query="What does the course say about regression?"
    )
    print(cb)

len(unique_docs) , unique_docs


# %%
unique_docs

# %%
