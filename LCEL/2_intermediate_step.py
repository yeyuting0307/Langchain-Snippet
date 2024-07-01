#%%
from typing import Any, Dict, List, Optional, TypedDict
class LogEntry(TypedDict):
    id: str
    """ID of the sub-run."""
    name: str
    """Name of the object being run."""
    type: str
    """Type of the object being run, eg. prompt, chain, llm, etc."""
    tags: List[str]
    """List of tags for the run."""
    metadata: Dict[str, Any]
    """Key-value pairs of metadata for the run."""
    start_time: str
    """ISO-8601 timestamp of when the run started."""

    streamed_output_str: List[str]
    """List of LLM tokens streamed by this run, if applicable."""
    final_output: Optional[Any]
    """Final output of this run.
    Only available after the run has finished successfully."""
    end_time: Optional[str]
    """ISO-8601 timestamp of when the run ended.
    Only available after the run has finished."""


class RunState(TypedDict):
    id: str
    """ID of the run."""
    streamed_output: List[Any]
    """List of output chunks streamed by Runnable.stream()"""
    final_output: Optional[Any]
    """Final output of the run, usually the result of aggregating (`+`) streamed_output.
    Only available after the run has finished successfully."""

    logs: Dict[str, LogEntry]
    """Map of run names to sub-runs. If filters were supplied, this list will
    contain only the runs that matched the filters."""

#%%
from typing import Dict

# mypy ./LCEL/intermediate_step.py  

# 創建一個 Dict，鍵是字符串，值是整數
my_dict: Dict[str, int] = {
    "apple": 5,
    "banana": 10
}

# 這是允許的，因為值仍然是整數
my_dict["orange"] = 3
my_dict["cherry"] = "red" 


# %%
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import FAISS

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

model = ChatOpenAI()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# prompt.with_config(prompt_name = "Prompt")
# model.with_config(model_name = "Model")

vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

retrieval_chain = (
    {
        "context": retriever.with_config(run_name="Docs"),
        "question": RunnablePassthrough().with_config(run_name="Pass"),
    }
    | prompt.with_config(run_name = "Prompt")
    | model.with_config(run_name = "Model")
    | StrOutputParser()
)

async for chunk in retrieval_chain.astream_log(
    "where did harrison work?", 
    include_names=["Prompt", "Docs", "Model", "Pass"]
):
    print("-" * 40)
    print(chunk)
# %%
async for chunk in retrieval_chain.astream_log(
    "where did harrison work?", 
    include_names=["Prompt", "Docs", "Model", "Pass"],
    diff=False
):
    print("-" * 40)
    print(chunk)
# %%
