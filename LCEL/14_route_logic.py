#%%
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

# %%
chain = (
    PromptTemplate.from_template(
        """Given the user question below, classify it as either being about `LangChain`, `Anthropic`, or `Other`.
                                     
Do not respond with more than one word.

<question>
{question}
</question>

Classification:"""
    )
    | ChatOpenAI()
    | StrOutputParser()
)
# %%
chain.invoke({"question": "how do I call Anthropic?"})

# %%
langchain_chain = (
    PromptTemplate.from_template(
        """You are an expert in langchain. \
Always answer questions starting with "As Harrison Chase told me". \
Respond to the following question:

Question: {question}
Answer:"""
    )
    | ChatOpenAI()
)
anthropic_chain = (
    PromptTemplate.from_template(
        """You are an expert in anthropic. \
Always answer questions starting with "As Dario Amodei told me". \
Respond to the following question:

Question: {question}
Answer:"""
    )
    | ChatOpenAI()
)
general_chain = (
    PromptTemplate.from_template(
        """Respond to the following question:

Question: {question}
Answer:"""
    )
    | ChatOpenAI()
)


# %%
from langchain.schema.runnable import RunnableBranch

branch = RunnableBranch(
    (lambda x: "anthropic" in x["topic"].lower(), anthropic_chain),
    (lambda x: "langchain" in x["topic"].lower(), langchain_chain),
    general_chain,
)
# %%
full_chain = {"topic": chain, "question": lambda x: x["question"]} | branch
# %%
full_chain.invoke({"question": "how do I use langchain?"})
# %%

# ================== Custom Route ==================

def route(info):
    if "anthropic" in info["topic"].lower():
        return anthropic_chain
    elif "langchain" in info["topic"].lower():
        return langchain_chain
    else:
        return general_chain
# %%
from langchain.schema.runnable import RunnableLambda

full_chain = {
    "topic": chain, 
    "question": lambda x: x["question"]
    } | RunnableLambda(route)
# %%
full_chain.invoke({"question": "whats 2 + 2"})
# %%
