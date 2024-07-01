#%%
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI()

# %%
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat([
    HumanMessage(content="Translate this sentence from English to French: I love programming.")
])
# %%
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="I love programming.")
]
chat(messages)
# %%
batch_messages = [
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love programming.")
    ],
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love artificial intelligence.")
    ],
]
result = chat.generate(batch_messages) # genereate for batch_messsages 
result
# %%
len(result)