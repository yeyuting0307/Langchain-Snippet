#%%
# ================= Switch LLM =================

from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import ConfigurableField
from langchain.chat_models.vertexai import ChatVertexAI
# %%
llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)\
    .configurable_alternatives(
        ConfigurableField(id="llm"),
        default_key="openai",
        gpt4=ChatOpenAI(model="gpt-4"),
        palm2 = ChatVertexAI(model_name= "chat-bison")
    )
prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | llm
# %%
chain.invoke({"topic": "llamas"})

# %%
chain.with_config(configurable={"llm": "gpt4"})\
    .invoke({"topic": "llamas"})
# %%
chain.with_config(configurable={"llm": "palm2"})\
    .invoke({"topic": "llamas"})
# %%

# =================  Switch Prompts =================
prompt = PromptTemplate.from_template(
    "Tell me a joke about {topic}"
).configurable_alternatives(
    ConfigurableField(id="prompt"), 
    default_key="joke",
    poem=PromptTemplate.from_template("Write a short poem about {topic}"),
)
chain = prompt | llm

#%%
chain.invoke({"topic": "bears"})

# %%
chain.with_config(configurable={"prompt": "poem"}).invoke({"topic": "bears"})
# %%
# =================  Switch LLM and Prompt =================
chain.with_config(
    configurable={"prompt": "poem", "llm": "palm2"}
).invoke(
    {"topic": "bears"}
)
# %%
