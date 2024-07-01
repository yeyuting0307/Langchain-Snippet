#%%
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain.schema.runnable import (
    ConfigurableField,
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableMap,
)


#%%

# ================= Config Model Param =================
model = ChatOpenAI(temperature=0).configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="The temperature of the LLM",
    )
)
# %%
model.invoke("What is lisianthus")


#%%
model.with_config(configurable={"llm_temperature": 0.9})\
    .invoke("What is lisianthus")

# %%
