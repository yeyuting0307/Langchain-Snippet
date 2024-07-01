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

# pip install langchainhub
from langchain.runnables.hub import HubRunnable

prompt = HubRunnable("rlm/rag-prompt").configurable_fields(
    owner_repo_commit=ConfigurableField(
        id="hub_commit",
        name="Hub Commit",
        description="The Hub commit to pull from",
    )
)
# %%
prompt.invoke({"question": "foo", "context": "bar"})


# %%
prompt.with_config(
    configurable={"hub_commit": "rlm/rag-prompt-llama"}
    ).invoke(
        {"question": "foo", "context": "bar"}
    )
# %%

# %%
