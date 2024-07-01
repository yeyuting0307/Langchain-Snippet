#%%
from langchain.schema.runnable import RunnableParallel
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
model = ChatOpenAI()

chain1 = (ChatPromptTemplate.from_template("tell me a joke about {topic}").with_config(run_name = "prompt1", tags = ["p1"])
    | model.with_config(run_name = "model1",  tags = ["m1"])
)
chain2 = (
    ChatPromptTemplate.from_template("write a short (2 line) poem about {topic}").with_config(run_name = "prompt2",  tags = ["p2"])
    | model.with_config(run_name = "model2", tags = ["m2"])
)
combined = RunnableParallel(joke=chain1, poem=chain2).with_config(run_name = "combined", tags = ["combined_tag"])
# %%
# combined.invoke({"topic": "bears"})
# %%

async for chunk in combined.astream_log(
    {"topic": "bears"}, 
    include_names=["combined", "prompt2", "model1", "model2"],
    diff=False
):
    print("-" * 40)
    print(chunk.state.get('logs').keys())
    print(chunk.state.get('logs', {}).get('model1'))

# %%

# %%
