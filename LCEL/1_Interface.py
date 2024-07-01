#%%
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic} in zh-tw Chinese")
chain = prompt | model
# %%
# =================== Input Schema ===================
# -- chain --
chain_schema = chain.input_schema.schema()
chain_schema.get("properties")

#%%
chain.invoke(input = {"topic": "颱風"})

# %%
# -- prompt --
prompt_schema = prompt.input_schema.schema()
prompt_schema.get("properties")

#%%
prompt.invoke(input = {"topic": "颱風"})

# %%
# -- model --
model_schema = model.input_schema.schema()
model_schema
#%%
# =================== Output Schema ===================
chain_output = chain.output_schema.schema()
chain_output.get("title")

# %%
promt_output = prompt.output_schema.schema()
promt_output.get("title")

# %%
# =================== chain method ===================
chain.invoke(input = {"topic": "颱風"})


# %%
for s in chain.stream({"topic": "颱風"}):
    print(s.content, end="", flush=True)


# %%

chain.batch([{"topic": "颱風"}, {"topic": "地震"}])

# %%
async for s in chain.astream({"topic": "颱風"}):
    print(s.content, end="", flush=True)

# %%
