#%%
# ----------------- Bad Chain -----------------
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

# First let's create a chain with a ChatModel
# We add in a string output parser here so the outputs between the two are the same type
from langchain.schema.output_parser import StrOutputParser

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a nice assistant who always includes a compliment in your response",
        ),
        ("human", "Why did the {animal} cross the road"),
    ]
)
# Here we're going to use a bad model name to easily create a chain that will error
chat_model = ChatOpenAI(model_name="gpt-fake")
bad_chain = chat_prompt | chat_model | StrOutputParser()

# %%
# ----------------- Good Chain -----------------
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

prompt_template = """Instructions: You should always include a compliment in your response.

Question: Why did the {animal} cross the road?"""
prompt = PromptTemplate.from_template(prompt_template)
llm = ChatOpenAI(model='gpt-4')
good_chain = prompt | llm

# %%
# ----------------- Chain with fallback -----------------

bad_chain.invoke({"animal": "turtle"})

#%%
good_chain.invoke({"animal": "turtle"})


#%%
chain = bad_chain.with_fallbacks([good_chain])

chain.invoke({"animal": "turtle"})
# %%
