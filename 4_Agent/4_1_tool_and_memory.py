#%%
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0)

# %%
from langchain.agents import tool

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

tools = [get_word_length]

#%%
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent
system_message = SystemMessage(
    content="You are very powerful assistant, but bad at calculating lengths of words.")
prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
prompt
# %%
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
agent

# %%
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor
# %%

agent_executor.run("how many letters in the word educa?")


# %%
# -------------- Memory --------------
from langchain.prompts import MessagesPlaceholder

MEMORY_KEY = "chat_history"
prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=MEMORY_KEY)]
)
prompt

# %%
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key=MEMORY_KEY, return_messages=True)


# %%
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
agent_executor.run("how many letters in the word educa?")
agent_executor.run("is that a real word?")
# %%
