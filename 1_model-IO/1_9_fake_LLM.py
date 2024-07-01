# %%
''' 不需要實際把LLM載入測試, 用FakseListLLM並且直接把response寫好就好了'''
from langchain.llms.fake import FakeListLLM

# %%
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType


# %%
tools = load_tools(["python_repl"])
# %%
responses = ["Action: Python REPL\nAction Input: print(2 + 2)", "Final Answer: 4"]
llm = FakeListLLM(responses=responses)

# %%
llm('hi')
# %%

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
# %%
agent.run("whats 2 + 2")
# %%
