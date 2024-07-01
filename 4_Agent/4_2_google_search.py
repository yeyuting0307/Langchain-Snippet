#%%
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
# from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent

# %%
from dotenv import load_dotenv
load_dotenv('../.env')

#%%
import langchain
langchain.debug = True


#%%
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
import os

github_search = GoogleSearchAPIWrapper(
    google_cse_id=os.getenv("GOOGLE_CSE_ID_GITHUB"), 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

tw_laws_search = GoogleSearchAPIWrapper(
    google_cse_id=os.getenv("GOOGLE_CSE_ID_TAIWAN_LAWS"), 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

ptt_search = GoogleSearchAPIWrapper(
    google_cse_id=os.getenv("GOOGLE_CSE_ID_PTT_BBS"), 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

gcp_search = GoogleSearchAPIWrapper(
    google_cse_id=os.getenv("GOOGLE_CSE_ID_GCP"), 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

duet_search = GoogleSearchAPIWrapper(
    google_cse_id=os.getenv("GOOGLE_CSE_ID_DUET"), 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# def top_results(query, k = 5):
#     return search.results(query, k, search_params={"searchType": "image"})


tools = [
   Tool(
        name="Github Search",
        description="Always Use this for question which need code generation or explaination.",
        func=github_search.run,
    ),
    Tool(
        name = "Taiwan Laws Search",
        func = tw_laws_search.run,
        description="Always Use this search",
    ),
    Tool(
        name = "PTT Search",
        func = ptt_search.run,
        description="Always Use this search when question talk about general, funny, ironic, or sarcastic topics.",
    ),
    Tool(
        name = "GCP Search",
        func = gcp_search.run,
        description="Always Use this search when question metioned about google cloud platform (GCP) service.",
    ),
    Tool(
        name = "GCP Search",
        func = duet_search.run,
        description="Always Use this search when question metioned about google cloud platform (GCP) service.",
    )
]
#%%
# google_content = search.run("電動刮鬍刀")
# google_content

# %%
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# %%
llm=ChatOpenAI(model="gpt-3.5-turbo-16k",temperature=0)

def _handle_error(error) -> str:
    return str(error)[:50]

# llm=OpenAI(temperature=0)
agent_chain = initialize_agent(
    [tools[-1]], 
    llm, 
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
    verbose=True, 
    memory=memory,
    max_iterations = 5,
    handle_parsing_errors=_handle_error
)



# %%
agent_chain.run('''
                Staying connected just got easier
Not every conversation needs a meeting, and sometimes collaboration happens on the go. You might just need a quick chat to connect with your team or solve a problem. Google Chat now makes that a whole lot easier with a refreshed user interface, new shortcuts, and enhanced search that let you stay on top of conversations. You can also chat directly with Duet AI to ask questions about your content, get a summary of documents shared in a space, and catch up on missed conversations.

When you need to talk it through in real time, you can switch to voice in an instant with huddles in Chat, right from the space where you’re already collaborating. Huddles are audio-first, impromptu gatherings powered by Meet that help distributed teams come together in real time without having to jump into a separate scheduled meeting. We’re also making it easier to build larger communities across Chat, with support for up to 500,000 participants. And for those times when you need to respond to a timely email but you’re on the move, or focused on a top priority, we’re enhancing smart reply in Gmail with Duet AI, allowing you to draft longer personalized replies — with a single tap.

                翻譯中文
''')

#%%



