#%%
from langchain.embeddings import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()
# %%
embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)
len(embeddings), len(embeddings[0])
# %%
embeddings[0][:5]
# %%

embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
embedded_query[:5]
# %%
