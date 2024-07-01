#%%
from langchain.document_loaders import UnstructuredHTMLLoader
# %%
loader = UnstructuredHTMLLoader("data/langchain.html")

docs = loader.load()
docs