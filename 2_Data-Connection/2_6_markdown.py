#%%
from langchain.document_loaders import UnstructuredMarkdownLoader

markdown_path = "./data/index1.md"
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()
docs
# %%
loader = UnstructuredMarkdownLoader(markdown_path, mode="elements")
docs = loader.load()
docs
# %%
