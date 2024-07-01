#%%
from langchain.document_loaders.csv_loader import CSVLoader


loader = CSVLoader(file_path='./data/實價登錄資料.csv')
data = loader.load()
data
# %%
