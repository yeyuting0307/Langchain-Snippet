#%%
import os
from langchain.vectorstores import MatchingEngine

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
REGION = os.getenv("GOOGLE_CLOUD_REGION")
GS_BUCKET_URL = f"gs://{PROJECT_ID}-embeddings"
INDEX_ID = f"{PROJECT_ID}-vme"
ENDPOINT_ID = "base_endpoint"


# %%
texts = [
    "The cat sat on",
    "the mat.",
    "I like to",
    "eat pizza for",
    "dinner.",
    "The sun sets",
    "in the west.",
]


vector_store = MatchingEngine.from_components(
    texts=texts,
    project_id=PROJECT,
    region=REGION,
    gcs_bucket_uri=GS_BUCKET_URL,
    index_id=INDEX_ID,
    endpoint_id=ENDPOINT_ID,
)

vector_store.add_texts(texts=texts)

vector_store.similarity_search("lunch", k=2)