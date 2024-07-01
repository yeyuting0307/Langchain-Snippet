#%%
import faiss
import time
import sys
import numpy as np
import pandas as pd

#%%
# ==================== useful func ====================
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def eucledean_distance(x, y):
    return np.linalg.norm(x - y)

def standardize(x):
    return (x - x.mean()) / x.std()

#%%
# ==================== faiss ====================
dim = 768 # embedding dimension
n_idx = 100000 # index database size
n_query = 10000 # query size
np.random.seed(20231205)

#%%
# make index
xb = np.random.random((n_idx, dim)).astype('float32')
faiss.normalize_L2(xb)

# make query
xq = np.random.random((n_query, dim)).astype('float32')
faiss.normalize_L2(xq)
#%%
# build index
index = faiss.IndexFlatL2(dim) # build the index proxy

assert index.is_trained , "index should be trained"
index.add(xb) # add the index database to index

assert index.ntotal == n_idx, "index should contain n_idx vectors"

#%%
# search
topK = 5
D, I = index.search(x = xq, k = topK) # D: square of distance, I: index

# %%
# validate index and distance
sample_size = 5
sample_idx = np.random.choice(n_query, sample_size, replace=False)

for i in sample_idx:
    q = xq[i]
    for j in range(topK):
        print('index', I[i][j], ":", eucledean_distance(xb[I[i][j]], q)**2, 'vs', D[i][j])
    print('-----------------')

# %%
# ==================== Index Factory ====================
dim = 64 
measure = faiss.METRIC_L2
param = "IVF100,Flat" # IVF{N},Flat -> N denote for N centers in K-means
topK = 5
index = faiss.index_factory(dim, param, measure)
if not index.is_trained:
    index.train(xb) # K-means clustering
assert index.is_trained, "index should be trained"
index.add(xb) 
D, I = index.search(x = xq, k = topK) 
for i in range(len(xq)):
    print('query:\n',xq[i])
    print('database:\n',xb[I[i][0],:])
    print('dist.: ',D[i][0])
    break
# %%
# ==================== Benchmark ====================
## make xb, xq
DIM = 768 
dim = DIM 
n_idx = 100000 
n_query = 10000 

np.random.seed(20231205)
xb = np.random.random((n_idx, dim)).astype('float32')
faiss.normalize_L2(xb)

xq = np.random.random((n_query, dim)).astype('float32')
faiss.normalize_L2(xq)

## Baseline
param = 'Flat'
dim = DIM
measure = faiss.METRIC_INNER_PRODUCT 
topK = 1
index = faiss.index_factory(dim, param, measure)
index.add(xb)

D, I_baseline = index.search(x = xq, k = topK)
I_baseline

#%%
params = ['Flat', 'IVF50,Flat','IVF100,Flat', 'PQ16', 'PQ32', 
          'IVF100,PQ16', 'IVF100,PQ32', 'HNSW64', 'HNSW64,Flat']

benchmark = []
for param in params:
    print(param)
    summary = {
        "param" : param, 
        "need_train" : False,
        "train_time" : 0, 
        "build_time" : 0,
        "search_time" : 0, 
        "index_size_mb" : 0,
        "acc_1" : 0, 
        "acc_5" : 0,
        "embedding_dim" : DIM,
        "database_size" : n_idx,
        "query_size" : n_query,
    }
    dim = DIM
    measure = faiss.METRIC_INNER_PRODUCT # Inner Product
    topK = 5
    index = faiss.index_factory(dim, param, measure)

    need_train = not index.is_trained
    summary["need_train"] = need_train
    if need_train:
        S_train = time.time()
        index.train(xb) # K-means clustering
        E_train = time.time()
        summary["train_time"] = E_train - S_train
    assert index.is_trained, "index should be trained"

    S_build = time.time()
    index.add(xb) 
    E_build = time.time()
    summary["build_time"] = E_build - S_build

    index_size_mb = sys.getsizeof(index) / (1024 * 1024)
    summary["index_size_mb"] = index_size_mb

    S_search = time.time()
    D, I = index.search(x = xq, k = topK) 
    E_search = time.time()
    summary["search_time"] = E_search - S_search

    acc_1 = sum(I[:,0].reshape(-1) == I_baseline.reshape(-1)) / len(I)
    summary["acc_1"] = acc_1

    acc_5 = 0
    for i in range(len(I)):
        if I_baseline[i][0] in I[i]:
            acc_5 += 1/len(I)
    summary["acc_5"] = acc_5

    E = time.time()
    benchmark.append(summary)

# %%
result = pd.DataFrame(benchmark)
result.to_csv('benchmark_mips.csv', index=False)
result
# %%

