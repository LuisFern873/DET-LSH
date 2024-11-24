import numpy as np

def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

movies = fvecs_read("movielens/movielens_base.fvecs")
queries = fvecs_read("movielens/movielens_query.fvecs")

print("Number of movies: ", movies.size)
print("Sample movie vector: ")
print(movies[0])
print("Sample dimension: ", movies[0].size)


print("Number of queries: ", queries.size)
print("Sample movie vector: ")
print(queries[0])
print("Sample dimension: ", queries[0].size)