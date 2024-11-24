import numpy as np
from LSH import LSH
from fvecs_read import fvecs_read
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == "__main__":

    dataset = fvecs_read("movielens/movielens_base.fvecs")

    # Init LSH :)
    lsh = LSH(K=50, L=16, d=dataset[0].size, w=5.0)




    # Asignar puntos a buckets en los L espacios proyectados
    buckets_per_space = lsh.assign_to_buckets(dataset)

    # print("Buckets en cada espacio proyectado:")
    # for i, buckets in enumerate(buckets_per_space):
    #     print(f"Espacio {i}:")
    #     for bucket_key, points in buckets.items():
    #         print(f"  Bucket {bucket_key}: points")

    queries = fvecs_read("movielens/movielens_query.fvecs")
    query = queries[0]
    
    candidates = lsh.query(query, buckets_per_space)

    print("Punto de consulta:", query)
    print("Candidatos:", candidates)


    # Como no tenemos interfaz graf...

    # calcular similitud coseno
    # para saber si LSH realmente retorna movies parecidas al query

    query = query.reshape(1, -1)
    candidates = np.array(candidates)
    similarities = cosine_similarity(query, candidates)[0]

    # resultados ordenados
    sorted_indices = np.argsort(similarities)[::-1]
    for i in sorted_indices:
        print(f"Candidate {i} Similarity: {similarities[i]}")