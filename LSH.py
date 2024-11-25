import numpy as np
import random

# 𝐾: dimensión de cada espacio proyectado
# 𝐿: número de espacios proyectados
# d: dimensión del espacio original
# w: ancho de los buckets
# H: hash functions (𝐾*𝐿)

class LSH:
    def __init__(self, K, L, d, w):
        self.K = K
        self.L = L
        self.d = d
        self.w = w
        self.H = self.generate_hash_functions()

    # During the encoding scheme...
    # "we first use 𝐾 · 𝐿 hash functions to calculate the 𝐾-dimensional points in 𝐿 projected spaces"

    def generate_hash_functions(self):
        H = np.zeros((self.L, self.K), dtype=object)
        for i in range(self.L):
            for j in range(self.K):
                a = np.random.normal(size=self.d)  # Vector aleatorio en d dimensiones
                b = random.uniform(0, self.w)    # Desplazamiento aleatorio
                H[i][j] = (a, b)
        return H

    # "H𝑖(𝑜) = [ℎ𝑖1(𝑜), ..., ℎ𝑖𝐾(𝑜)] denote a point 𝑜 in the 𝑖-th projected space."
    def project_point(self, point, space_index):
        # point: Punto en el espacio original
        # space_index: Índice del espacio proyectado (0 a 𝐿 - 1)

        hashes = []
        for a, b in self.H[space_index]:
            h = np.floor((np.dot(a, point) + b) / self.w)
            hashes.append(int(h))
        return tuple(hashes) # H𝑖(𝑜)
    

    def project_dataset(self, dataset):

        n = len(dataset)
        projected_points = np.zeros((self.L, n, self.K), dtype=int)

        for i in range(self.L):  # Para cada espacio proyectado
            for idx, point in enumerate(dataset):  # Para cada punto en el dataset
                projected_points[i, idx] = self.project_point(point, i)
        
        return projected_points


    def assign_to_buckets(self, dataset):

        buckets = [{} for _ in range(self.L)]

        for point in dataset: # Proyectamos todos los puntos del dataset en cada espacio 0 a 𝐿 - 1

            for space_index in range(self.L):

                bucket_key = self.project_point(point, space_index)

                if bucket_key not in buckets[space_index]:
                    buckets[space_index][bucket_key] = []
                buckets[space_index][bucket_key].append(point)
        return buckets

    def query(self, query_point, buckets):

        candidates = set()
        for space_index in range(self.L): # en cada espacio...
            bucket_key = self.project_point(query_point, space_index) # proyectamos el punto de consulta

            if bucket_key in buckets[space_index]:
                print(f"Type: {type(buckets[space_index][bucket_key])}")
                candidates.update(tuple(point) for point in buckets[space_index][bucket_key])

        return list(candidates)
