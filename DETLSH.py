import numpy as np
import random

class MultiLSH:
    def __init__(self, num_projections, num_spaces, dimensions, bucket_width):
        """
        Inicializa el esquema LSH con múltiples espacios proyectados.
        :param num_projections: Dimensiones de cada espacio proyectado (k).
        :param num_spaces: Número de espacios proyectados (L).
        :param dimensions: Dimensiones del espacio original (d).
        :param bucket_width: Ancho de los buckets (w).
        """
        self.num_projections = num_projections  # k
        self.num_spaces = num_spaces            # L
        self.dimensions = dimensions            # d
        self.bucket_width = bucket_width        # w
        self.hash_functions = self._generate_hash_functions()

    def _generate_hash_functions(self):
        """
        Genera funciones hash para cada uno de los L espacios proyectados.
        :return: Lista de L conjuntos de funciones hash.
        """

        # L * K hash functions!
        all_hash_functions = []
        for _ in range(self.num_spaces): # L
            hash_functions = []
            for _ in range(self.num_projections): # k
                a = np.random.normal(size=self.dimensions)  # Vector aleatorio en d dimensiones
                b = random.uniform(0, self.bucket_width)    # Desplazamiento aleatorio
                hash_functions.append((a, b))
            all_hash_functions.append(hash_functions)
        return all_hash_functions

    def project_point(self, point, space_index):
        """
        Proyecta un punto al espacio proyectado especificado por space_index.
        :param point: Punto en el espacio original.
        :param space_index: Índice del espacio proyectado (0 a L-1).
        :return: Coordenadas del punto en el espacio proyectado.
        """
        hashes = []
        for a, b in self.hash_functions[space_index]:
            h = np.floor((np.dot(a, point) + b) / self.bucket_width)
            hashes.append(int(h))
        return tuple(hashes)

    def assign_to_buckets(self, dataset):
        """
        Asigna puntos a buckets en los L espacios proyectados.
        :param dataset: Conjunto de puntos en el espacio original.
        :return: Lista de L diccionarios de buckets -> lista de puntos.
        """
        buckets_per_space = [{} for _ in range(self.num_spaces)]
        for point in dataset:
            for space_index in range(self.num_spaces):
                bucket_key = self.project_point(point, space_index)
                if bucket_key not in buckets_per_space[space_index]:
                    buckets_per_space[space_index][bucket_key] = []
                buckets_per_space[space_index][bucket_key].append(point)
        return buckets_per_space

    def query_buckets(self, query_point, buckets_per_space):
        """
        Consulta los buckets en los L espacios proyectados para un punto de consulta.
        :param query_point: Punto de consulta en el espacio original.
        :param buckets_per_space: Lista de L diccionarios de buckets.
        :return: Conjunto de puntos candidatos de los L espacios proyectados.
        """
        candidates = set()
        for space_index in range(self.num_spaces):
            bucket_key = self.project_point(query_point, space_index)
            if bucket_key in buckets_per_space[space_index]:
                print(f"Type: {type(buckets_per_space[space_index][bucket_key])}")
                candidates.update(tuple(point) for point in buckets_per_space[space_index][bucket_key])

        return list(candidates)


# Ejemplo de uso
if __name__ == "__main__":
    # Dataset aleatorio de puntos en un espacio de 5 dimensiones
    dataset = np.random.rand(10, 5)

    # Configuración de LSH
    k = 3         # Dimensiones de cada espacio proyectado
    L = 4         # Número de espacios proyectados
    d = 5         # Dimensiones del espacio original
    w = 1.0       # Ancho de los buckets

    multi_lsh = MultiLSH(num_projections=k, num_spaces=L, dimensions=d, bucket_width=w)

    # Asignar puntos a buckets en los L espacios proyectados
    buckets_per_space = multi_lsh.assign_to_buckets(dataset)
    print("Buckets en cada espacio proyectado:")
    for i, buckets in enumerate(buckets_per_space):
        print(f"Espacio {i}:")
        for bucket_key, points in buckets.items():
            print(f"  Bucket {bucket_key}: {points}")

    # Consultar vecinos aproximados para un punto específico
    query_point = np.random.rand(d)
    print("\nPunto de consulta:", query_point)
    candidates = multi_lsh.query_buckets(query_point, buckets_per_space)
    print("Candidatos en los L espacios proyectados:", candidates)