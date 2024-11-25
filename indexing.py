from DETree import DETree

# Contruimos un DE-Tree por cada espacio proyectado

def create_index(encoded_points, max_leaf_size):

    L = len(encoded_points)

    detrees = []

    for i in range(L):  
        detree = DETree(max_leaf_size)
        n = len(encoded_points[i])
        # Insertar los puntos codificados en el árbol
        for j in range(n):
            point = encoded_points[i][j]

            # Insertar el punto codificado (se usa como punto original tmb)
            detree.insert(point, point)  
        detrees.append(detree)
    return detrees