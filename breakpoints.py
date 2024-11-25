import numpy as np
from LSH import LSH
from fvecs_read import fvecs_read

# TO-DO: revisar si esto es correcto, por ahora si retorna los punto codificados! :)

def breakpoints_selection(K, L, dataset, sample_size, num_regions):

    # Dimensiones: 𝐾 * 𝐿 * (num_regions + 1)
    breakpoints = np.zeros((L, K, num_regions + 1))  
    
    for i in range(L):  # Para cada espacio proyectado
        for j in range(K):  # Para cada dimensión del espacio proyectado
            
            # Muestrear puntos de los datos en la dimensión actual
            sampled_data = np.random.choice(dataset[:, j], size=sample_size, replace=False)
            
            # Ordenar los datos muestreados
            sampled_data.sort()
            
            # Seleccionar breakpoints dividiendo uniformemente los datos
            for z in range(1, num_regions):
                breakpoint_index = int((z / num_regions) * len(sampled_data))
                breakpoints[i, j, z] = sampled_data[breakpoint_index]
            
            # Agregar mínimo y máximo a los extremos
            breakpoints[i, j, 0] = sampled_data[0]  # Mínimo
            breakpoints[i, j, -1] = sampled_data[-1]  # Máximo

    return breakpoints

def dynamic_encoding(projected_points, breakpoints):

    L, n, K = projected_points.shape
    num_regions = breakpoints.shape[2] - 1  # Determina el número de regiones
    encoded_points = np.zeros_like(projected_points, dtype=int)  # Matriz de salida (L x n x K)


    # Iterar sobre los espacios proyectados, dimensiones y puntos
    for i in range(L):
        for j in range(K):
            for idx in range(n):

                
                # Encuentra el rango del punto según los breakpoints usando búsqueda binaria
                for r in range(num_regions):
                    if breakpoints[i, j, r] <= projected_points[i, idx, j] < breakpoints[i, j, r + 1]:
                        encoded_points[i, idx, j] = r
                        break
    return encoded_points



dataset = fvecs_read("movielens/movielens_base.fvecs")
dataset = dataset[:1000]

K = 50
L = 16
d = dataset[0].size
w = 5.0

lsh = LSH(K, L, d, w)
projected_points = lsh.project_dataset(dataset)
print("All points projected.")

sample_size = 20 
num_regions = 8

breakpoints = breakpoints_selection(K, L, projected_points[0], sample_size, num_regions)
print("Breakpoints selected.")

encoded_points = dynamic_encoding(projected_points, breakpoints)
print("Points encoded.")

print(encoded_points)


