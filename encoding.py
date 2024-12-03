import numpy as np

def breakpoints_selection(K, L, dataset, sample_size, num_regions):
    # Dimensiones: 𝐾 * 𝐿 * (num_regions + 1)
    breakpoints = np.zeros((L, K, num_regions + 1))  
    for i in range(L):
        for j in range(K):
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
    encoded_points = np.zeros_like(projected_points, dtype=int)  # (L x n x K)


    # Iterar sobre los espacios proyectados, dimensiones y puntos
    for i in range(L):
        for j in range(K):
            for idx in range(n):

                # Encontramos el rango del punto según los breakpoints
                # búsqueda binaria
                for r in range(num_regions):
                    if breakpoints[i, j, r] <= projected_points[i, idx, j] < breakpoints[i, j, r + 1]:
                        encoded_points[i, idx, j] = r
                        break
    return encoded_points