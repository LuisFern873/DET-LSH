import numpy as np

class DETreeNode:
    def __init__(self, is_leaf=True):
        self.is_leaf = is_leaf
        self.children = []  # Hijos del nodo (vacío para hojas)
        self.points = []  # Puntos en el nodo (solo para hojas)
        self.split_dimension = None  # Dimensión usada para dividir (para nodos internos)
        self.split_value = None  # Valor de división (para nodos internos)


class DETree:
    def __init__(self, max_leaf_size):
        self.root = DETreeNode(is_leaf=True)
        self.max_leaf_size = max_leaf_size

    def insert(self, point, encoded_representation):

        current_node = self.root
        while not current_node.is_leaf:  # Bajar hasta un nodo hoja
            if encoded_representation[current_node.split_dimension] <= current_node.split_value:
                current_node = current_node.children[0]
            else:
                current_node = current_node.children[1]
        
        # Agregar el punto al nodo hoja actual
        current_node.points.append((point, encoded_representation))
        
        # Dividir el nodo hoja si excede el tamaño máximo
        if len(current_node.points) > self.max_leaf_size:
            self.split_node(current_node)

    def split_node(self, node):
        # Selecciona la dimensión para dividir (balance dinámico)
        num_dimensions = len(node.points[0][1])  # Número de dimensiones en la representación codificada
        split_dimension = self.select_split_dimension(node.points, num_dimensions)

        # Encuentra el valor de división como el punto medio
        values = [point[1][split_dimension] for point in node.points]
        split_value = sum(values) / len(values)

        # Crear hijos y distribuir los puntos
        left_child = DETreeNode(is_leaf=True)
        right_child = DETreeNode(is_leaf=True)

        for point, encoded_representation in node.points:
            if encoded_representation[split_dimension] <= split_value:
                left_child.points.append((point, encoded_representation))
            else:
                right_child.points.append((point, encoded_representation))

        # Convertir el nodo actual en nodo interno
        node.is_leaf = False
        node.children = [left_child, right_child]
        node.split_dimension = split_dimension
        node.split_value = split_value
        node.points = []  # Limpiar puntos del nodo convertido

    def select_split_dimension(self, points, num_dimensions):
        variances = []
        for d in range(num_dimensions):
            values = [point[1][d] for point in points]
            variances.append(np.var(values))  # Varianza de los valores en la dimensión d
        return np.argmax(variances)  # Retorna la dimensión con mayor varianza

    def range_query(self, query_point, radius):
        candidates = []
        self._range_query_recursive(self.root, query_point, radius, candidates)
        return candidates

    def _range_query_recursive(self, node, query_point, radius, candidates):
        if node.is_leaf:
            # Revisar todos los puntos en la hoja
            for point, encoded_representation in node.points:
                distance = np.linalg.norm(np.array(query_point) - np.array(encoded_representation))
                if distance <= radius:
                    candidates.append(point)
        else:
            # Verificar límites de la división
            if query_point[node.split_dimension] - radius <= node.split_value:
                self._range_query_recursive(node.children[0], query_point, radius, candidates)
            if query_point[node.split_dimension] + radius > node.split_value:
                self._range_query_recursive(node.children[1], query_point, radius, candidates)


    def print_tree(self, node=None, depth=0):

        if node is None:
            node = self.root

        # Imprimir información del nodo
        indent = " " * depth * 2

        if node.is_leaf:
            print(f"{indent}Hoja: {len(node.points)} puntos")
            for point, encoded in node.points:
                print(f"{indent}  Punto: {point}, Codificado: {encoded}")
        else:
            print(f"{indent}Nodo Interno - Dividido por dimensión {node.split_dimension} con valor {node.split_value}")
            print(f"{indent}  Hijos:")
            for i, child in enumerate(node.children):
                print(f"{indent}    Hijo {i}:")
                self.print_tree(child, depth + 1)  # imprimir los hijos


