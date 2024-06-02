import pandas as pd
import numpy as np
from Arbol import Arbol
from Entropia import Entropia

class ArbolID3(Arbol):
    
    def __init__(self, dato = None, es_hoja: bool = False) -> None:
        super().__init__(dato)
        self._es_hoja = es_hoja
        self._hijos: dict = {}

    def __str__(self):
        def mostrar(t: ArbolID3, nivel: int):
            tab = '.' * 4
            indent = tab * nivel
            out = indent + str(t.dato) + '\n'
            for valor, subarbol in t._hijos.items():
                out += indent + f"Valor: {valor}\n"
                out += mostrar(subarbol, nivel + 1)
            return out
        return mostrar(self, 0)
    
    @classmethod
    # X: dataset convertido en arrays sin la primer columna de atributos
    # y: columna con las clases
    def construir(cls, X: np.ndarray, y: np.ndarray, 
                  atributos: list[int],
                  profundidad_max: int = None, 
                  minimas_obs_n: int = None, 
                  minimas_obs_h: int = None, 
                  ganancia_minima: float = 0.0, 
                  profundidad_actual: int = 0
                  ) -> "ArbolID3":
        
        # Criterio de parada: Nodo puro (todos los elementos del nodo pertenecen a la misma clase)
        if len(np.unique(y)) == 1:
            return ArbolID3(y[0], es_hoja=True)
        
        # Criterio de parada: Maxima profundidad
        if profundidad_max is not None and profundidad_actual >= profundidad_max:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            return ArbolID3(clase_mayoritaria, es_hoja=True)
        
        # Criterio de parada: Mínimas observaciones por nodo
        if minimas_obs_n is not None and len(y) < minimas_obs_n:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            return ArbolID3(clase_mayoritaria, es_hoja=True)
        
        # Criterio de parada: Sin atributos para dividir
        if not atributos:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            return ArbolID3(clase_mayoritaria, es_hoja=True)
        
        # Seleccionamos el mejor atributo en base a entropía y ganancia de información
        ganancias = [Entropia.ganancia_informacion_atributo(X, y, atributo) for atributo in atributos]
        mejor_atributo = atributos[np.argmax(ganancias)]
        
        print(f"Ganancias en profundidad {profundidad_actual}: {ganancias}")
        print(f"Mejor atributo en profundidad {profundidad_actual}: {mejor_atributo} con ganancia {ganancias[np.argmax(ganancias)]}")
        
        # Criterio de parada: Ganancia mínima
        if ganancia_minima is not None and ganancias[np.argmax(ganancias)] < ganancia_minima:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            return ArbolID3(clase_mayoritaria, es_hoja=True)
        
        # Creamos el árbol con el mejor atributo
        arbol = ArbolID3(mejor_atributo)
        
        
        # Creamos nodos para cada valor del mejor atributo
        for valor in np.unique(X[:, mejor_atributo]):
            atributos_restantes = atributos.copy()
            atributos_restantes.remove(mejor_atributo)
        
            indices = np.where(X[:, mejor_atributo] == valor)[0]
            sub_X = X[indices]
            sub_y = y[indices]

            # Criterio de parada: Mínimas observaciones por hoja
            if minimas_obs_h is not None and minimas_obs_h > len(sub_y):
                clase_mayoritaria = cls.clase_mayoritaria(sub_y)
                subarbol = ArbolID3(valor=clase_mayoritaria, es_hoja=True)
            else:
                subarbol = cls.construir(sub_X, sub_y, atributos_restantes, profundidad_max, minimas_obs_n, minimas_obs_h, ganancia_minima, profundidad_actual + 1)
            
            arbol._hijos[valor] = subarbol
        
        return arbol
    
    

        # Función para visualizar el árbol
    def visualizar_arbol(self, arbol=None, padre=None, etiquetas=None):
        if etiquetas is None:
            etiquetas = {}

        if arbol is None:
            arbol = self

        if arbol.es_hoja():
            nodo_id = str(id(arbol))
            etiquetas[nodo_id] = f"{arbol.dato} -> Clase: {self.clase_mayoritaria(arbol)}"  # Mostrar la clase mayoritaria en las hojas
        else:
            nodo_id = str(id(arbol))
            etiquetas[nodo_id] = str(arbol.dato)

            if padre is not None:
                etiquetas[padre] += f" -> {arbol.dato}"

            for valor, hijo in arbol._hijos.items():
                self.visualizar_arbol(hijo, nodo_id, etiquetas)

        return etiquetas

    @staticmethod
    def clase_mayoritaria(y: np.ndarray) -> int:
        clases, conteo = np.unique(y, return_counts=True)
        return clases[np.argmax(conteo)]



# Carga del conjunto de datos (reemplaza 'tu_archivo.csv' con el nombre de tu archivo)
data = pd.read_csv('C:/Users/Administrador/Desktop/algo2/trabajoFinal/diabetes_cat.csv')

# Selecciona las columnas de atributos y la columna de etiquetas de clase
X = data[['HighChol', 'CholCheck', 'Smoker', 'Stroke',
          'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
          'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
          'DiffWalk', 'Sex']].to_numpy()  # Convertir a numpy array
y = data['Diabetes_012'].to_numpy()  # Convertir a numpy array

# Lista de atributos (columnas de X)
atributos = list(range(X.shape[1]))  # Asegura que la lista de atributos cubra todas las columnas

# Construcción del árbol ID3
arbol = ArbolID3.construir(X, y, atributos, profundidad_max=5)

# Imprime el árbol
print("Árbol de decisión ID3:")
print(arbol)