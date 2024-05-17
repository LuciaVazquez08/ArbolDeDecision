from typing import List
import pandas as pd
import numpy as np
import Arbol

# vamos a intentar
class ArbolID3(Arbol):
    
    def __init__(self, es_hoja: bool = False) -> None:
        super().__init__()
        self.es_hoja = es_hoja
    
    @classmethod
    # X: dataset convertido en arrays (hacerlo) sin la primer columna de atributos
    # y: columna con las clases
    def id3(cls, X: np.ndarray, y: np.ndarray, atributos: List[int]) -> "ArbolID3":
        
        # Criterio de parada: Nodo puro (todos los elementos del nodo perteneces a misma clase)
        if len(np.unique(y)) == 1:
            return ArbolID3(y[0], es_hoja = True)
        
        # Criterio de parada: Profundidad máxima del árbol
        
            
        
        # Seleccionamos el mejor atributo en base a entropía y ganancia de informacion
        ganancias = [cls.ganancia_informacion_atributo(X, y, atributo) for atributo in atributos]
        mejor_atributo = atributos[np.argmax(ganancias)]
        
        # Creamos el árbol con el mejor atributo
        arbol = ArbolID3(mejor_atributo)
        
        atributos_restantes = atributos.copy()
        atributos_restantes.remove(mejor_atributo)
        
        # Creamos nodos para cada valor del mejor atributo
        for valor in np.unique(X[:, mejor_atributo]): # del mejor atributo agarra todos los valores unicos
            indices = np.where(X[:, mejor_atributo] == valor)[0]
            
            sub_df = X[indices]
            sub_y = y[indices]
            
            # Recursión para construir el árbol
            subarbol = cls.id3(sub_df, sub_y, atributos_restantes)
            arbol.insertar_subarbol(subarbol)
            
        return arbol
            
    def csv_a_numpy(ruta_csv: str) -> tuple(np.ndarray, np.ndarray, List[int]):
        df = pd.read_csv(ruta_csv)
        
        X = df.iloc[1:, :-1].values # todas las columnas menos la ultima
        y = df.iloc[:, -1].values # solo la ultima columna
        atributos = list(range(X.shape[1])) # lista de indices que representan los atributos
        
        return X, y, atributos
        
        
    @staticmethod
    def entropia(y: np.ndarray) -> float: 
        clases_por_atributo, cantidad_cada_clase = np.unique(y, return_counts=True) 
        probabilidad_por_clase = cantidad_cada_clase / len(y) 
        return -np.sum(probabilidad_por_clase * np.log2(probabilidad_por_clase)) 

    @staticmethod
    def ganancia_informacion_atributo(X: np.ndarray, y: np.ndarray, indice_atributo: int) -> float: 
        entropia_inicial = entropia(y) 
        clases, cantidad_por_clase = np.unique(X[:, indice_atributo], return_counts=True) 
        entropia_ponderada = 0 
        for cla, cant in enumerate(clases): 
            cantidad = cantidad_por_clase[cla] 
            subconjunto_clase = y[X[:, indice_atributo] == clases[cla]] 
            entropia_ponderada += (cantidad / len(y)) * entropia(subconjunto_clase) 
        return entropia_inicial - entropia_ponderada
        
        
        
    
        