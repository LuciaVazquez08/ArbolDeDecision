from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import Arbol, Entropia

class ArbolID3(Arbol):
    
    def __init__(self, valor = None, es_hoja: bool = False) -> None:
        super().__init__(valor)
        self.valor = valor
        self.es_hoja = es_hoja
        self.hijos = {}
    
    @classmethod
    # X: dataset convertido en arrays sin la primer columna de atributos
    # y: columna con las clases
    def id3(cls, X: np.ndarray, y: np.ndarray, atributos: List[int],
            profundidad_max: Optional[int] = None, minimas_obs_n: int = 0, ganancia_minima: float = 0, profundidad_actual: int = 0
            ) -> "ArbolID3":
        
        # Criterio de parada: Nodo puro (todos los elementos del nodo perteneces a misma clase)
        if len(np.unique(y)) == 1:
            return ArbolID3(y[0], es_hoja = True)
        
        # Criterio de parada: Maxima profundidad (Agrego un is not none o pongo una prfundidad maxima????)
        if profundidad_max is not None and profundidad_max <= profundidad_actual:
            clase_mayoritaria= cls.clase_mayoritaria(y)
            return ArbolID3(clase_mayoritaria, es_hoja = True)
        
        # Criterio de parada: Mininimas observaciones por nodo(Que diferencia hay con minimas observaciones por hoja???)
        if minimas_obs_n > len(y):
            clase_mayoritaria= cls.clase_mayoritaria(y)
            return ArbolID3(clase_mayoritaria, es_hoja = True)
        
        # Criterio de parada: Sin atributos para dividir
        if not atributos:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            return ArbolID3(clase_mayoritaria, es_hoja=True)
        
        # Seleccionamos el mejor atributo en base a entropía y ganancia de informacion
        ganancias = [Entropia.ganancia_informacion_atributo(X, y, atributo) for atributo in atributos]
        mejor_atributo = atributos[np.argmax(ganancias)]

        # Criterio de parada: ganancia minima
        if ganancias[np.argmax(ganancias)]< ganancia_minima:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            return ArbolID3(clase_mayoritaria, es_hoja=True)
        
        # Creamos el árbol con el mejor atributo
        arbol = ArbolID3(mejor_atributo)
        
        atributos_restantes = atributos.copy()
        atributos_restantes.remove(mejor_atributo)
        
        # Creamos nodos para cada valor del mejor atributo
        for valor in np.unique(X[:, mejor_atributo]):
            indices = np.where(X[:, mejor_atributo] == valor)[0]
            
            sub_X = X[indices]
            sub_y = y[indices]
            
            # Recursión para construir el árbol
            subarbol = cls.id3(sub_X, sub_y, atributos_restantes, profundidad_max, minimas_obs_n, ganancia_minima, profundidad_actual + 1)
            arbol.hijos[valor] = subarbol
            
        return arbol
    
    @staticmethod
    def clase_mayoritaria(y: np.ndarray) -> int:
        clases, conteo = np.unique(y, return_counts=True)
        return clases[np.argmax(conteo)]

    @staticmethod      
    def csv_a_numpy(ruta_csv: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        df = pd.read_csv(ruta_csv)
        
        X = df.iloc[1:, :-1].values # todas las columnas menos la ultima
        y = df.iloc[:, -1].values # solo la ultima columna
        atributos = list(range(X.shape[1])) # lista de indices que representan los atributos
        
        return X, y, atributos
    
        
        
    
        