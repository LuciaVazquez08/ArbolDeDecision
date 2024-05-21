from typing import List
import pandas as pd
import numpy as np
from Arbol import Arbol  # Asegúrate de tener esta clase implementada correctamente

class ArbolID3(Arbol):
    
    def __init__(self, valor=None, es_hoja: bool = False) -> None:
        super().__init__()
        self.valor = valor
        self.es_hoja = es_hoja
        self.hijos = {}

    @classmethod
    def id3(cls, X: np.ndarray, y: np.ndarray, atributos: List[int], profundidad_max: None, minimas_obs_n: int = 0, ganancia_minima: int= 0, profundidad_actual: int = 0) -> "ArbolID3":
        # Criterio de parada: Nodo puro (todos los elementos del nodo pertenecen a la misma clase)
        if len(np.unique(y)) == 1:
            return ArbolID3(y[0], es_hoja = True)
        
        #Criterio de parada: Maxima profundidad (Agrego un is not none o pongo una prfundidad maxima????)
        if profundidad_max is not None and profundidad_max <= profundidad_actual:
            clase_mayoritaria= cls.clase_mayoritaria(y)
            return ArbolID3(clase_mayoritaria, es_hoja = True)
        
        #criterio de parada: Mininimas observaciones por nodo(Que diferencia hay con minimas observaciones por hoja???)
        if minimas_obs_n > len(y):
            clase_mayoritaria= cls.clase_mayoritaria(y)
            return ArbolID3(clase_mayoritaria, es_hoja = True)
        
        
        # Criterio de parada: Sin atributos para dividir
        if not atributos:
            # Devolver la clase mayoritaria
            clase_mayoritaria = cls.clase_mayoritaria(y)
            return ArbolID3(clase_mayoritaria, es_hoja=True)
        
        
        
        # Seleccionar el mejor atributo en base a la ganancia de información
        ganancias = [cls.ganancia_informacion_atributo(X, y, atributo) for atributo in atributos]
        mejor_atributo = atributos[np.argmax(ganancias)]
        
        #criterio de parada: ganancia minima
        if ganancias[np.argmax(ganancias)]< ganancia_minima:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            return ArbolID3(clase_mayoritaria, es_hoja=True)
        
        
        
        # Crear el árbol con el mejor atributo
        arbol = ArbolID3(mejor_atributo)
        
        # Atributos restantes para la recursión
        atributos_restantes = atributos.copy()
        atributos_restantes.remove(mejor_atributo)
        
        # Crear nodos para cada valor del mejor atributo
        for valor in np.unique(X[:, mejor_atributo]):
            indices = np.where(X[:, mejor_atributo] == valor)[0]
            sub_X = X[indices]
            sub_y = y[indices]
            
            # Recursión para construir el subárbol
            subarbol = cls.id3(sub_X, sub_y, atributos_restantes, profundidad_max, minimas_obs_n, ganancia_minima, profundidad_actual + 1)
            arbol.hijos[valor] = subarbol
        
        
        return arbol
            
    @staticmethod
    def csv_a_numpy(ruta_csv: str) -> tuple:
        df = pd.read_csv(ruta_csv)
        
        X = df.iloc[:, :-1].values  # Todas las columnas menos la última
        y = df.iloc[:, -1].values  # Solo la última columna
        atributos = list(range(X.shape[1]))  # Lista de índices que representan los atributos
        
        return X, y, atributos
    
    @staticmethod
    def clase_mayoritaria(y: np.ndarray) -> int:
        clases, conteo = np.unique(y, return_counts=True)
        return clases[np.argmax(conteo)]

        
    @staticmethod
    def entropia(y: np.ndarray) -> float:
        clases_por_atributo, cantidad_cada_clase = np.unique(y, return_counts=True)
        probabilidad_por_clase = cantidad_cada_clase / len(y)
        return -np.sum(probabilidad_por_clase * np.log2(probabilidad_por_clase + 1e-9))  # Agregar una pequeña constante para evitar log(0)

    @staticmethod
    def ganancia_informacion_atributo(X: np.ndarray, y: np.ndarray, indice_atributo: int) -> float:
        entropia_inicial = ArbolID3.entropia(y)
        clases, cantidad_por_clase = np.unique(X[:, indice_atributo], return_counts=True)
        entropia_ponderada = 0
        
        for cla in clases:
            subconjunto_clase = y[X[:, indice_atributo] == cla]
            entropia_ponderada += (len(subconjunto_clase) / len(y)) * ArbolID3.entropia(subconjunto_clase)
        
        return entropia_inicial - entropia_ponderada

        
        
        
    
        