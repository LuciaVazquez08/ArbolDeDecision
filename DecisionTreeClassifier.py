from ArbolID3 import ArbolID3
from ArbolC4_5 import ArbolC4_5
import numpy as np
from typing import Generic, TypeVar
T = TypeVar('T')

class DecisionTreeClassifier:
    def __init__(self, algoritmo: ArbolID3 | ArbolC4_5 = ArbolID3, 
                 profundidad_max: int = None,
                 minimas_obs_n: int = None, 
                 minimas_obs_h: int = None, 
                 ganancia_minima: float = 0.0, 
                ):
        self.algoritmo = algoritmo
        self.profundidad_max = profundidad_max
        self.minimas_obs_n = minimas_obs_n
        self.minimas_obs_h = minimas_obs_h
        self.ganancia_minima = ganancia_minima
        self.arbol = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) == len(y):
            atributos = list(range(X.shape[1]))
            self.arbol = self.algoritmo.construir(X, y, atributos, self.profundidad_max, self.minimas_obs_n, self.minimas_obs_h, self.ganancia_minima)
        else:
            raise ValueError("debe haber la misma cantidad de instancias en los features y en el target")
    # Implementar la clasificación en base a los X recibidos -> devuelve la clase predecida para cada X
    def predict(self, X: np.ndarray) -> list[T]:
        predicciones = []
        for instancia in X:
            predicciones.append(self._predict_instancia(instancia, self.arbol))
        return np.array(predicciones)
    
    def _predict_instancia(self, x: np.ndarray, arbol: ArbolID3 | ArbolC4_5) -> T: 
        if arbol._es_hoja:
            return arbol.dato
        else:
            atributo = arbol.dato
            valor_atributo = x[atributo]
            if valor_atributo in arbol._hijos.keys():
                arbol = arbol._hijos[valor_atributo]
            else:
                # Si el valor no se encuentra en los hijos, retornamos la clase mayoritaria del nodo actual
                return ArbolID3.clase_mayoritaria([nodo.dato for nodo in arbol._hijos.values() if nodo._es_hoja])

    # TODO: get_params -> devuelve los hiperparametros no nulos
    # TODO: set_params
    # TODO: Implementar transform: Se encarga del encoding
