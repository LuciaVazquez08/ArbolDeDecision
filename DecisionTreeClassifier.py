from ArbolID3 import ArbolID3
from ArbolC4_5 import ArbolC4_5
import numpy as np

class DecisionTreeClassifier:
    def __init__(self, algoritmo: ArbolID3 | ArbolC4_5, 
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
        atributos = list(range(X.shape[1]))
        self.arbol = self.algoritmo.construir(X, y, atributos, self.profundidad_max, self.minimas_obs_n, self.minimas_obs_h, self.ganancia_minima)
        print(self.arbol)
        
    # Implementar la clasificación en base a los X recibidos -> devuelve la clase predecida para cada X
    def predict(self, X: np.ndarray) -> list[int]:
        predicciones = []
        for instancia in X:
            predicciones.append(self._predict_instancia(instancia, self.arbol))
        print(predicciones)
        return predicciones
    
    # TODO: Implementa la predicción para una instancia específica 
    def _predict_instancia(self, x: np.ndarray, arbol: ArbolID3 | ArbolC4_5) -> int: 
        if arbol._es_hoja:
            return arbol.dato
        else:
            valor_atributo = x[arbol.dato]
            subarbol = arbol._hijos.get(valor_atributo)
            if subarbol is None:
                return arbol.clase_mayoritaria(x)
            return self._predict_instancia(x, subarbol)

    # TODO: Implementar transform: Se encarga del encoding
