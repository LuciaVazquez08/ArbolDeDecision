from typing import TypeVar
from pandas import DataFrame
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

    def fit(self, X: DataFrame, y: DataFrame) -> None:
        X_array = np.array(X)
        y_array = np.array(y)
        if len(X_array) == len(y_array):
            indice_atributos = list(range(X_array.shape[1]))
            nombres_atributos = X.columns.tolist()
            self.arbol = self.algoritmo.construir(X_array, y_array, indice_atributos, nombres_atributos, self.profundidad_max, self.minimas_obs_n, self.minimas_obs_h, self.ganancia_minima)
        else:
            raise ValueError("Debe haber la misma cantidad de instancias en los features y en el target.")
        
    def predict(self, X: DataFrame) -> list[list[T]]:
        X_array = np.array(X)
        def _predict_instancia(instancia: np.ndarray, nodo_actual: ArbolID3 | ArbolC4_5) -> T:

            if nodo_actual._es_hoja:
                return nodo_actual.dato
            atributo = nodo_actual.dato
            valor = instancia[atributo]
            tipo_atributo = ArbolC4_5.determinar_tipo_atributo(X_array[:, atributo])

            # Manejamos las predicciones en donde el atributo es numérico
            if tipo_atributo == 'continuo':
                for (operador, umbral), hijo in nodo_actual._hijos.items():
                    if (operador == '<=' and valor <= umbral) or (operador == '>' and valor > umbral):
                        return _predict_instancia(instancia, hijo)
            
            # Manejamos las predicciones en donde el atributo es categórico
            elif tipo_atributo == 'categorico':
                if valor in nodo_actual._hijos:
                    return _predict_instancia(instancia, nodo_actual._hijos[valor])
                else:
                    # Si el valor no se encuentra en los hijos, retornamos la clase mayoritaria del nodo actual
                    clases = [nodo.dato for nodo in nodo_actual._hijos.values() if nodo._es_hoja] 
                    return self.algoritmo.clase_mayoritaria(np.array(clases))
            else:
                raise ValueError("Tipo de atributo desconocido")
          
        predicciones = [_predict_instancia(instancia, self.arbol) for instancia in X_array]
        return predicciones

    # TODO: get_params -> devuelve los hiperparametros no nulos
    # TODO: set_params
    # TODO: Implementar transform: Se encarga del encoding
