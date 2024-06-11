from typing import TypeVar
from pandas import DataFrame
from ArbolID3 import ArbolID3
from ArbolC4_5 import ArbolC4_5
import numpy as np
from typing import Generic, TypeVar
T = TypeVar('T')

class DecisionTreeClassifier:
    def __init__(self, algoritmo: str = "ID3", 
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
        X_array = np.asarray(X)
        y_array = np.asarray(y)
        if len(X_array) == len(y_array):
            indice_atributos = list(range(X_array.shape[1]))
            nombres_atributos = X.columns.tolist()
            if self.algoritmo == "ID3":
                self.arbol = ArbolID3.construir(X_array, y_array, indice_atributos, nombres_atributos, self.profundidad_max, self.minimas_obs_n, self.minimas_obs_h, self.ganancia_minima)
            elif self.algoritmo == "C4.5":
                self.arbol = ArbolC4_5.construir(X_array, y_array, indice_atributos, nombres_atributos, self.profundidad_max, self.minimas_obs_n, self.minimas_obs_h, self.ganancia_minima)
            else:
                raise ValueError("No existe ese algoritmo")
        else:
            raise ValueError("Debe haber la misma cantidad de instancias en los features y en el target.")
        
    def predict(self, X: DataFrame) -> list[list[T]]:
        X_array = np.asarray(X)
        def _predict_instancia(instancia: np.ndarray, nodo_actual: ArbolID3 | ArbolC4_5) -> T:
            if nodo_actual._es_hoja:
                return nodo_actual.label
            atributo = nodo_actual.dato
            valor = instancia[atributo]
            
            # Manejamos las predicciones en donde el atributo es numérico
            if isinstance(valor, (int, float)):
                for (operador, umbral), hijo in nodo_actual._hijos.items():
                    if (operador == '<=' and valor <= umbral) or (operador == '>' and valor > umbral):
                        return _predict_instancia(instancia, hijo)
            
            # Manejamos las predicciones en donde el atributo es categórico
            else:
                if valor in nodo_actual._hijos:
                    return _predict_instancia(instancia, nodo_actual._hijos[valor])
                else:
                    # Si el valor no se encuentra en los hijos, retornamos la clase mayoritaria del nodo actual
                    clases = [nodo.label for nodo in nodo_actual._hijos.values() if nodo._es_hoja]
                    if self.algoritmo == "ID3":
                        return ArbolID3.clase_mayoritaria(np.asarray(clases))
                    else:
                        return ArbolC4_5.clase_mayoritaria(np.asarray(clases))
          
        predicciones = [_predict_instancia(instancia, self.arbol) for instancia in X_array]
        return predicciones
    
    def get_params(self):
        return self.__dict__
    
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self,key):
                setattr(self,key,value)
            else:
                raise ValueError(f"{key} no es un atributo valido")
            
    def score(self, X, y):
        X_array = np.asarray(X)
        y_array = np.asarray(y)
        if len(X_array) == len(y_array):
            pred = self.predict(X_array)
            acc = sum(p == t for p,t in zip(pred,y_array))
            accuracy = acc / len(y_array)
            return accuracy
        else:
            raise ValueError("Debe haber la cantidad de instancias en los features que en el target")
    
    def decision_path(self, X: DataFrame):
        pass
    
    # TODO: Implementar transform: Se encarga del encoding
