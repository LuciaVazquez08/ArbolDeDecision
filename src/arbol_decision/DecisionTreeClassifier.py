from arbol_decision.ArbolID3 import ArbolID3
from arbol_decision.ArbolC4_5 import ArbolC4_5
import pandas as pd
import numpy as np
from typing import Generic, TypeVar
T = TypeVar('T')

class DecisionTreeClassifier:

    """
    Implementación del DecisionTreeClassifier.

    Parámetros
    ----------
    algoritmo : ArbolID3 | ArbolC4_5, default=ArbolID3
        El algoritmo elegido para realizar la construcción del árbol de decisión.

    profundidad_max : int, default=None 
        La profundidad máxima que puede alcanzar el árbol.

    minimas_obs_n : int, default=None 
        La cantidad mínima de observaciones requeridas para dividir un nodo interno. 

    minimas_obs_h : int, default=None 
        La cantidad mínima de observaciones requeridas presentes en una hoja. 

    ganancia_minima : float, default=None 
        La ganancia mínima al elegir el mejor atributo para dividir un nodo.

    top_atributos : int, default=3
        El top de atributos, ordenados de mayor a menor por aparición, 
        que serán seleccionados para diferenciar atributos continuos de categóricos. 

    umbral : float, default=0.8
        El umbral de proporción para diferenciar atributos continuos de categóricos. 

    Atributos
    ---------
    _tipos_atributos: Los tipos de atributos del conjunto de entrenamiento X, default=None

    _y : Los valores del target en formato Array, default=None

    _arbol : Instancia del árbol, default=None

    """

    def __init__(self, 
                 algoritmo: ArbolID3 | ArbolC4_5 = ArbolID3, 
                 profundidad_max: int = None,
                 minimas_obs_n: int = None, 
                 minimas_obs_h: int = None, 
                 ganancia_minima: float = None, 
                 top_atributos: int = 3,
                 umbral: float = 0.8
                ):
        self.algoritmo = algoritmo
        self.profundidad_max = profundidad_max
        self.minimas_obs_n = minimas_obs_n
        self.minimas_obs_h = minimas_obs_h
        self.ganancia_minima = ganancia_minima
        self.top_atributos = top_atributos
        self.umbral = umbral
        self._tipos_atributos = None
        self._y = None
        self._arbol = None


    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Entrena un árbol de decisión a partir del conjunto de entrenamiento (X, y).

        Parámetros
        ----------
        X: DataFrame
            Las muestras del conjunto de entrenamiento.

        y: DataFrame
            Los valores del target con valores integers o strings.

        Raises
        ------
        ValueError : Si el número de muestras en X y y no es el mismo.
        """
        X_array = np.array(X)
        y_array = np.array(y)
        self._y = y_array  
        
        if len(X_array) == len(y_array):
            
            # Completamos los valores faltantes si los hay
            if self.algoritmo == ArbolC4_5:
                X_array = ArbolC4_5.imputar_valores_faltantes(X_array, self.top_atributos, self.umbral)

            indice_atributos = list(range(X_array.shape[1]))
            nombres_atributos = X.columns.tolist()
            self._tipos_atributos = [ArbolC4_5.determinar_tipo_atributo(X_array[:, atributo], self.top_atributos, self.umbral) for atributo in indice_atributos]
            self._arbol = self.algoritmo.construir(X_array, y_array, self._tipos_atributos, indice_atributos, nombres_atributos, 
                                                  self.profundidad_max, self.minimas_obs_n, self.minimas_obs_h, self.ganancia_minima)
        else:
            raise ValueError("Debe haber la misma cantidad de instancias en X y en y")
        
    def predict(self, X: pd.DataFrame) -> list[T]:
        """
        Realiza la predicción del conjunto de datos de entrada utilizando el DecisionTreeClassifier entrenado.

        Parámetros
        ----------
        X : DataFrame
            Las muestras de entrada para las cuales se realizarán las predicciones.

        Returns
        -------
        list[T] : Devuelve una lista con las predicciones para cada instancia de X.
        """
        X_array = np.array(X)
        
        def _predict_instancia(instancia: np.ndarray, nodo_actual: self.algoritmo) -> T:

            if nodo_actual._es_hoja:
                return nodo_actual.label
            
            atributo = nodo_actual.dato
            valor = instancia[atributo]
            tipo_atributo = self._tipos_atributos[atributo]

            # Manejamos las predicciones en donde el atributo es continuo
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
                    clases = [nodo.label for nodo in nodo_actual._hijos.values() if nodo._es_hoja]
                    # Si la lista de clases está vacía, devuelve la clase mayoritaria de todo el conjunto de entrenamiento y
                    if not clases: 
                        return self.algoritmo.clase_mayoritaria(self._y)
                    return self.algoritmo.clase_mayoritaria(np.array(clases))
                
            else:
                raise ValueError("Tipo de atributo desconocido")
          
        predicciones = [_predict_instancia(instancia, self._arbol) for instancia in X_array]
        return predicciones
    
    def get_params(self):
        """
        Permite obtener los parametros del arbol.

        Parámetros
        ----------
        self : DecisionTreeClassifier

        Returns
        -------
        dict() : nombre de los parametros del DecisionTreeClassifier y sus valores
        """
        return self.__dict__
    
    def set_params(self, **params : list[str]) -> None:
        """
        Permite setear los parametros del arbol.

        Parámetros
        ----------
        self : DecisionTreeClassifier
        params: list[str]
            Nombres de los parametros a setear

        Returns
        -------
        None

        Raises
        ------
        ValueError : Si uno de los nombres de parametro pasados no corresponde a un parametro de DecisionTreeClassifier
        """
        for key, value in params.items():
            if hasattr(self,key):
                setattr(self,key,value)
            else:
                raise ValueError(f"{key} no es un atributo valido")
            
    def score(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        """
        Permite evaluar la precision de la prediccion del arbol.

        Parámetros
        ----------
        self : DecisionTreeClassifier
        X : DataFrame
            Conjunto de datos de entrada
        y : DataFrame
            Etiquetas correspondientes a X

        Returns
        -------
        float : precision de la prediccion sobre instancias 

        Raises
        ------
        ValueError : Si el tamaño de las instancias presentadas y de los target no coinciden
        """
        X_array = np.asarray(X)
        y_array = np.asarray(y)
        if len(X_array) == len(y_array):
            pred = self.predict(X_array)
            acc = sum(p == t for p,t in zip(pred,y_array))
            accuracy = acc / len(y_array)
            return accuracy
        else:
            raise ValueError("Debe haber la cantidad de instancias en los features que en el target")
    


