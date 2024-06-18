import numpy as np
from pandas import DataFrame
from ArbolID3 import ArbolID3
from ArbolC4_5 import ArbolC4_5
from DecisionTreeClassifier import DecisionTreeClassifier
from collections import Counter
from typing import Generic, TypeVar
T = TypeVar('T')

class RandomForest:

    """
    Implementación del RandomForestClassifier.

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

    n_estimadores : int, default=100
        La cantidad de árboles en el bosque.

    bootstrap : bool, default=True
        Si es True, utiliza muestras bootstrap para entrenar cada árbol.

    feature_selection_method : str, default="sqrt"
        El método para seleccionar los atributos a considerar al buscar el mejor 
        atributo para dividir. Puede ser:
        - "sqrt" (raíz cuadrada de la cantidad total de atributos).
        - "log2" (logaritmo en base 2 de la cantidad total de atributos).
        - "none" (selecciona todos los atributos).

    Atributos
    ---------
    _arboles : list[DecisionTreeClassifier]
        Lista que contiene los árboles de decisión entrenados.

    """

    def __init__(self, algoritmo: ArbolID3| ArbolC4_5 = ArbolID3, 
                 profundidad_max: int = None,
                 minimas_obs_n: int = None, 
                 minimas_obs_h: int = None, 
                 ganancia_minima: float = None,
                 top_atributos: int = 3,
                 umbral: float = 0.8, 
                 n_estimadores: int = 100, 
                 bootstrap: bool = True,
                 feature_selection_method: str = "sqrt"
                ):
        self.algoritmo = algoritmo
        self.profundidad_max = profundidad_max
        self.minimas_obs_n = minimas_obs_n
        self.minimas_obs_h = minimas_obs_h
        self.ganancia_minima = ganancia_minima
        self.ganancia_minima = ganancia_minima
        self.top_atributos = top_atributos
        self.umbral = umbral
        self.n_estimadores = n_estimadores
        self.bootstrap = bootstrap
        self.feature_selection_method = feature_selection_method
        self.arboles: list[DecisionTreeClassifier] = [] 

    @staticmethod
    def bootstraping(X: np.ndarray , y: np.ndarray, n_estimadores: int) -> list[list[np.ndarray]]:
        """
        Genera muestras bootstrap del conjunto de datos (X, y) para cada estimador del bosque.

        Parámetros
        ----------
        X : np.ndarray
            Las muestras del conjunto de datos.

        y : np.ndarray
            Los valores del target.

        n_estimadores : int
            El número de muestras bootstrap a generar, equivalente al número de estimadores en el bosque.

        Returns
        -------
        list[list[np.ndarray]] : Una lista de listas, donde cada sublista contiene dos arrays: 
                                la muestra bootstrap de X y la muestra bootstrap de y.
        """
        muestras = [] 
        n_muestras = len(y)
        random_state = np.random.RandomState(seed=42) 

        for arbol in range(n_estimadores):
            choices = random_state.choice(n_muestras, size=n_muestras, replace=True) 
            new_X = X[choices]
            new_y = y[choices]
            muestras.append([new_X, new_y])
        return muestras
    
    @staticmethod
    def random_feature_selection(muestras_boostrapeadas: list[list[np.ndarray]], 
                                 feature_selection_method: str, 
                                 nombres_atributos: list[str]
                                 ) -> list[list[np.ndarray, list[str]]]:
        """
        Realiza la selección aleatoria de atributos para cada muestra bootstrap.

        Parámetros
        ----------
        muestras_boostrapeadas : list[list[np.ndarray]]
            Lista de muestras bootstrap, donde cada sublista contiene dos arrays: 
            la muestra bootstrap de X y la muestra bootstrap de y.
        
        feature_selection_method : str
            El método de selección de atributos. Puede ser:
            - "log" : Selecciona log(cantidad de atributos).
            - "sqrt" : Selecciona sqrt(cantidad de atributos).
            - "none" : Selecciona todos los atributos.

        nombres_atributos : list[str]
            Lista con los nombres de los atributos originales en X.

        Returns
        -------
        list[list[np.ndarray, list[str]]] : Una lista de listas, donde cada sublista contiene:
            - La muestra bootstrap con los atributos seleccionados.
            - Los valores del target para la muestra bootstrap.
            - Los nombres de los atributos seleccionados.
        
        Raises
        ------
        ValueError : Si el método de selección de atributos no es válido.
        """
        muestras_finales = [] 
        random_state = np.random.RandomState(seed=42)
        numero_features = muestras_boostrapeadas[0][0].shape[1]

        if feature_selection_method == "log":
            n_features = round(np.log(numero_features))
        elif feature_selection_method == "sqrt":
            n_features = round(np.sqrt(numero_features))
        elif feature_selection_method == "none":
            n_features = numero_features
        else:
            raise ValueError("No es un metodo valido de selección de atributos.")

        for muestra in muestras_boostrapeadas:
            choices = random_state.choice(numero_features, size=n_features, replace=False)
            choices = sorted(choices)
            x_selec = muestra[0][:, choices]
            nombres_atributos_seleccionados = [nombres_atributos[i] for i in choices]
            muestras_finales.append([x_selec, muestra[1], nombres_atributos_seleccionados])
        
        return muestras_finales

    def fit(self, X: DataFrame, y: DataFrame) -> None:
        """
        Entrena el bosque de árboles de decisión a partir del conjunto de datos de entrenamiento (X, y).

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
        if len(X_array) == len(y_array):
            if self.bootstrap:
                muestras = RandomForest.bootstraping(X_array, y_array, self.n_estimadores)
            else:
                muestras = [[X_array, y_array] for _ in range(self.n_estimadores)]

            nombres_atributos = X.columns.tolist()              
            muestras = RandomForest.random_feature_selection(muestras, feature_selection_method=self.feature_selection_method, nombres_atributos=nombres_atributos)

            for n in range(self.n_estimadores):
                arbol = DecisionTreeClassifier(self.algoritmo, self.profundidad_max, self.minimas_obs_n, self.minimas_obs_h, self.ganancia_minima, 
                                               self.top_atributos, self.umbral)
                arbol.fit(DataFrame(muestras[n][0], columns=muestras[n][2]), DataFrame(muestras[n][1]))
                self.arboles.append(arbol)
        else:
            raise ValueError("Debe haber la misma cantidad de instancias en X y en y")
        
    def predict(self, X: DataFrame) -> list[T]:
        """
        Realiza la predicción del conjunto de datos de entrada utilizando el RandomForestClassifier entrenado.

        Parámetros
        ----------
        X: DataFrame
            Las muestras de entrada para las cuales se realizarán las predicciones.

        Returns
        -------
        list[T] : Devuelve una lista con las predicciones para cada instancia de X.

        """
        pred_arboles = []

        for arbol in self.arboles:
            preds = arbol.predict(X)
            pred_arboles.append(preds)
            
        preds_finales = []
        for i in range(len(X)):
            pred_i = [pred[i] for pred in pred_arboles]
            preds_finales.append(Counter(pred_i).most_common(1)[0][0])
        
        return preds_finales
    
    def get_params():
    #TODO: get_params -> devuelve los hiperparametros no nulos    
        pass

    def set_params():
    #TODO: set_params
        pass

    def predict_proba():
    #TODO: predice las probabilidades de cada clase dado un array de instancias 
        pass

    def score():
    #TODO: devuelve la media de la accuracy+
        pass

    def decision_path():
    #TODO: decision_path(x)     
        pass
    