import numpy as np
from pandas import DataFrame
from ArbolID3 import ArbolID3
from ArbolC4_5 import ArbolC4_5
from DecisionTreeClassifier import DecisionTreeClassifier
from collections import Counter

class RandomForest:
    def __init__(self, algoritmo: ArbolID3| ArbolC4_5 = ArbolID3, 
                 profundidad_max: int = None,
                 minimas_obs_n: int = None, 
                 minimas_obs_h: int = None, 
                 ganancia_minima: float = None,
                 top_atributos: int = 3,
                 umbral: float = 0.8, 
                 numero_estimadores: int = 100, 
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
        self.numero_estimadores = numero_estimadores
        self.bootstrap = bootstrap
        self.feature_selection_method = feature_selection_method
        self.arboles: list[DecisionTreeClassifier] = [] 

    @staticmethod
    def bootstraping(X: np.ndarray , y: np.ndarray, n_estimadores: int) -> list[list[np.ndarray]]:
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
    def random_feature_selection(muestras_boostrapeadas: list[list[np.ndarray]], feature_selection_method: str, nombres_atributos: list[str]) -> list[list[np.ndarray, list[str]]]:
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
        X_array = np.array(X)
        y_array = np.array(y)
        if len(X_array) == len(y_array):
            if self.bootstrap:
                muestras = RandomForest.bootstraping(X_array, y_array, self.numero_estimadores)
            else:
                muestras = [[X_array, y_array] for _ in range(self.numero_estimadores)]

            nombres_atributos = X.columns.tolist()              
            muestras = RandomForest.random_feature_selection(muestras, feature_selection_method=self.feature_selection_method, nombres_atributos=nombres_atributos)

            for n in range(self.numero_estimadores):
                arbol = DecisionTreeClassifier(self.algoritmo, self.profundidad_max, self.minimas_obs_n, self.minimas_obs_h, self.ganancia_minima, 
                                               self.top_atributos, self.umbral)
                arbol.fit(DataFrame(muestras[n][0], columns=muestras[n][2]), DataFrame(muestras[n][1]))
                self.arboles.append(arbol)
        else:
            raise ValueError("debe haber la misma cantidad de instancias en los features y en el target")
        
    def predict(self, X: DataFrame):
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
    