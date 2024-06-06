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
                 numero_estimadores: int = 5, # cantidad de árboles que queremos construir
                 bootstrap: bool = True,
                 feature_selection_method: str = "sqrt"
                ):
        self.algoritmo = algoritmo
        self.profundidad_max = profundidad_max
        self.minimas_obs_n = minimas_obs_n
        self.minimas_obs_h = minimas_obs_h
        self.ganancia_minima = ganancia_minima
        self.numero_estimadores = numero_estimadores
        self.bootstrap = bootstrap
        self.feature_selection_method = feature_selection_method
        self.arboles: list[DecisionTreeClassifier] = []  #no se si es necesario que este como atrubuto pero bueno

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
    def random_feature_selection(muestras_boostrapeadas: list[list[np.ndarray]], feature_selection_method: str) -> list[list[np.ndarray]]:
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
            raise ValueError("No es un metodo valido de seleccion")

        for muestra in muestras_boostrapeadas:
            choices = random_state.choice(numero_features, size=n_features, replace=False)
            choices = sorted(choices)
            x_selec = muestra[0][:, choices]
            muestras_finales.append([x_selec, muestra[1]])
        
        return muestras_finales

    def fit(self, X: DataFrame, y: DataFrame) -> None:
        X_array = X.values
        y_array = y.values

        if len(X_array) == len(y_array):
            if self.bootstrap:
                muestras = RandomForest.bootstraping(X_array, y_array, self.numero_estimadores)
            else:
                muestras = [[X_array, y_array] for _ in range(self.numero_estimadores)]
                        
            muestras = RandomForest.random_feature_selection(muestras, feature_selection_method=self.feature_selection_method)

            for n in range(self.numero_estimadores):
                arbol = DecisionTreeClassifier(self.algoritmo, self.profundidad_max, self.minimas_obs_n, self.minimas_obs_h, self.ganancia_minima)
                arbol.fit(DataFrame(muestras[n][0]), DataFrame(muestras[n][1]))
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

        print(f'Predicciones de cada árbol: {pred_arboles}')
        print(f'Predicciones finales: {preds_finales}')
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
    