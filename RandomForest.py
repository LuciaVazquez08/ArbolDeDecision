from ArbolID3 import ArbolID3
from ArbolC4_5 import ArbolC4_5
import DecisionTreeClassifier
import numpy as np
from collections import Counter

class RandomForest():
    def __init__(self, algoritmo: ArbolID3 | ArbolC4_5 = ArbolID3, #no deberia ser "ID3" O "C4_5"
                 profundidad_max: int = None,
                 minimas_obs_n: int = None, 
                 minimas_obs_h: int = None, 
                 ganancia_minima: float = 0.0, 
                 numero_estimadores: int = 100, 
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
        self.arboles = []  #no se si es necesario que este como atrubuto pero bueno

    @staticmethod
    def bootstraping(X: np.ndarray , y: np.ndarray, n_estimadores: int) -> list[list[np.ndarray, np.ndarray]]:
        muestras = []
        n_muestras = len(y)
        random_state = np.random.RandomState(seed=42)

        for _ in range(n_estimadores):
            choices = random_state.choice(n_muestras, size=n_muestras, replace=True)
            new_X = X[choices]
            new_y = y[choices]
            muestras.append([new_X, new_y])
        return muestras
    
    @staticmethod
    def random_feature_selection(muestras_boostrapeadas: list[list[np.ndarray, np.ndarray]], feature_selection_method: str) -> list[list[np.ndarray, np.ndarray]]:
        muestras_finales = [] 

        random_state = np.random.RandomState(seed=42)
        numero_features = shape(muestras_boostrapeadas[0][0])[1]

        if feature_selection_method == "log":
            n_features = round(np.log(numero_features))
        elif feature_selection_method == "sqrt":
            n_features = round(np.sqrt(numero_features))
        elif feature_selection_method == "none":
            n_features = numero_features
        else:
            raise ValueError("No es un metodo valido de seleccion")

        for muestra in muestras_boostrapeadas:
            shape = shape(muestra[0])
            choices = random_state.choice(shape[1],size= n_features, replace=False)
            choices = sorted(choices)
            x_selec = muestra[0][:,choices]
            muestras_finales.append([x_selec, muestra[1]])
        
        return muestras_finales

    def fit(self, X: np.ndarray , y: np.ndarray) -> None:
        if len(X) == len(y):

            if self.bootstrap:
                muestras = RandomForest.bootstraping(X,y, self.numero_estimadores)
            else:
                muestras = [[X, y] for _ in range(self.numero_estimadores)]
                        
            muestras = RandomForest.random_feature_selection(muestras, feature_selection_method = self.feature_selection_method)

            for n in range(self.numero_estimadores):
                arbol = DecisionTreeClassifier(self.algoritmo, self.profundidad_max, self.minimas_obs_n, self.minimas_obs_h, self.ganancia_minima)
                arbol.fit(muestras[n][0], muestras[n][1])
                self.arboles.append(arbol)
        else:
            raise ValueError("debe haber la misma cantidad de instancias en los features y en el target")
        
    def predict(self, X: np.ndarray):
        predictions = []

        for arbol in self.arboles:
            pred = arbol.predict(X)
            predictions.append(pred)
        
        return Counter(predictions).most_common(1)[0][0]
    
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
    