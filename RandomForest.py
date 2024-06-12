import numpy as np
from pandas import DataFrame
from ArbolID3 import ArbolID3
from ArbolC4_5 import ArbolC4_5
from DecisionTreeClassifier import DecisionTreeClassifier
from collections import Counter
from Balanceo import Balanceo

class RandomForestClassifier:
    def __init__(self, algoritmo: str = "ID3", 
                 profundidad_max: int = None,
                 minimas_obs_n: int = None, 
                 minimas_obs_h: int = None, 
                 ganancia_minima: float = None, 
                 numero_estimadores: int = 5,
                 bootstrap: bool = True,
                 feature_selection_method: str = "sqrt",
                 tecnica_balanceo: str = None,
                 class_weight: str | dict = None
                ):
        self.algoritmo = algoritmo
        self.profundidad_max = profundidad_max
        self.minimas_obs_n = minimas_obs_n
        self.minimas_obs_h = minimas_obs_h
        self.ganancia_minima = ganancia_minima
        self.numero_estimadores = numero_estimadores
        self.bootstrap = bootstrap
        self.feature_selection_method = feature_selection_method
        self.class_weight = class_weight
        self.tecnica_balanceo = tecnica_balanceo
        self.arboles: list[DecisionTreeClassifier] = []  
    
    @staticmethod
    def balancear(X: np.ndarray, y: np.ndarray, weights: dict) -> str:
        return ""

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
        X_array = np.asarray(X)
        y_array = np.array(y)

        if len(X_array) == len(y_array):
            if self.class_weight:
                if self.class_weight == "balanced":
                    pesos = {}
                    total_obs = len(y_array)
                    clases = y_array.unique()
                    cantidad = len(clases)
                    for clase in clases:
                        obs_clase = np.count_nonzero(y_array == clase)
                        pesos[clase] = total_obs / (cantidad * obs_clase)
                    X_array, y_array = RandomForestClassifier.balancear(X_array,y_array,pesos)
                elif self.class_weight.isinstance(dict):
                    X_array,y_array = RandomForestClassifier.balancear(X_array,y_array, self.class_weight)
                else:
                    ValueError("Las opciones son balanceado o un diccionario con porcentajes")
            if self.tecnica_balanceo:
                if self.tecnica_balanceo == "RandomUnder":
                    X_array, y_array = Balanceo.random_undersample(X_array,y_array)
                elif self.tecnica_balanceo == "RandomOver":
                    X_array, y_array = Balanceo.random_oversample(X_array,y_array)
                elif self.tecnica_balanceo == "TomekLinks":
                    X_array, y_array = Balanceo.tomek_links(X_array,y_array)
                elif self.tecnica_balanceo == "SMOTE":
                    X_array, y_array = Balanceo.smote(X_array,y_array)
                elif self.tecnica_balanceo == "NearMiss":
                    X_array, y_array = Balanceo.nearmiss(X_array,y_array)
                else:
                    raise ValueError("las opciones validas son RandomUnder, RandomOver, TomekLinks, SMOTE y Nearmiss")
            if self.bootstrap:
                muestras = RandomForestClassifier.bootstraping(X_array, y_array, self.numero_estimadores)
            else:
                muestras = [[X_array, y_array] for _ in range(self.numero_estimadores)]
                        
            muestras = RandomForestClassifier.random_feature_selection(muestras, feature_selection_method=self.feature_selection_method)

            for n in range(self.numero_estimadores):
                arbol = DecisionTreeClassifier(self.algoritmo, self.profundidad_max, self.minimas_obs_n, self.minimas_obs_h, self.ganancia_minima)
                arbol.fit(DataFrame(muestras[n][0]), DataFrame(muestras[n][1]))
                self.arboles.append(arbol)
        else:
            raise ValueError("debe haber la misma cantidad de instancias en los features y en el target")
        
    def predict(self, X: DataFrame):
        pred_arboles = []

        for arbol in self.arboles:
            pred = arbol.predict(X)
            pred_arboles.append(pred)

        pred_finales = []
        for i in range(len(X)):
            pred_i = [pred[i] for pred in pred_arboles]
            pred_finales.append(Counter(pred_i).most_common(1)[0][0])

        print(f'Predicciones de cada árbol: {pred_arboles}')
        print(f'Predicciones finales: {pred_finales}')
        return pred_finales
    
    def get_params(self):
        return self.__dict__

    def set_params(self, **params):
        for key, value in params.items():
                if hasattr(self,key):
                    setattr(self,key,value)
                else:
                    raise ValueError(f"{key} no es un atributo valido")

    def predict_proba(self, X: DataFrame):
        n_samples = X.shape[0]
        cantidades = {c: np.zeros(n_samples) for c in np.unique(self.arboles[0].predict(X))}

        for arbol in self.arboles:
            pred = arbol.predict(X)
            for i, pred in enumerate(pred):
                cantidades[pred][i] += 1

        prob = np.zeros((n_samples, len(cantidades)))
        for i, cls in enumerate(cantidades):
            prob[:, i] = cantidades[cls] / len(self.arboles)
        
        return prob

    def score(self, X,y):
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
        paths = []

        for arbol in self.arboles:
            paths.append(arbol.decision_path(X)) 

        return paths
    