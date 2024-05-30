from ArbolID3 import ArbolID3
from ArbolC4_5 import ArbolC4_5
import DecisionTreeClassifier
import numpy as np
import random

class RandomForest():
    def __init__(self, algoritmo: ArbolID3 | ArbolC4_5, 
                 profundidad_max: int = None,
                 minimas_obs_n: int = None, 
                 minimas_obs_h: int = None, 
                 ganancia_minima: float = 0.0, 
                 numero_estimadores: int = 100, 
                 bootstrap: bool = True
                ):
        self.algoritmo = algoritmo
        self.profundidad_max = profundidad_max
        self.minimas_obs_n = minimas_obs_n
        self.minimas_obs_h = minimas_obs_h
        self.ganancia_minima = ganancia_minima
        self.numero_estimadores = numero_estimadores
        self.bootstrap = bootstrap
        self.arboles = []  #no se si es necesario que este como atrubuto pero bueno

    @staticmethod
    def bootstraping(X: np.ndarray , y: np.ndarray, n_estimadores: int) -> list[list[np.ndarray, np.ndarray]]:
        muestras = []
        n_muestras = len(y)
        for _ in range(n_estimadores):
            choices = np.random.choice(n_muestras, size=n_muestras, replace=True)
            new_X = X[choices]
            new_y = y[choices]
            muestras.append([new_X, new_y])
        return muestras

    def fit(self, X: np.ndarray , y: np.ndarray) -> None:
        if self.bootstrap:
            muestras = bootstraping(X,y, self.numero_estimadores)
        else:
            muestras = [[X, y] for _ in range(self.numero_estimadores)]

        for n in range(self.numero_estimadores):
            arbol = DecisionTreeClassifier(self.algoritmo, self.profundidad_max, self.minimas_obs_n, self.minimas_obs_h, self.ganancia_minima)
            arbol.fit(muestras[n][0], muestras[n][1])
            self.arboles.append(arbol)
        

    def predict(self, X: np.ndarray):
        for arbol in self.arboles:
            pred = arbol.predict(X)

    #TODO: aply(self, X) -> devulñeve pára cada arbol del bosque, en que hoja quedo el feature
    #TODO: decision_path(x)
