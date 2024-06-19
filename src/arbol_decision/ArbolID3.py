import numpy as np
from arbol_decision.Arbol import Arbol
from arbol_decision.Ganancia import Ganancia
from typing import TypeVar
T = TypeVar('T')

class ArbolID3(Arbol):

    """
    Implementación del algoritmo de construcción ID3.

    Parámetros
    ----------
    dato : T 
        El dato almacenado en el nodo del árbol.

    atributo : str, default=None
        El atributo utilizado para dividir el conjunto de datos en el nodo actual.

    es_hoja : bool, default=False
        Indica si el nodo es una hoja o no.
    
    Atributos
    ---------
    _hijos : dict
        Diccionario que almacena los hijos (subárboles) de cada nodo.

    _num_samples : int 
        La cantidad de muestras almacenadas en cada nodo.
    """
    
    def __init__(self, dato = T, atributo: str = None, es_hoja: bool = False):
        super().__init__(dato) 
        self._es_hoja = es_hoja
        self._hijos = {}
        self._atributo = atributo
        self._num_samples = None

    def __str__(self, nivel=0) -> str:
        espacio_indentado = "    " * nivel
        if self._es_hoja:
            return f"{espacio_indentado}[Hoja: {self.dato}, Samples: {self._num_samples}]\n"
        else:
            nombre_atributo = self._atributo
            resultado = f"{espacio_indentado}[Atributo: {nombre_atributo}, Samples: {self._num_samples}]\n"
            for valor, hijo in self._hijos.items():
                resultado += f"{espacio_indentado}├── Valor: {valor}\n"
                resultado += hijo.__str__(nivel + 1)
            return resultado


    @classmethod
    def construir(cls, 
                  X: np.ndarray, 
                  y: np.ndarray, 
                  tipos_atributos: list[str],
                  indice_atributos: list[int],
                  nombres_atributos: list[str],
                  profundidad_max: int = None, 
                  minimas_obs_n: int = None, 
                  minimas_obs_h: int = None, 
                  ganancia_minima: float = None, 
                  profundidad_actual: int = 0
                  ) -> "ArbolID3":
        
        """
        Construye un árbol de decisión utilizando el algoritmo ID3.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de características del conjunto de datos de entrenamiento.

        y : np.ndarray
            Array con las etiquetas del conjunto de datos de entrenamiento.

        tipos_atributos : list[str]
            Lista que contiene los tipos de atributos en X, categóricos o continuos.
        
        indice_atributos : list[int]
            Lista que contiene los índices de los atributos en X.
        
        nombres_atributos : list[str]
            Lista que contiene los nombres de los atributos en X.

        profundidad_max : int, default=None 
            La profundidad máxima que puede alcanzar el árbol.

        minimas_obs_n : int, default=None 
            La cantidad mínima de observaciones requeridas para dividir un nodo interno. 

        minimas_obs_h : int, default=None 
            La cantidad mínima de observaciones requeridas presentes en una hoja. 

        ganancia_minima : float, default=None 
            La ganancia mínima al elegir el mejor atributo para dividir un nodo.

        profundidad_actual : int, default=0
            Profundidad actual del nodo en la construcción del árbol.

        Returns
        -------
        ArbolID3 : Devuelve un objeto ArbolID3 del árbol de decisión construido.
        """
        
        # Criterio de parada: Nodo puro 
        if len(np.unique(y)) == 1:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolID3(clase_mayoritaria, atributo=None, es_hoja=True)
            hoja._num_samples = len(y)
            return hoja
            
        # Criterio de parada: Maxima profundidad
        if profundidad_max is not None and profundidad_actual >= profundidad_max:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolID3(clase_mayoritaria, atributo=None, es_hoja=True)
            hoja._num_samples = len(y)
            return hoja
        
        # Criterio de parada: Mínimas observaciones por nodo
        if minimas_obs_n is not None and len(y) < minimas_obs_n:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolID3(clase_mayoritaria, atributo=None, es_hoja=True)
            hoja._num_samples = len(y)
            return hoja
        
        # Criterio de parada: Sin atributos para dividir
        if not indice_atributos:  
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolID3(clase_mayoritaria, atributo=None, es_hoja=True)
            hoja._num_samples = len(y)
            return hoja
        
        # Calculamos la ganancia de información de cada atributo
        ganancias = [Ganancia.ganancia_informacion_atributo(X, y, atributo) for atributo in indice_atributos]

        # Criterio de parada: Ganancia mínima
        if ganancia_minima is not None and ganancias[np.argmax(ganancias)] < ganancia_minima:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolID3(clase_mayoritaria, atributo=None, es_hoja=True)
            hoja._num_samples = len(y)
            return hoja

        # Seleccionamos el atributo con mayor ganancia y creamos un arbol con ese atributo
        mejor_atributo = indice_atributos[np.argmax(ganancias)]
        arbol = ArbolID3(mejor_atributo, atributo=nombres_atributos[mejor_atributo])
        arbol._num_samples = len(y)
        
        # Creamos nodos para cada valor del mejor atributo
        for valor in np.unique(X[:, mejor_atributo]):
                
            atributos_restantes = indice_atributos.copy()
            atributos_restantes.remove(mejor_atributo)
        
            indices = np.where(X[:, mejor_atributo] == valor)[0]
            sub_X = X[indices]
            sub_y = y[indices]

            # Criterio de parada: Mínimas observaciones por hoja
            if minimas_obs_h is not None and minimas_obs_h > len(sub_y):
                clase_mayoritaria = cls.clase_mayoritaria(sub_y)
                subarbol = ArbolID3(clase_mayoritaria, atributo=None, es_hoja=True)
                subarbol._num_samples = len(sub_y)
            else:
                subarbol = cls.construir(sub_X, sub_y, tipos_atributos, atributos_restantes, nombres_atributos, profundidad_max, minimas_obs_n, minimas_obs_h, ganancia_minima, profundidad_actual + 1)
                
            arbol._hijos[valor] = subarbol

        return arbol

    @staticmethod
    def clase_mayoritaria(y: np.ndarray) -> int:
        """
        Obtiene la clase mayoritaria en un conjunto de etiquetas y.

        Parámetros
        ----------
        y : np.ndarray
            Array de etiquetas del cual se desea encontrar la clase mayoritaria.

        Returns
        -------
        int : Devuelve la clase que tiene la mayor frecuencia en el array de etiquetas.
        """
        clases, conteo = np.unique(y, return_counts=True)
        return clases[np.argmax(conteo)]
    

    
    
        
        
    

    
        
        
    
