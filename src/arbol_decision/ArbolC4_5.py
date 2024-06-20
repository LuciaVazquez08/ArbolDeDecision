import pandas as pd
from arbol_decision.Ganancia import Ganancia
from arbol_decision.Arbol import Arbol
import numpy as np
from typing import Counter, TypeVar
T = TypeVar('T')

class ArbolC4_5(Arbol):

    """
    Implementación del algoritmo de construcción C4.5.

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
                  ) -> "ArbolC4_5":
        
        """
        Construye un árbol de decisión utilizando el algoritmo C4.5.

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
        ArbolC4_5 : Devuelve un objeto ArbolC4_5 del árbol de decisión construido.
        """

        # Criterio de parada: Nodo puro 
        if len(np.unique(y)) == 1:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolC4_5(clase_mayoritaria, atributo=None, es_hoja=True)
            hoja._num_samples = len(y)
            return hoja
            
        # Criterio de parada: Maxima profundidad
        if profundidad_max is not None and profundidad_actual >= profundidad_max:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolC4_5(clase_mayoritaria, atributo=None, es_hoja=True)
            hoja._num_samples = len(y)
            return hoja
        
        # Criterio de parada: Mínimas observaciones por nodo
        if minimas_obs_n is not None and len(y) < minimas_obs_n:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolC4_5(clase_mayoritaria, atributo=None, es_hoja=True)
            hoja._num_samples = len(y)
            return hoja
        
        # Criterio de parada: Sin atributos para dividir
        if not indice_atributos:  
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolC4_5(clase_mayoritaria, atributo=None, es_hoja=True)
            hoja._num_samples = len(y)
            return hoja

        # Seleccionar el mejor atributo
        mejor_atributo, mejor_umbral, mejor_ganancia = cls.seleccionar_mejor_atributo(X, y, tipos_atributos, indice_atributos)

        # Criterio de parada: Ganancia mínima (verifica si la ganancia supera el umbral mínimo)
        if ganancia_minima is not None and np.all(mejor_ganancia < ganancia_minima):
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolC4_5(clase_mayoritaria, atributo=None, es_hoja=True)
            hoja._num_samples = len(y)
            return hoja

        # Creamos el árbol con el mejor atributo
        arbol = ArbolC4_5(mejor_atributo, atributo=nombres_atributos[mejor_atributo])
        arbol._num_samples = len(y)

        # Caso 1: El mejor atributo es continuo
        if mejor_umbral is not None:  

            atributos_restantes = indice_atributos.copy()
            atributos_restantes.remove(mejor_atributo)

            indices_izq = np.where(X[:, mejor_atributo] <= mejor_umbral)[0]
            indices_der = np.where(X[:, mejor_atributo] > mejor_umbral)[0]

            sub_X_izq = X[indices_izq]
            sub_y_izq = y[indices_izq]
            sub_X_der = X[indices_der]
            sub_y_der = y[indices_der]

            # Criterio de parada: Mínimas observaciones por hoja (no anda)
            if minimas_obs_h is not None and minimas_obs_h > len(sub_y_izq): 
                clase_mayoritaria = cls.clase_mayoritaria(sub_y_izq)
                sub_arbol_izq = ArbolC4_5(clase_mayoritaria, atributo=None, es_hoja=True)
                sub_arbol_izq._num_samples = len(sub_y_izq)
            else:
                sub_arbol_izq = cls.construir(sub_X_izq, sub_y_izq, tipos_atributos, atributos_restantes, nombres_atributos, profundidad_max, minimas_obs_n, minimas_obs_h, ganancia_minima, profundidad_actual + 1)


            # Criterio de parada: Mínimas observaciones por hoja (no anda)
            if minimas_obs_h is not None and minimas_obs_h > len(sub_y_der): 
                clase_mayoritaria = cls.clase_mayoritaria(sub_y_der)
                sub_arbol_der = ArbolC4_5(clase_mayoritaria, atributo=None, es_hoja=True)
                sub_arbol_der._num_samples = len(sub_y_der)
            else:
                sub_arbol_der = cls.construir(sub_X_der, sub_y_der, tipos_atributos, atributos_restantes, nombres_atributos, profundidad_max, minimas_obs_n, minimas_obs_h, ganancia_minima, profundidad_actual + 1)

            arbol._hijos[('<=', mejor_umbral)] = sub_arbol_izq
            arbol._hijos[('>', mejor_umbral)] = sub_arbol_der

        # Caso 2: El mejor atributo es categórico
        else:  
            for valor in np.unique(X[:, mejor_atributo]):
                
                atributos_restantes = indice_atributos.copy()
                atributos_restantes.remove(mejor_atributo)
            
                indices = np.where(X[:, mejor_atributo] == valor)[0]
                sub_X = X[indices]
                sub_y = y[indices]

                # Criterio de parada: Mínimas observaciones por hoja
                if minimas_obs_h is not None and minimas_obs_h > len(sub_y):
                    clase_mayoritaria = cls.clase_mayoritaria(sub_y)
                    subarbol = ArbolC4_5(clase_mayoritaria, atributo=None, es_hoja=True)
                    subarbol._num_samples = len(sub_y)
                else:
                    subarbol = cls.construir(sub_X, sub_y, tipos_atributos, atributos_restantes, nombres_atributos, profundidad_max, minimas_obs_n, minimas_obs_h, ganancia_minima, profundidad_actual + 1)
                    
                arbol._hijos[valor] = subarbol
        return arbol
    

    @staticmethod
    def seleccionar_mejor_atributo(X, y, tipos_atributos, atributos):
        """
        Selecciona el mejor atributo para dividir el conjunto de datos basado en la ganancia de información.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de características del conjunto de datos de entrenamiento.
            
        y : np.ndarray
            Array con las etiquetas del conjunto de datos de entrenamiento.
            
        tipos_atributos : list[str]
            Lista que contiene los tipos de atributos en X, categóricos o continuos.
            
        atributos : list[int]
            Lista de índices de los atributos a considerar para la selección.

        Returns
        -------
        mejor_atributo : int
            El índice del mejor atributo para la división.
            
        mejor_umbral : float
            El umbral óptimo para la división en caso de que el atributo sea continuo, 
            o None si el atributo es categórico.
        """
        mejor_ganancia = -np.inf
        mejor_atributo = None
        mejor_umbral = None

        for atributo in atributos:      
            valores_atributo = X[:, atributo]
            tipo_atributo = tipos_atributos[atributo]

            if tipo_atributo == 'continuo':
                umbral, ganancia = ArbolC4_5.obtener_umbral_y_gain_ratio(valores_atributo, y)
            else:
                ganancia = Ganancia.ganancia_informacion_atributo(X, y, atributo)
                umbral = None

            if ganancia > mejor_ganancia:
                mejor_ganancia = ganancia
                mejor_atributo = atributo
                mejor_umbral = umbral

        return mejor_atributo, mejor_umbral, mejor_ganancia
    
    # AGREGAR DOCS
    def imputar_valores_faltantes(self, X: np.ndarray) -> np.ndarray:
           
            X_imputado = X.copy()

            for atributo in range(X.shape[1]):
                columna_atributo = X[:, atributo]
                valores_faltantes = pd.isnull(columna_atributo)  

                if np.any(valores_faltantes):
                    tipo_atributo = self.determinar_tipo_atributo(columna_atributo[~valores_faltantes], top_n=3, umbral=0.8)

                    if tipo_atributo == "categorico":
            
                        valores_clase = Counter(columna_atributo[~valores_faltantes])
                        valor_mas_comun = max(valores_clase, key=valores_clase.get)
                        X_imputado[valores_faltantes, atributo] = valor_mas_comun
                    elif tipo_atributo == "continuo":
                    
                        media_atributo = np.nanmean(columna_atributo[~valores_faltantes].astype(float))
                        media_atributo = round(float(media_atributo), 2)  
                        X_imputado[valores_faltantes, atributo] = media_atributo

            print("Conjunto de datos después de imputar valores faltantes:")
            print(X_imputado)

            return X_imputado
    
    
    @staticmethod
    def determinar_tipo_atributo(atributo: np.ndarray, top_n: int, umbral: float) -> str:
        """
        Determina el tipo de un atributo (categórico o continuo) basado en la proporción 
        de sus valores más frecuentes.

        Parámetros
        ----------
        atributo : np.ndarray
            Array de valores del atributo a analizar (columna con todos los valores).
            
        top_n : int
            Número de valores más frecuentes a considerar para calcular la proporción.
            
        umbral : float
            Umbral de proporción para decidir si el atributo es categórico o continuo.

        Returns
        -------
        str : 'categorico' si la proporción de los top_n valores es mayor o igual al umbral, 
              'continuo' en caso contrario.
        """

        if np.issubdtype(atributo.dtype, np.number):
            atributo = atributo[~np.isnan(atributo)]

        valores_unicos, conteos = np.unique(atributo, return_counts=True)

        # Si la proporción de los top_n valores es alta, se considera categórico
        top_n = min(top_n, len(conteos))
        indices_ordenados = np.argsort(conteos)[::-1]
        top_conteos = conteos[indices_ordenados][:top_n]
        proporcion = np.sum(top_conteos) / len(atributo)

        if proporcion >= umbral:
            return 'categorico'
        else:
            return 'continuo'    

    @staticmethod
    def obtener_umbral_y_gain_ratio(atributo_continuo, y):
        """
        Calcula el umbral óptimo y el gain ratio para un atributo continuo.

        Parámetros
        ----------
        atributo_continuo : np.ndarray
            Array de valores del atributo continuo a analizar (columna con todos los valores).
            
        y : np.ndarray
            Array con las etiquetas asociado al conjunto de datos de entrenamiento.

        Returns
        -------
        tuple : Un par (umbral_optimo, ganancia_maxima) donde:
                - umbral_optimo : float
                    El valor óptimo del umbral que maximiza el gain ratio.
                - ganancia_maxima : float
                    El valor máximo del gain ratio obtenido para el umbral óptimo.
        """
        ganancia_maxima = -1
        umbral_optimo = None
        
        # Ordena los valores únicos del atributo continuo
        valores_unicos = np.sort(np.unique(atributo_continuo))
        
        for i in range(len(valores_unicos) - 1):
            umbral = (valores_unicos[i] + valores_unicos[i + 1]) / 2
            
            grupo_1_y = y[atributo_continuo <= umbral]
            grupo_2_y = y[atributo_continuo > umbral]

            n_izquierda = len(grupo_1_y)
            n_derecha = len(grupo_2_y)
            
            if n_izquierda == 0 or n_derecha == 0:
                continue
            
            # Crea una matriz X para gain_ratio
            X_dividido = np.concatenate((np.zeros(n_izquierda), np.ones(n_derecha)))
            y_dividido = np.concatenate((grupo_1_y, grupo_2_y))
            
            gain_ratio = Ganancia.gain_ratio(X_dividido.reshape(-1, 1), y_dividido, 0)
            
            if gain_ratio > ganancia_maxima:
                ganancia_maxima = gain_ratio
                umbral_optimo = umbral
        
        return umbral_optimo, ganancia_maxima

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



    
            

    
            


    
            
