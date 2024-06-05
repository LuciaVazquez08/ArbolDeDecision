from Entropia import Entropia
import numpy as np
from Arbol import Arbol

class ArbolC4_5:
    
    def __init__(self, dato=None, es_hoja=False):
        self.dato = dato
        self.es_hoja = es_hoja
        self.hijos = {}

    @classmethod
    def construir(cls, X, y, atributos, profundidad_max=None, minimas_obs_n=None, minimas_obs_h=None, ganancia_minima=0.0, profundidad_actual=0):
        # Criterios de parada
        if len(np.unique(y)) == 1:  # Nodo puro
            return cls(dato=y[0], es_hoja=True)
        if profundidad_max is not None and profundidad_actual >= profundidad_max:  # Máxima profundidad
            clase_mayoritaria = cls.clase_mayoritaria(y)
            return cls(dato=clase_mayoritaria, es_hoja=True)
        if minimas_obs_n is not None and len(y) < minimas_obs_n:  # Mínimas observaciones por nodo
            clase_mayoritaria = cls.clase_mayoritaria(y)
            return cls(dato=clase_mayoritaria, es_hoja=True)
        if not atributos:  # Sin atributos para dividir
            clase_mayoritaria = cls.clase_mayoritaria(y)
            return cls(dato=clase_mayoritaria, es_hoja=True)

        # Seleccionar el mejor atributo
        mejor_atributo, mejor_umbral = cls.seleccionar_mejor_atributo(X, y, atributos)

        # Creamos el árbol con el mejor atributo
        arbol = cls(mejor_atributo)

        if mejor_umbral is not None:  # El mejor atributo es continuo
            indices_izq = np.where(X[:, mejor_atributo] <= mejor_umbral)[0]
            indices_der = np.where(X[:, mejor_atributo] > mejor_umbral)[0]

            sub_X_izq = X[indices_izq]
            sub_y_izq = y[indices_izq]
            sub_X_der = X[indices_der]
            sub_y_der = y[indices_der]

            if minimas_obs_h is not None and minimas_obs_h > len(sub_y_izq):
                clase_mayoritaria = cls.clase_mayoritaria(sub_y_izq)
                sub_arbol_izq = cls(clase_mayoritaria, es_hoja=True)
            else:
                sub_arbol_izq = cls.construir(sub_X_izq, sub_y_izq, atributos, profundidad_max, minimas_obs_n, minimas_obs_h, ganancia_minima, profundidad_actual + 1)

            if minimas_obs_h is not None and minimas_obs_h > len(sub_y_der):
                clase_mayoritaria = cls.clase_mayoritaria(sub_y_der)
                sub_arbol_der = cls(clase_mayoritaria, es_hoja=True)
            else:
                sub_arbol_der = cls.construir(sub_X_der, sub_y_der, atributos, profundidad_max, minimas_obs_n, minimas_obs_h, ganancia_minima, profundidad_actual + 1)

            arbol.hijos[('<=', mejor_umbral)] = sub_arbol_izq
            arbol.hijos[('>', mejor_umbral)] = sub_arbol_der
        else:  # El mejor atributo es categórico
            for valor in np.unique(X[:, mejor_atributo]):
                atributos_restantes = atributos.copy()
                atributos_restantes.remove(mejor_atributo)

                indices = np.where(X[:, mejor_atributo] == valor)[0]
                sub_X = X[indices]
                sub_y = y[indices]

                # Criterio de parada: Mínimas observaciones por hoja
                if minimas_obs_h is not None and minimas_obs_h > len(sub_y):
                    clase_mayoritaria = cls.clase_mayoritaria(sub_y)
                    sub_arbol = cls(clase_mayoritaria, es_hoja=True)
                else:
                    sub_arbol = cls.construir(sub_X, sub_y, atributos_restantes, profundidad_max, minimas_obs_n, minimas_obs_h, ganancia_minima, profundidad_actual + 1)

                arbol.hijos[valor] = sub_arbol

        return arbol

    @staticmethod
    def determinar_tipo_atributo(X: np.ndarray, top_n: int = 3, umbral: float = 0.8) -> str:
        tipo_atributo = 'continuo'  

        valores_unicos, conteos = np.unique(X, return_counts=True)
        
        if len(conteos) > 0:
        
            top_n = min(top_n, len(conteos))

    
            indices_ordenados = np.argsort(conteos)[::-1]
            top_conteos = conteos[indices_ordenados][:top_n]

            proporcion = np.sum(top_conteos) / len(X)

    
            if proporcion >= umbral:
                tipo_atributo = 'categorico'

        return tipo_atributo

    @staticmethod
    def seleccionar_mejor_atributo(X, y, atributos):
        mejor_ganancia = -np.inf
        mejor_atributo = None
        mejor_umbral = None

        for atributo in atributos:
            valores_atributo = X[:, atributo]

            # Verificar el tipo de atributo 
            tipo_atributo = ArbolC4_5.determinar_tipo_atributo(valores_atributo)

            if tipo_atributo == 'continuo':
                umbral, ganancia = ArbolC4_5.obtener_umbral_y_gain_ratio(valores_atributo, y)
            else:
                ganancia = Entropia.ganancia_informacion_atributo(X, y, atributo)
                umbral = None

            if ganancia > mejor_ganancia:
                mejor_ganancia = ganancia
                mejor_atributo = atributo
                mejor_umbral = umbral

        return mejor_atributo, mejor_umbral

    @staticmethod
    def obtener_umbral_y_gain_ratio(atributo_continuo, y):
        ganancia_maxima = -1
        umbral_optimo = None
        
        # Ordenar los valores únicos del atributo continuo
        valores_unicos = np.sort(np.unique(atributo_continuo))
        
        for i in range(len(valores_unicos) - 1):
            umbral = (valores_unicos[i] + valores_unicos[i + 1]) / 2
            
            grupo_1_y = y[atributo_continuo <= umbral]
            grupo_2_y = y[atributo_continuo > umbral]

            n_izquierda = len(grupo_1_y)
            n_derecha = len(grupo_2_y)
            
            if n_izquierda == 0 or n_derecha == 0:
                continue
            
            # Crear una matriz X para gain_ratio
            X_dividido = np.concatenate((np.zeros(n_izquierda), np.ones(n_derecha)))
            y_dividido = np.concatenate((grupo_1_y, grupo_2_y))
            
            gain_ratio = Entropia.gain_ratio(X_dividido.reshape(-1, 1), y_dividido, 0)
            
            if gain_ratio > ganancia_maxima:
                ganancia_maxima = gain_ratio
                umbral_optimo = umbral
        
        return umbral_optimo, ganancia_maxima

    @staticmethod
    def clase_mayoritaria(y):
        valores, counts = np.unique(y, return_counts=True)
        return valores[np.argmax(counts)]

if __name__ == "__main__":
    # Datos ficticios: combinamos columnas continuas (floats) y una categórica (enteros distintos a las clases)
    X = np.array([
        [2, 5.1, 3.5, 1.4, 0.2],  # Clase 0
        [3, 4.9, 3.0, 1.4, 0.2],  # Clase 1
        [1, 4.7, 3.2, 1.3, 0.2],  # Clase 0
        [2, 4.6, 3.1, 1.5, 0.2],  # Clase 1
        [3, 5.0, 3.6, 1.4, 0.2],  # Clase 0
        [1, 5.4, 3.9, 1.7, 0.4],  # Clase 1
        [2, 4.6, 3.4, 1.4, 0.3],  # Clase 0
        [3, 5.0, 3.4, 1.5, 0.2],  # Clase 1
        [1, 4.4, 2.9, 1.4, 0.2],  # Clase 0
        [3, 4.9, 3.1, 1.5, 0.1],  # Clase 1
        [1, 5.5, 3.5, 1.3, 0.2],  # Clase 0
        [2, 5.7, 3.8, 1.7, 0.3],  # Clase 1
        [1, 5.0, 3.0, 1.6, 0.2],  # Clase 0
        [2, 5.1, 3.8, 1.9, 0.4],  # Clase 1
        [1, 5.2, 3.5, 1.5, 0.2],  # Clase 0
        [2, 5.2, 3.4, 1.4, 0.2],  # Clase 1
        [1, 4.8, 3.1, 1.6, 0.2],  # Clase 0
        [2, 5.4, 3.4, 1.5, 0.4],  # Clase 1
        [1, 5.2, 4.1, 1.5, 0.1],  # Clase 0
        [2, 5.5, 4.2, 1.4, 0.2],  # Clase 1
    ])

    # Supongamos que las clases para cada instancia son binarias (0 o 1)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Atributos a considerar para el árbol de decisión (índices de las columnas)
    atributos = list(range(X.shape[1]))

    # Ajustar los parámetros del árbol
    profundidad_max = 8
    minimas_obs_n = 1
    minimas_obs_h = 1
    ganancia_minima = 0.0

    # Construir el árbol C4.5 con los datos ficticios
    arbol = ArbolC4_5.construir(X, y, atributos, profundidad_max, minimas_obs_n, minimas_obs_h, ganancia_minima)

    # Función para imprimir el árbol de decisión
    def imprimir_arbol(arbol, nivel=0):
        if arbol.es_hoja:
            print("  " * nivel + f"Clase: {arbol.dato}")
        else:
            print("  " * nivel + f"Atributo: {arbol.dato}")
            for valor, hijo in arbol.hijos.items():
                print("  " * (nivel + 1) + f"{valor}:")
                imprimir_arbol(hijo, nivel + 2)

    # Imprimir el árbol construido
    imprimir_arbol(arbol)


    
            
