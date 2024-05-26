from typing import List

class DecissionTreeClassifier:
    def __init__(self):
        self.tree = None

    # TODO: Implementar la construcción del árbol con el conjunto de entrenamiento
    def fit(self, X: List[List[float]], y: List[str]):
        pass 

    # TODO: Implementar la clasificación en base a los X recibidos -> devuelve la clase predecida para cada X
    def predict(self, X: List[List[float]]) -> List[str]:
        predicciones = []
        for instancia in X:
            predicciones.append(self._predict_instancia(instancia, self.tree))
        return predicciones
    
    # Implementa la predicción para una instancia específica 
    def _predict_instancia(self, instance: List[float], tree_node) -> str:
        pass

    # TODO: Implementar transform: Se encarga del encoding