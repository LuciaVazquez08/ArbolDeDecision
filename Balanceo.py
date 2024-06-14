import numpy as np
import os
import pandas as pd
from collections import defaultdict

class Balanceo:
    @staticmethod
    def calcular_distancia(x1, x2):
        return np.linalg.norm(x1 - x2)

    @staticmethod
    def random_undersample(X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        target_classes = np.unique(y)
        keep_indices = []
        state = np.random.RandomState(42)

        size = np.inf
        for target_class in target_classes:
            class_indices = np.where(y == target_class)[0]
            class_size = len(class_indices)
            if class_size < size:
                size = class_size
        
        for target_class in target_classes:
            target_indices = np.where(y == target_class)[0]
            keep_index = state.choice(target_indices, size=size, replace=False)
            keep_indices.extend(keep_index)
        
        filtered_X = X[keep_indices, :]
        filtered_y = y[keep_indices, :]
        
        return filtered_X, filtered_y

    def random_oversample(X, y, target_classes=None, oversampling_ratio=1.0):
        target_classes = np.unique(y)
        keep_indices = []
        state = np.random.RandomState(42)

        size = -np.inf
        for target_class in target_classes:
            class_indices = np.where(y == target_class)[0]
            class_size = len(class_indices)
            if class_size > size:
                size = class_size
        
        for target_class in target_classes:
            target_indices = np.where(y == target_class)[0]
            keep_index = state.choice(target_indices, size=size, replace=True)
            keep_indices.extend(keep_index)
        
        filtered_X = X[keep_indices, :]
        filtered_y = y[keep_indices, :]
        
        return filtered_X, filtered_y
    
    @staticmethod
    def tomek_links(X, y):
        minority_classes = np.unique(y)
        
        toremove_indices = set()
        
        for minority_class in minority_classes:
            minority_indices = np.where(y == minority_class)[0]
            minority_samples = X[minority_indices]
            
            for i, minority_sample in enumerate(minority_samples):
                min_dist = float('inf')
                min_dist_idx = None
                for class_label in np.unique(y):
                    if class_label != minority_class:
                        class_indices = np.where(y == class_label)[0]
                        class_samples = X[class_indices]
                        for class_sample in class_samples:
                            distance = Balanceo.calcular_distancia(minority_sample, class_sample)
                            if distance < min_dist:
                                min_dist = distance
                                min_dist_idx = i
                
                if min_dist_idx is not None:
                    toremove_indices.add(minority_indices[min_dist_idx])
        
        filtered_X = np.delete(X, list(toremove_indices), axis=0)
        filtered_y = np.delete(y, list(toremove_indices))
        
        return filtered_X, filtered_y

    @staticmethod
    def nearmiss(X, y, n_neighbors=1):
        nearest_samples = defaultdict(list)
        
        minority_classes = np.unique(y)
        
        for minority_class in minority_classes:
            minority_indices = np.where(y == minority_class)[0]
            minority_samples = X[minority_indices]
            
            for minority_sample in minority_samples:
                distances = []
                for class_label in np.unique(y):
                    if class_label != minority_class:
                        class_indices = np.where(y == class_label)[0]
                        class_samples = X[class_indices]
                        for class_sample in class_samples:
                            distance = Balanceo.calcular_distancia(minority_sample, class_sample)
                            distances.append((distance, class_label))
                
                distances.sort()
                nearest_neighbors = distances[:n_neighbors]
                
                for distance, nearest_class in nearest_neighbors:
                    nearest_samples[minority_class].append((minority_sample, nearest_class))
        
        undersampled_X = []
        undersampled_y = []
        for minority_class, samples in nearest_samples.items():
            for sample, nearest_class in samples:
                undersampled_X.append(sample)
                undersampled_y.append(minority_class)
        
        return np.array(undersampled_X), np.array(undersampled_y)
    
    @staticmethod
    def smote(X, y, k_neighbors=5, oversampling_ratio=1.0):
        minority_classes = np.unique(y)
        
        synthetic_samples_per_class = defaultdict(int)
        for minority_class in minority_classes:
            minority_samples_count = np.sum(y == minority_class)
            synthetic_samples_count = int(minority_samples_count * oversampling_ratio) - minority_samples_count
            synthetic_samples_per_class[minority_class] = synthetic_samples_count
        
        synthetic_X = []
        synthetic_y = []
        
        for minority_class in minority_classes:
            minority_indices = np.where(y == minority_class)[0]
            
            minority_samples = X[minority_indices]
            
            for minority_sample in minority_samples:
                distances = []
                for class_label in np.unique(y):
                    if class_label != minority_class:
                        class_indices = np.where(y == class_label)[0]
                        class_samples = X[class_indices]
                        for class_sample in class_samples:
                            distance = Balanceo.calcular_distancia(minority_sample, class_sample)
                            distances.append((distance, class_label))
                
                distances.sort()
                nearest_neighbors = distances[:k_neighbors]
                
                selected_neighbor_index = np.random.randint(0, k_neighbors)
                selected_neighbor = nearest_neighbors[selected_neighbor_index]
                selected_neighbor_sample = minority_sample + (selected_neighbor[0] / 2) * (selected_neighbor[1] - minority_class)
                
                synthetic_X.append(selected_neighbor_sample)
                synthetic_y.append(minority_class)
        
        synthetic_X = np.vstack((X, np.array(synthetic_X)))
        synthetic_y = np.hstack((y, np.array(synthetic_y)))
        
        return synthetic_X, synthetic_y

if __name__ == "__main__":
    directorio_actual = os.getcwd()
    ruta_archivo = os.path.join(directorio_actual,"datasets/tratamiento.csv")
    df = pd.read_csv(ruta_archivo)

    balance = df['AdministrarTratamiento'].value_counts() 
    print(balance)

    X = df.drop(['AdministrarTratamiento', 'Paciente'], axis=1)
    y = df[['AdministrarTratamiento']]

    balanceo = Balanceo()

    x_val, y_val = balanceo.random_undersample(X,y)

    print(y_val.value_counts())

