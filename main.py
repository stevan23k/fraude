import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class SemiSupervisedFraudDetection:
    def __init__(self, confidence_threshold=0.8):
        self.confidence_threshold = confidence_threshold
        self.random_forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def preprocess_data(self, data):
        df = data.copy()
        
        # Extraer características temporales
        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
        
        # Codificar variables categóricas
        categorical_columns = ['Location', 'PaymentMethod']
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                df[column] = self.label_encoders[column].fit_transform(df[column])
            else:
                df[column] = self.label_encoders[column].transform(df[column])
        
        features = ['Amount', 'Location', 'PaymentMethod', 'Hour', 'DayOfWeek']
        X = df[features]
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=features)
    
    def balance_data(self, X, y):
        """Balance data using random duplication for extreme imbalance cases"""
        classes, counts = np.unique(y, return_counts=True)
        max_count = max(counts)
        
        X_balanced = X.copy()
        y_balanced = y.copy()
        
        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            if len(cls_indices) < max_count:
                n_needed = max_count - len(cls_indices)
                duplicate_indices = np.random.choice(cls_indices, size=n_needed, replace=True)
                X_balanced = pd.concat([X_balanced, X.iloc[duplicate_indices]])
                y_balanced = pd.concat([y_balanced, pd.Series([cls] * n_needed)])
        
        return X_balanced, y_balanced
    
    def split_labeled_unlabeled(self, X, y):
        labeled_mask = ~y.isna()
        X_labeled = X[labeled_mask]
        y_labeled = y[labeled_mask]
        X_unlabeled = X[~labeled_mask]
        
        return X_labeled, y_labeled, X_unlabeled
    
    def train_initial_model(self, X_labeled, y_labeled):
        self.random_forest.fit(X_labeled, y_labeled)
    
    def predict_probabilities(self, X):
        return self.random_forest.predict_proba(X)
    
    def self_training(self, X_labeled, y_labeled, X_unlabeled, max_iterations=5):
        current_X = X_labeled.copy()
        current_y = y_labeled.copy()
        remaining_X = X_unlabeled.copy()

        for iteration in range(max_iterations):
            # Balance datos antes de entrenar
            current_X_balanced, current_y_balanced = self.balance_data(current_X, current_y)

            # Entrenar modelo
            self.train_initial_model(current_X_balanced, current_y_balanced)

            if len(remaining_X) == 0:
                print("No hay más datos no etiquetados para procesar. Deteniendo...")
                break

            # Predecir probabilidades
            probas = self.predict_probabilities(remaining_X)
            max_probas = np.max(probas, axis=1)

            # Seleccionar predicciones confiables
            confident_mask = max_probas >= self.confidence_threshold
            new_X = remaining_X[confident_mask]
            new_y = self.random_forest.predict(new_X) if len(new_X) > 0 else []

            if len(new_X) == 0:
                print(f"Iteración {iteration + 1}: No se encontraron nuevas muestras confiables. Deteniendo...")
                break

            # Actualizar conjuntos
            current_X = pd.concat([current_X, new_X])
            current_y = pd.concat([current_y, pd.Series(new_y)])
            remaining_X = remaining_X[~confident_mask]

            print(f"Iteración {iteration + 1}: {len(new_X)} nuevas muestras añadidas")

        return current_X, current_y
    
    def evaluate_model(self, X_test, y_test):
        y_pred = self.random_forest.predict(X_test)
        print("\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred))
        print("\nMatriz de Confusión:")
        print(confusion_matrix(y_test, y_pred))
        
        # Calcular y mostrar probabilidades de fraude para el conjunto de prueba
        probas = self.predict_probabilities(X_test)
        print("\nDistribución de probabilidades de fraude:")
        fraud_probas = probas[:, 1]  # Probabilidades de la clase 1 (fraude)
        print(f"Media: {np.mean(fraud_probas):.3f}")
        print(f"Mediana: {np.median(fraud_probas):.3f}")
        print(f"Desv. Estándar: {np.std(fraud_probas):.3f}")

def run_fraud_detection(data_path):
    # Cargar datos
    print("Cargando datos...")
    data = pd.read_excel(data_path)
    
    # Crear modelo
    model = SemiSupervisedFraudDetection(confidence_threshold=0.7)  # Reducido el umbral de confianza
    
    # Preprocesar datos
    print("Preprocesando datos...")
    X = model.preprocess_data(data)
    y = data['IsFraud']
    
    # Separar datos etiquetados y no etiquetados
    X_labeled, y_labeled, X_unlabeled = model.split_labeled_unlabeled(X, y)
    
    print("\nDistribución de clases en datos etiquetados:")
    print(y_labeled.value_counts())
    
    # Balance inicial de datos etiquetados
    print("\nBalanceando datos etiquetados...")
    X_labeled_balanced, y_labeled_balanced = model.balance_data(X_labeled, y_labeled)
    
    # Split de training/testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_labeled_balanced, y_labeled_balanced,
        test_size=0.2,
        random_state=42
    )
    
    print("\nIniciando entrenamiento semi-supervisado...")
    final_X, final_y = model.self_training(X_train, y_train, X_unlabeled)
    
    print("\nEntrenando modelo final...")
    model.train_initial_model(final_X, final_y)
    
    print("\nEvaluando modelo...")
    model.evaluate_model(X_test, y_test)
    
    return model

if __name__ == "__main__":
    model = run_fraud_detection('transactions_dataset.xlsx')
