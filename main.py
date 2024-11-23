import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

class SemiSupervisedFraudDetection:
    def __init__(self, confidence_threshold=0.8, n_clusters=4):
        self.confidence_threshold = confidence_threshold
        self.n_clusters = n_clusters
        self.random_forest = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight='balanced',
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=42,
            n_init=10
        )
        
        # Crear directorio para visualizaciones
        self.visualization_dir = self._create_visualization_directory()
        
    def _create_visualization_directory(self):
        """Crea un directorio para almacenar las visualizaciones"""
        base_dir = 'fraud_detection_visualizations'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_path = f'{base_dir}_{timestamp}'
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def save_figure(self, name):
        """Guarda la figura actual con un nombre específico"""
        plt.savefig(os.path.join(self.visualization_dir, f'{name}.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()

    def preprocess_data(self, data):
        df = data.copy()
        
        # Manejar valores faltantes
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        
        # Extraer características temporales
        df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour
        df['DayOfWeek'] = pd.to_datetime(df['Timestamp']).dt.dayofweek
        df['DayOfMonth'] = pd.to_datetime(df['Timestamp']).dt.day
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        df['IsNightTime'] = df['Hour'].isin([22, 23, 0, 1, 2, 3]).astype(int)
        
        # Crear características para transacciones repetidas
        transaction_counts = df.groupby(['City', 'Hour'])['TransactionID'].transform('count')
        df['TransactionsInCityHour'] = transaction_counts
        
        # Codificar variables categóricas
        categorical_columns = ['City', 'District', 'PaymentMethod']
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                df[column] = self.label_encoders[column].fit_transform(df[column])
            else:
                df[column] = self.label_encoders[column].transform(df[column])
        
        features = [
            'Amount', 'City', 'District', 'PaymentMethod',
            'Hour', 'DayOfWeek', 'DayOfMonth', 'IsWeekend',
            'IsNightTime', 'TransactionsInCityHour'
        ]
        
        X = df[features]
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=features), df

    def visualize_clusters(self, X, clusters, y=None):
        """Visualiza los clusters usando t-SNE y guarda las visualizaciones"""
        try:
            X_clean = np.nan_to_num(X)
            
            # Reducir dimensionalidad con t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            X_tsne = tsne.fit_transform(X_clean)
            
            # Visualización de clusters
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='viridis')
            plt.title('Visualización de Clusters usando t-SNE')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.colorbar(scatter)
            self.save_figure('clusters_tsne')
            
            # Visualización de etiquetas reales si están disponibles
            if y is not None and not y.isna().all():
                plt.figure(figsize=(12, 8))
                y_clean = y.fillna(-1)
                scatter2 = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_clean, cmap='Set1')
                plt.title('Visualización de Etiquetas Reales usando t-SNE')
                plt.xlabel('t-SNE 1')
                plt.ylabel('t-SNE 2')
                plt.colorbar(scatter2)
                self.save_figure('real_labels_tsne')
                
        except Exception as e:
            print(f"Error en la visualización: {str(e)}")

    def analyze_clusters(self, X, clusters, original_data):
        """Analiza las características de cada cluster y genera visualizaciones relevantes"""
        try:
            cluster_data = original_data.copy()
            cluster_data['Cluster'] = clusters
            
            # Análisis por cluster
            cluster_stats = []
            for i in range(self.n_clusters):
                cluster_i = cluster_data[cluster_data['Cluster'] == i]
                stats = {
                    'Cluster': i,
                    'Tamaño': len(cluster_i),
                    'Amount_Media': cluster_i['Amount'].mean(),
                    'Amount_Std': cluster_i['Amount'].std(),
                    'Hora_Media': cluster_i['Hour'].mean(),
                    '%_Noche': (cluster_i['IsNightTime'] == 1).mean() * 100,
                    '%_FinesSemana': (cluster_i['IsWeekend'] == 1).mean() * 100
                }
                cluster_stats.append(stats)
            
            cluster_summary = pd.DataFrame(cluster_stats)
            
            # Generar y guardar visualizaciones clave
            self.plot_cluster_characteristics(cluster_data)
            
            # Guardar resumen de clusters
            cluster_summary.to_csv(os.path.join(self.visualization_dir, 'cluster_summary.csv'))
            
            return cluster_summary
            
        except Exception as e:
            print(f"Error en el análisis de clusters: {str(e)}")
            return pd.DataFrame()

    def plot_cluster_characteristics(self, cluster_data):
        """Genera y guarda visualizaciones clave de las características de los clusters"""
        try:
            # 1. Distribución de montos por cluster (importante para detectar anomalías)
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='Cluster', y='Amount', data=cluster_data)
            plt.title('Distribución de Montos por Cluster')
            self.save_figure('amount_distribution_by_cluster')
            
            # 2. Heatmap de actividad temporal (crucial para patrones de fraude)
            plt.figure(figsize=(15, 8))
            pivot_table = pd.pivot_table(
                cluster_data, 
                values='TransactionID',
                index='Hour',
                columns=['DayOfWeek', 'Cluster'],
                aggfunc='count',
                fill_value=0
            )
            sns.heatmap(pivot_table, cmap='YlOrRd')
            plt.title('Patrones Temporales de Transacciones por Cluster')
            self.save_figure('temporal_patterns_heatmap')
            
            # 3. Análisis de métodos de pago por cluster
            plt.figure(figsize=(12, 6))
            payment_cluster = pd.crosstab(
                cluster_data['Cluster'],
                cluster_data['PaymentMethod']
            )
            payment_cluster.plot(kind='bar', stacked=True)
            plt.title('Distribución de Métodos de Pago por Cluster')
            plt.legend(title='Método de Pago')
            self.save_figure('payment_methods_by_cluster')
            
        except Exception as e:
            print(f"Error en las visualizaciones de características: {str(e)}")

def run_fraud_detection(data_path):
    try:
        print("Cargando datos...")
        data = pd.read_excel(data_path)
        
        model = SemiSupervisedFraudDetection(confidence_threshold=0.75)
        
        print("Preprocesando datos...")
        X, original_data = model.preprocess_data(data)
        y = data['IsFraud']
        
        print("\nRealizando clustering...")
        clusters = model.kmeans.fit_predict(X)
        
        print("\nGenerando visualizaciones...")
        model.visualize_clusters(X, clusters, y)
        
        print("\nAnalizando características de los clusters...")
        cluster_summary = model.analyze_clusters(X, clusters, original_data)
        print("\nResumen de clusters guardado en:", model.visualization_dir)
        print(cluster_summary)
        
        return model, cluster_summary
        
    except Exception as e:
        print(f"Error en el proceso principal: {str(e)}")
        return None, None

if __name__ == "__main__":
    model, cluster_summary = run_fraud_detection('transactions_dataset.xlsx')