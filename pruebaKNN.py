import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

def cargar_base():
    direccion = r"C:\Users\tomas\Desktop\Google Drive\UDC\MASTER\2º MASTER\2º CUATRI\Introducción al Aprendizaje Automático\GitHub\Trabajo01\harry_potter_1000_students.csv"
    df = pd.read_csv(direccion, na_values=["?"])
    return df

def codificacion_caracteristicas(df):
    encoder = preprocessing.OrdinalEncoder(dtype=int)
    df[['Blood Status', 'House']] = encoder.fit_transform(df[['Blood Status', 'House']])
    return df

def extract_features_labels(df):
    columnas_caracteristicas = ['Blood Status', 'Bravery', 'Intelligence', 'Loyalty', 'Ambition', 'Dark Arts Knowledge', 'Quidditch Skills', 'Dueling Skills', 'Creativity']
    return np.asarray(df[columnas_caracteristicas]), np.asarray(df['House'])

def balance_data(X_train, y_train, method):
    if method == 'oversampling':
        return SMOTE(random_state=1).fit_resample(X_train, y_train)
    elif method == 'undersampling':
        return NearMiss().fit_resample(X_train, y_train)
    return X_train, y_train

def scale_data(X_train, X_test):
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

def apply_pca(X_train, X_test, n_components):
    if n_components:
        pca = PCA(n_components=n_components)
        return pca.fit_transform(X_train), pca.transform(X_test)
    return X_train, X_test

def train_knn(X_train, X_test, y_train, y_test, k, results):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    y_predict_knn = neigh.predict(X_test)
    accuracy = np.mean(y_predict_knn == y_test)
    results.append((k, accuracy))

def main():
    df = cargar_base()
    df = codificacion_caracteristicas(df)
    X, y = extract_features_labels(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    
    for method in [None, 'oversampling', 'undersampling']:
        X_train_bal, y_train_bal = balance_data(X_train, y_train, method)
        X_train_scaled, X_test_scaled = scale_data(X_train_bal, X_test)
        results = []
        
        for k in range(1, 101):
            train_knn(X_train_scaled, X_test_scaled, y_train_bal, y_test, k, results)
        
        results_df = pd.DataFrame(results, columns=['K', 'Accuracy'])
        results_df.to_csv(f"results_{method}.csv", index=False)
        print(f"Resultados guardados para {method if method else 'sin balanceo'}")

if __name__ == "__main__":
    main()
