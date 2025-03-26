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
    print(f"Tamaño de la base de datos: {df.shape}")
    print(f"Cantidad de valores nulos: \n{df.isnull().sum()}\n")
    print(f"Número de alumnos por casa: \n{df['House'].value_counts()}")
    return df

def codificacion_caracteristicas(df):
    encoder = preprocessing.OrdinalEncoder(dtype=int)
    df_encoded = encoder.fit_transform(df[['Blood Status', 'House']])
    df['Blood Status'] = df_encoded[:, 0]
    df['House'] = df_encoded[:, 1]
    print(df.head())
    return df

def extract_features_labels(df):
    columnas_caracteristicas = ['Blood Status', 'Bravery', 'Intelligence', 'Loyalty', 'Ambition', 'Dark Arts Knowledge', 'Quidditch Skills', 'Dueling Skills', 'Creativity']
    X = np.asarray(df[columnas_caracteristicas])
    y = np.asarray(df['House'])
    print("Matriz de características:", X[:5])
    print("Matriz de clases:", y[:5])
    return X, y

def split_data(X, y,random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state)
    print(f'Train set: {X_train.shape}, {y_train.shape}')
    print(f'Test set: {X_test.shape}, {y_test.shape}')
    return X_train, X_test, y_train, y_test

def oversampling(X_train, y_train):
    if method == 'oversampling':
        sm = SMOTE(random_state=1)
    else:
        sm = NearMiss()
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    print(f'Balanceado: {X_train_bal.shape}, {y_train_bal.shape}')
    unique, counts = np.unique(y_train_bal, return_counts=True)
    print(dict(zip(unique, counts)))
    return X_train_bal, y_train_bal

def scale_data(X_train, X_test):
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def KNN(X,y):
    for i in range(5):
        X_train, X_test, y_train, y_test = split_data(X,y,i)
        oversampling(X_train, y_train)

def main():
    df = cargar_base()
    df = codificacion_caracteristicas(df)
    X, y = extract_features_labels(df)
    KNN(X,y)
    X_train, y_train = balance_data(X_train, y_train, method='undersampling')
    X_train, X_test = scale_data(X_train, X_test)
    
if __name__=="__main__":
    main()