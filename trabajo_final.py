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

def split_data(X, y, random_state):
    return train_test_split(X, y, random_state=random_state)

def balance_data(X_train, y_train, method):
    if method == 'oversampling':
        sm = SMOTE(random_state=1)
    elif method == 'undersampling':
        sm = NearMiss()
    else:
        return X_train, y_train
    
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    return X_train_bal, y_train_bal

def scale_data(X_train, X_test):
    scaler = preprocessing.StandardScaler()

    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    scaler.fit(X_test)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

def apply_pca(X_train, X_test, n_components):
    if n_components:
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        return X_train_pca, X_test_pca
    return X_train, X_test

def train_knn(X_train, X_test, y_train, y_test, k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    ini = time.time()
    neigh.fit(X_train, y_train)
    print(f"Tiempo entrenamiento KNN = {(time.time() - ini)*1000:.3f} ms")
    
    ini = time.time()
    y_predict_knn = neigh.predict(X_test)
    print(f"Tiempo de predicción KNN = {(time.time() - ini)*1000:.3f} ms")
    
    print("Exactitud media obtenida con k-NN:", neigh.score(X_test, y_test))
    cm_kNN = confusion_matrix(y_test, y_predict_knn, labels=[0,1,2,3])
    disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_kNN, display_labels=['Gryffindor(0)','Hufflepuff(1)','Ravenclaw(2)','Slytherin(3)'])
    disp_knn.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de confusión k-NN")
    plt.show()

def train_svm(X_train, X_test, y_train, y_test):
    clf_svm = svm.SVC(kernel='linear')
    ini = time.time()
    clf_svm.fit(X_train, y_train)
    print(f"Tiempo entrenamiento SVM = {(time.time() - ini)*1000:.3f} ms")
    
    ini = time.time()
    y_predict_svm = clf_svm.predict(X_test)
    print(f"Tiempo de predicción SVM = {(time.time() - ini)*1000:.3f} ms")
    
    print("Exactitud media obtenida con SVM:", clf_svm.score(X_test, y_test))
    cm_svm = confusion_matrix(y_test, y_predict_svm, labels=[0,1,2,3])
    disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=['Gryffindor(0)','Hufflepuff(1)','Ravenclaw(2)','Slytherin(3)'])
    disp_svm.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de confusión SVM")
    plt.show()

def main():
    df = cargar_base()
    df = codificacion_caracteristicas(df)
    X, y = extract_features_labels(df)
    
    for random_state in range(1, 6):
        X_train, X_test, y_train, y_test = split_data(X, y, random_state)
        for method in [None, 'oversampling', 'undersampling']:
            X_train_bal, y_train_bal = balance_data(X_train, y_train, method)
            X_train_scaled, X_test_scaled = scale_data(X_train_bal, X_test)
            for n_components in [None, 7, 8]:
                X_train_pca, X_test_pca = apply_pca(X_train_scaled, X_test_scaled, n_components)
                for k in [3, 5, 7, 9]:
                    train_knn(X_train_pca, X_test_pca, y_train_bal, y_test, k)
                train_svm(X_train_pca, X_test_pca, y_train_bal, y_test)

if __name__ == "__main__":
    main()
