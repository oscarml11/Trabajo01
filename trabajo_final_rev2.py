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
    direccion = r"D:\Universidad\2º Master\2 Cuatri\AprendizajeAutomatico\GitHubtomasito\Trabajo01\harry_potter_1000_students.csv"
    df = pd.read_csv(direccion, na_values=["?"])
    return df

def codificacion_caracteristicas(df):
    encoder = preprocessing.OrdinalEncoder(dtype=int)
    df_encoded = encoder.fit_transform(df[['Blood Status', 'House']])
    df['Blood Status'] = df_encoded[:, 0]
    df['House'] = df_encoded[:, 1]
    return df

def extract_features_labels(df):
    columnas_caracteristicas = ['Blood Status', 'Bravery', 'Intelligence', 'Loyalty', 'Ambition', 'Dark Arts Knowledge', 'Quidditch Skills', 'Dueling Skills', 'Creativity']
    X = np.asarray(df[columnas_caracteristicas])
    y = np.asarray(df['House'])
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
    
    return sm.fit_resample(X_train, y_train)

def scale_data(X_train, X_test):
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def apply_pca(X_train, X_test, n_components, varianza):
    if n_components is None and varianza != 1:
        pca = PCA()
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Cálculo de varianza explicada
        print("\nVarianza que aporta cada componente:")
        variance = pca.explained_variance_ratio_
        print(variance)

        print("\nVarianza acumulada:")
        acumvar = variance.cumsum()

        for i in range(len(acumvar)):
            print(f" {(i+1):2} componentes: {acumvar[i]:.8f}")
        
        varianza = 1
        return X_train_pca, X_test_pca, varianza
    
    else:
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        return X_train_pca, X_test_pca, varianza

def evaluate_model(y_test, y_pred, classifier, random_state, method, n_components, k, train_time, predict_time, results):
    accuracy = np.mean(y_pred == y_test)
    num_errors = np.sum(y_pred != y_test)
    num_correct = np.sum(y_pred == y_test)
    results.append({
        "Random State": random_state,
        "Balancing Method": method if method else "None",
        "PCA Components": n_components if n_components else "None",
        "Classifier": classifier,
        "K (if KNN) / Kernel (if SVM)": k if classifier == "KNN" else "Linear",
        "Accuracy": accuracy,
        "Errors": num_errors,
        "Correct Predictions": num_correct,
        "Training Time (ms)": train_time * 1000,
        "Prediction Time (ms)": predict_time * 1000
    })

def plot_confusion_matrix(cm, title, random_state, method, n_components, classifier, k):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Gryffindor(0)','Hufflepuff(1)','Ravenclaw(2)','Slytherin(3)'])
    disp.plot(cmap=plt.cm.Blues)
    if k == "Linear":
        plt.title(f"{title}\nRandom State: {random_state}, Balanceo: {method}, PCA: {n_components}, Técnica: {classifier}, Kernel: {k}")
        # plt.show(block=False)
    else:
        plt.title(f"{title}\nRandom State: {random_state}, Balanceo: {method}, PCA: {n_components}, Técnica: {classifier}, K: {k}")
        # plt.show(block=False)
    filename = f"confusion_matrix_{classifier}_RS{random_state}_Bal{method}_PCA{n_components}_K{k}.png"
    plt.savefig(filename)
    plt.close()

def train_knn(X_train, X_test, y_train, y_test, k, random_state, method, n_components, results):
    neigh = KNeighborsClassifier(n_neighbors=k)
    ini = time.time()
    neigh.fit(X_train, y_train)
    train_time = time.time() - ini
    
    ini = time.time()
    y_predict_knn = neigh.predict(X_test)
    predict_time = time.time() - ini
    
    evaluate_model(y_test, y_predict_knn, "KNN", random_state, method, n_components, k, train_time, predict_time, results)
    
    cm_knn = confusion_matrix(y_test, y_predict_knn, labels=[0,1,2,3])
    plot_confusion_matrix(cm_knn, "Matriz de confusión k-NN", random_state, method, n_components, "KNN", k)

def train_svm(X_train, X_test, y_train, y_test, random_state, method, n_components, results):
    clf_svm = svm.SVC(kernel='linear')
    ini = time.time()
    clf_svm.fit(X_train, y_train)
    train_time = time.time() - ini
    
    ini = time.time()
    y_predict_svm = clf_svm.predict(X_test)
    predict_time = time.time() - ini
    
    evaluate_model(y_test, y_predict_svm, "SVM", random_state, method, n_components, "Linear", train_time, predict_time, results)
    
    cm_svm = confusion_matrix(y_test, y_predict_svm, labels=[0,1,2,3])
    plot_confusion_matrix(cm_svm, "Matriz de confusión SVM", random_state, method, n_components, "SVM", "Linear")

def main():
    df = cargar_base()
    df = codificacion_caracteristicas(df)
    X, y = extract_features_labels(df)
    results = []
    varianza =0
    
    for random_state in range(1, 6):
        X_train, X_test, y_train, y_test = split_data(X, y, random_state)
        for method in [None, 'oversampling', 'undersampling']:
            X_train_bal, y_train_bal = balance_data(X_train, y_train, method)
            X_train_scaled, X_test_scaled = scale_data(X_train_bal, X_test)
            for n_components in [None, 7, 8]:
                X_train_pca, X_test_pca, varianza = apply_pca(X_train_scaled, X_test_scaled, n_components, varianza)
                for k in [3, 5, 7, 9]:
                    train_knn(X_train_pca, X_test_pca, y_train_bal, y_test, k, random_state, method, n_components, results)
                train_svm(X_train_pca, X_test_pca, y_train_bal, y_test, random_state, method, n_components, results)
    
    df_results = pd.DataFrame(results)
    df_results.to_csv("model_results.csv", index=False, float_format="%.3f")
    print("Resultados guardados en 'model_results.csv'")
    

if __name__ == "__main__":
    main()
