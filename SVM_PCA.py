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

url_tomas=r"C:\Users\tomas\Desktop\Google Drive\UDC\MASTER\2º MASTER\2º CUATRI\Introducción al Aprendizaje Automático\GitHub\Trabajo01\harry_potter_1000_students.csv"
url=r"D:\Universidad\2º Master\2 Cuatri\AprendizajeAutomatico\GitHubtomasito\Trabajo01\harry_potter_1000_students.csv"

df_orig=pd.read_csv(url_tomas, na_values=["?"])
#print(df_orig.head())
print(f" Tamaño de la base de datos: {df_orig.shape}")

print(f"Cantidad de valores repetidos: \n {df_orig.isnull().sum()} \n") 

print(f"Número de alumnos por casa: \n {df_orig['House'].value_counts()}")


#CATEGORIZAR CARACTERISTICAS
encoder = preprocessing.OrdinalEncoder(dtype=int)
df_encoded = encoder.fit_transform(df_orig[['Blood Status', 'House']])
df_orig["Blood Status"] = df_encoded[:,0]
df_orig["House"] = df_encoded[:,1]
print(df_orig.head())

# MATRIZ CARACATERISTICAS
feature_df = df_orig[['Blood Status', 'Bravery', 'Intelligence', 'Loyalty', 'Ambition', 'Dark Arts Knowledge', 'Quidditch Skills','Dueling Skills', 'Creativity']]
X_readed = np.asarray(feature_df)
print(X_readed[0:5])

#MATRIZ CLASES
y_readed = np.asarray(df_orig['House'])
y_readed[0:5]
print(y_readed[0:5])

#SEPARACION ENTRENAMIENTO Y TESTEO
X_train, X_test, y_train, y_test = train_test_split(X_readed, y_readed, random_state = 1)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

print ('Conjunto entrenamiento original', X_train.shape,  y_train.shape)
unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))

# Balanceo de datos: ejemplo de oversampling
# sm = SMOTE(random_state=1)
# X_train, y_train= sm.fit_resample(X_train, y_train)

# Balanceo de datos: ejemplo de undersampling
sm = NearMiss()
X_train, y_train= sm.fit_resample(X_train, y_train)

print('\nBalanceado:', X_train.shape,  y_train.shape)
unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))

#ESCALADO DE DATOS
scaler = preprocessing.StandardScaler()
scaler.fit(X_train) # fit realiza los cálculos y los almacena
X_train = scaler.transform(X_train) # aplica los cálculos sobre el conjunto de datos de entrada para escalarlos

scaler.fit(X_test)
X_test = scaler.transform(X_test)

mypca = PCA()
mypca.fit(X_train)

#PESOS DE LAS CARACTERISTICAS
print("\n Varianza que aporta cada componente:")
variance = mypca.explained_variance_ratio_
print(variance)

print("\n Varianza acumulada:")
acumvar = variance.cumsum()

for i in range(len(acumvar)):
    print(f" {(i+1):2} componentes: {acumvar[i]:.8f} ")

mypca2 = PCA(n_components=7)
mypca2.fit(X_train)
mypca2.fit(X_test)
X_train_PCA = mypca2.transform(X_train)
X_test_PCA = mypca2.transform(X_test)

X_train_back = mypca2.inverse_transform(X_train_PCA)
loss2 = ((X_train - X_train_back) ** 2).mean()
print("Projection los (2 components): " + str(loss2))


#ENTRENAMIENTO SVM
# Creamos nuestra instancia del modelo
clf_svm = svm.SVC(kernel='linear')

ini = time.time() 
#Entrenamiento del modelo, llamando a su método fit 
clf_svm.fit(X_train_PCA, y_train) 
print(f"Tiempo entrenamiento = {(time.time() - ini)*1000:.3f} ms") 

#PREDICCION
ini = time.time() 
y_predict_svm = clf_svm.predict(X_test_PCA)
print(f"Tiempo de predicción = {(time.time() - ini)*1000:.3f} ms") 

y_predict_svm [0:5] # muestra del resultado de la predicción

print("Exactitud media obtenida con SVM: ", clf_svm.score(X_test_PCA, y_test))

#MATRIZ DE CONFUSIÓN
cm_svm = confusion_matrix(y_test, y_predict_svm, labels=[0,1,2,3])

disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=['Gryffindor(0)','Hufflepuff(1)','Ravenclaw(2)','Slytherin(3)'])
disp_svm.plot(cmap=plt.cm.Blues)
plt.title("Matriz de confusión SVM")
plt.show()