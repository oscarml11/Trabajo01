import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

url_tomas=r"C:\Users\tomas\Desktop\Google Drive\UDC\MASTER\2º MASTER\2º CUATRI\Introducción al Aprendizaje Automático\GitHub\Trabajo01\harry_potter_1000_students.csv"
url=r"D:\Universidad\2º Master\2 Cuatri\AprendizajeAutomatico\GitHubtomasito\Trabajo01\harry_potter_1000_students.csv"

df_orig=pd.read_csv(url_tomas, na_values=["?"])
#print(df_orig.head())
print(f" Tamaño de la base de datos: {df_orig.shape}")

print(f"Cantidad de valores repetidos: \n {df_orig.isnull().sum()} \n") 

print(f"Número de alumnos por casa: \n {df_orig['House'].value_counts()}")

encoder = preprocessing.OrdinalEncoder(dtype=int)
df_encoded = encoder.fit_transform(df_orig[['Blood Status', 'House']])
df_orig["Blood Status"] = df_encoded[:,0]
df_orig["House"] = df_encoded[:,1]
print(df_orig.head())

feature_df = df_orig[['Blood Status', 'Bravery', 'Intelligence', 'Loyalty', 'Ambition', 'Dark Arts Knowledge', 'Quidditch Skills','Dueling Skills', 'Creativity']]
X_readed = np.asarray(feature_df)
print(X_readed[0:5])

y_readed = np.asarray(df_orig['House'])
y_readed[0:5]
print(y_readed[0:5])

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

# scaler = preprocessing.StandardScaler()
# scaler.fit(X_train) # fit realiza los cálculos y los almacena

# X_train = scaler.transform(X_train) # aplica los cálculos sobre el conjunto de datos de entrada para escalarlos
# print(X_train[0:5])

mypca = PCA()
mypca.fit(X_train)
# print(mypca.explained_variance_ratio_)

# print(mypca.explained_variance_ratio_.sum())  # Suma todos los elementos de la lista

print("\n Varianza que aporta cada componente:")
variance = mypca.explained_variance_ratio_
print(variance)

print("\n Varianza acumulada:")
acumvar = variance.cumsum()

for i in range(len(acumvar)):
    print(f" {(i+1):2} componentes: {acumvar[i]:.8f} ")

# mypca2 = PCA(n_components=2)
# mypca2.fit(X_train)
# values_proj2 = mypca2.transform(X_train)

# X_projected2 = mypca2.inverse_transform(values_proj2)
# loss2 = ((X_train - X_projected2) ** 2).mean()
# print("Projection loss (2 components): " + str(loss2))

# plt.figure()
# plt.subplot(1,2,1) # 1 - numrows, 2 - numcols, 1 - index
# plt.title("Datos originales con dos atributos")
# plt.scatter(X_train[: ,0] , X_train[: ,1] ,marker='o' ,c=y_train)
# plt.subplot(1,2,2) # 1 - numrows, 2 - numcols, 2 - index
# plt.scatter(values_proj2[: ,0] , values_proj2[: ,1],marker='o' ,c=y_train)
# plt.title("Proyección PCA con 2 componentes")
# plt.subplots_adjust(right=1.9) # Distancia a la derecha
# plt.show()