import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.decomposition import PCA


url_tomas=r"C:\Users\tomas\Desktop\Google Drive\UDC\MASTER\2º MASTER\2º CUATRI\Introducción al Aprendizaje Automático\GitHub\Trabajo01\harry_potter_1000_students.csv"
url=r"D:\Universidad\2º Master\2 Cuatri\AprendizajeAutomatico\GitHubtomasito\Trabajo01\harry_potter_1000_students.csv"

df_orig=pd.read_csv(url_tomas, na_values=["?"])
print(df_orig.head())
print(df_orig.shape)

print(df_orig.isnull().sum())

print(f"Número de alumnos por casa: \n {df_orig['House'].value_counts()}")

encoder = preprocessing.OrdinalEncoder(dtype=int)
df_encoded = encoder.fit_transform(df_orig[['Blood Status', 'House']])
encoder.categories_
df_orig["Blood Status"] = df_encoded[:,0]
df_orig["House"] = df_encoded[:,1]
print(df_orig.head())
