import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.decomposition import PCA
url=r"D:\Universidad\2º Master\2 Cuatri\AprendizajeAutomatico\GitHubtomasito\Trabajo01\harry_potter_1000_students.csv"
df_orig=pd.read_csv(url, na_values=["?"])
print(df_orig.head())
