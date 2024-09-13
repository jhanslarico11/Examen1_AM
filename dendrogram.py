import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Datos proporcionados en la tabla
data = {
    'sexo': [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
    'edad': [9, 10, 9, 9, 9, 10, 10, 10, 9, 9, 10, 9, 9, 9, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10],
    'divertid': [6, 2, 7, 4, 1, 6, 5, 7, 2, 3, 1, 5, 2, 4, 6, 3, 4, 3, 3, 4, 2, 5, 4, 3, 1],
    'pidocomp': [7, 1, 6, 4, 2, 6, 6, 7, 3, 3, 2, 5, 2, 4, 4, 4, 7, 2, 4, 3, 2, 7, 2, 5, 2],
    'aprendom': [3, 4, 3, 4, 2, 4, 4, 4, 3, 3, 3, 4, 4, 4, 4, 4, 5, 2, 7, 4, 2, 5, 2, 4, 2],
    'excur': [3, 4, 3, 6, 2, 4, 3, 4, 3, 3, 3, 4, 4, 7, 4, 4, 5, 6, 7, 4, 2, 5, 7, 5, 3],
    'quitatie': [4, 4, 3, 5, 2, 4, 3, 4, 4, 6, 5, 4, 5, 6, 2, 5, 4, 7, 6, 7, 3, 4, 7, 4, 2],
    'nomeint': [2, 3, 1, 3, 6, 3, 3, 3, 4, 5, 5, 2, 4, 6, 5, 4, 2, 4, 4, 6, 2, 4, 7, 4, 4],
    'gustovis': [1, 0, 1, 1, 0, 1, 1, 1, 0, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 3, 1, 0, 0]
}

# Convertir a DataFrame
df = pd.DataFrame(data)

# Escalado de las variables (normalización)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Cálculo del linkage matrix utilizando el método de Ward
Z = linkage(df_scaled, method='ward')

# Crear el dendrograma
plt.figure(figsize=(10, 7))
dendrogram(Z, labels=df.index, orientation='top', distance_sort='descending')
plt.title('Dendrograma con Enlace de Ward')
plt.xlabel('Índice')
plt.ylabel('Distancia')
plt.show()
