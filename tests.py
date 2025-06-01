import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Carregar os dados
df = pd.read_csv('./dataset/heart.csv')

# Visualizar as primeiras linhas
print(df.head())

# # Informações gerais sobre o dataset
# print(df.info())

# # Estatísticas descritivas
# print(df.describe(include='all'))

# # Histogramas para variáveis numéricas
# num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
# df[num_cols].hist(figsize=(12, 8))
# plt.tight_layout()
# plt.show()

# # Boxplots para identificar outliers
# plt.figure(figsize=(12, 6))
# df[num_cols].boxplot()
# plt.xticks(rotation=45)
# plt.show()

# # Contagem para variáveis categóricas
# cat_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'HeartDisease']
# for col in cat_cols:
#     plt.figure(figsize=(8, 4))
#     sns.countplot(data=df, x=col)
#     plt.title(f'Distribuição de {col}')
#     plt.show()

#     # Verificar valores ausentes
# print(df.isnull().sum())

# # Visualização gráfica
# plt.figure(figsize=(10, 6))
# sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
# plt.title('Mapa de Valores Ausentes')
# plt.show()