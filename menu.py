import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

def load_data(filepath):
    """Carrega o dataset"""
    return pd.read_csv(filepath)

def translate_dataset(df):
    """Traduz colunas e valores categóricos para português"""

    # Dicionário de tradução das colunas
    traducoes_colunas = {
      'Age': 'Idade',
      'Sex': 'Sexo',
      'ChestPainType': 'TipoDorToracica',
      'RestingBP': 'PressaoArterialRepouso',
      'Cholesterol': 'Colesterol',
      'FastingBS': 'GlicemiaJejum',
      'RestingECG': 'ECGRepouso',
      'MaxHR': 'FrequenciaCardiacaMaxima',
      'ExerciseAngina': 'DorAngina',
      'Oldpeak': 'DepressaoSegST',
      'ST_Slope': 'InfradesnivelamentoSegST',
      'HeartDisease': 'DoencaCardiaca'
    }
    
    # Renomear colunas
    df = df.rename(columns=traducoes_colunas)
    
    # Dicionários para traduzir valores categóricos
    traducoes_valores = {
        'Sexo': {'M': 'Masculino', 'F': 'Feminino'},
        'TipoDorToracica': {
            'ATA': 'Angina Típica',
            'NAP': 'Angina Atípica',
            'ASY': 'Assintomático',
            'TA': 'Angina Típica'  # Caso exista esta categoria
        },
        'ECGRepouso': {
            'Normal': 'Normal',
            'ST': 'Anormalidade onda ST-T',
            'LVH': 'Hipertrofia Ventricular Esquerda'
        },
        'AnginaExercicio': {'Y': 'Sim', 'N': 'Não'},
        'InclinacaoST': {
            'Up': 'Ascendente',
            'Flat': 'Plano',
            'Down': 'Descendente'
        }
    }
    
    # Aplicar tradução aos valores categóricos
    for col, traducoes in traducoes_valores.items():
        if col in df.columns:
            df[col] = df[col].replace(traducoes)
    
    return df

def show_basic_info(df):
    """Mostra informações básicas do dataset"""
    print("\n=== Primeiras linhas do dataset ===")
    print(df.head())
    
    print("\n=== Informações do dataset ===")
    print(df.info())
    
    print("\n=== Estatísticas descritivas ===")
    print(df.describe(include='all'))

def analyze_patterns(df):
    """Identificação de padrões nos dados"""
    print("\n=== Identificação de Padrões ===")
    
    # Padrões em variáveis numéricas
    num_cols = ['Idade', 'PressaoArterialRepouso', 'Colesterol', 'FrequenciaCardiacaMaxima', 'DepressaoSegST']
    print("\nMédias por status de doença cardíaca:")
    print(df.groupby('DoencaCardiaca')[num_cols].mean())
    
    # Padrões em variáveis categóricas
    cat_cols = ['Sexo', 'TipoDorToracica', 'GlicemiaJejum', 'ECGRepouso', 'DorAngina', 'InfradesnivelamentoSegST']
    for col in cat_cols:
        print(f"\nDistribuição de {col} por status de doença cardíaca:")
        print(pd.crosstab(df[col], df['DoencaCardiaca'], normalize='index') * 100)
    
    # Visualização de alguns padrões
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='DoencaCardiaca', y='Idade')
    plt.title('Distribuição de Idade por Status de Doença Cardíaca')
    plt.show()

def analyze_correlations(df):
    """Análise de correlações entre variáveis"""
    print("\n=== Análise de Correlações ===")
    
    # Matriz de correlação
    plt.figure(figsize=(12, 8))
    corr_matrix = df.corr(numeric_only=True)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlação')
    plt.show()
    
    # Correlação com a variável target
    print("\nCorrelação com DoencaCardiaca:")
    print(corr_matrix['DoencaCardiaca'].sort_values(ascending=False))
    
    # Scatter plots para relações interessantes
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Idade', y='FrequenciaCardiacaMaxima', hue='DoencaCardiaca', alpha=0.7)
    plt.title('Relação entre Idade e Frequência Cardíaca Máxima')
    plt.show()

def analyze_outliers(df):
    """Identificação e análise de outliers"""
    print("\n=== Análise de Outliers ===")
    
    num_cols = ['Idade', 'PressaoArterialRepouso', 'Colesterol', 'FrequenciaCardiacaMaxima', 'DepressaoSegST']
    
    # Boxplots para visualização
    plt.figure(figsize=(12, 6))
    df[num_cols].boxplot()
    plt.xticks(rotation=45)
    plt.title('Boxplots para Identificação de Outliers')
    plt.show()
    
    # Identificação usando Z-score
    z_scores = np.abs(stats.zscore(df[num_cols]))
    outliers = (z_scores > 3).any(axis=1)
    print(f"\nNúmero de outliers (Z-score > 3): {outliers.sum()}")
    print("\nRegistros considerados outliers:")
    print(df[outliers])

def analyze_distributions(df):
    """Análise das distribuições das variáveis"""
    print("\n=== Análise de Distribuições ===")
    
    # Histogramas para variáveis numéricas
    num_cols = ['Idade', 'PressaoArterialRepouso', 'Colesterol', 'FrequenciaCardiacaMaxima', 'DepressaoSegST']
    df[num_cols].hist(figsize=(12, 8))
    plt.tight_layout()
    plt.suptitle('Distribuição de Variáveis Numéricas', y=1.02)
    plt.show()
    
    # Gráficos de barras para variáveis categóricas
    cat_cols = ['Sexo', 'TipoDorToracica', 'GlicemiaJejum', 'ECGRepouso', 'DorAngina', 'InfradesnivelamentoSegST']
    for col in cat_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col)
        plt.title(f'Distribuição de {col}')
        plt.show()

def exploratory_analysis_menu(df):
    """Menu para análise exploratória"""
    while True:
        print("\n=== Análise Exploratória ===")
        print("1 - Identificação de padrões")
        print("2 - Correlações")
        print("3 - Outliers")
        print("4 - Distribuição das variáveis")
        print("5 - Voltar ao menu principal")
        
        choice = input("Escolha uma opção: ")
        
        if choice == '1':
            analyze_patterns(df)
        elif choice == '2':
            analyze_correlations(df)
        elif choice == '3':
            analyze_outliers(df)
        elif choice == '4':
            analyze_distributions(df)
        elif choice == '5':
            break
        else:
            print("Opção inválida. Tente novamente.")

def main():
    """Função principal"""
    filepath = './dataset/heart.csv'
    df = load_data(filepath)
    df =  translate_dataset(df)
    
    while True:
        print("\n=== Menu Principal ===")
        print("1 - Análise exploratória")
        print("2 - Sair")
        
        choice = input("Escolha uma opção: ")
        
        if choice == '1':
            exploratory_analysis_menu(df)
        elif choice == '2':
            print("Encerrando o programa...")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()