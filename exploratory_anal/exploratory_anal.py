import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os

from .distribution.distribution import distributions_menu
from .correlation.correlation import correlations_menu


def show_basic_info(df):
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clears terminal
        print("\n=== Análise Exploratória ===")
        print("1 - Informações sobre as colunas")
        print("2 - Descrição dos dados")
        print("3 - Voltar")
        
        choice = input("Escolha uma opção: ")
        
        if choice == '1':
            os.system('cls' if os.name == 'nt' else 'clear')  # Clears terminal
            print(df.info())
            input("\n\nPressione Enter para continuar...")
        elif choice == '2':
            os.system('cls' if os.name == 'nt' else 'clear')  # Clears terminal
            print(df.describe(include='all'))
            input("\n\nPressione Enter para continuar...")
        elif choice == '3':
            break
        else:
            print("Opção inválida. Tente novamente.")

    print("\n=== Primeiras linhas do dataset ===")
    print(df.head())
    
    print("\n=== Informações do dataset ===")
    print(df.info())
    
    print("\n=== Estatísticas descritivas ===")
    print(df.describe(include='all'))

def analyze_outliers(df):
    """Identificação e análise de outliers"""
    num_cols = ['Idade', 'PressaoArterialRepouso', 'Colesterol', 'FrequenciaCardiacaMaxima', 'DepressaoSegST']
    tex_cols = ['Sexo', 'TipoDorToracica', 'ECGRepouso', 'DorAngina', 'InfradesnivelamentoSegST']

    # cat_cols = [col for col in df.columns if col not in num_cols and df[col].dtype == 'object']
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clears terminal
        print("\n=== Análise de Outliers e Valores Únicos ===")
        print("\nEscolha uma opção:")
        print("1 - Colunas numéricas (boxplot)")
        print("2 - Colunas textuais (valores únicos)")
        print("3 - Voltar")
        
        opcao = input("Escolha uma opção: ")
        
        if opcao == '1':
            # Boxplots para visualização de colunas numéricas
            plt.figure(figsize=(12, 6))
            df[num_cols].boxplot()
            plt.xticks(rotation=0)
            plt.title('Boxplots para Identificação de Outliers')
            plt.show()
            
        elif opcao == '2':
            # Análise de colunas textuais/categóricas
            # if not cat_cols:
            #     print("\nNão há colunas textuais no dataframe.")
            #     return
                
            print("\nValores únicos em colunas textuais:")
            for col in tex_cols:
                unique_vals = df[col].unique()
                print(f"\nColuna '{col}':")
                print(unique_vals)
            
        elif opcao == '3':
            break
                
        else:
            print("Opção inválida. Por favor escolha 1 ou 2.")
      
def exploratory_analysis_menu(df):
    """Menu para análise exploratória"""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clears terminal
        print("\n=== Análise Exploratória ===")
        print("1 - Correlações")
        print("2 - Outliers")
        print("3 - Distribuição das variáveis")
        print("4 - Visualizar estrutura geral dos dados")
        print("5 - Voltar")
        
        choice = input("Escolha uma opção: ")
        
        if choice == '1':
            correlations_menu(df)
        elif choice == '2':
            analyze_outliers(df)
        elif choice == '3':
            distributions_menu(df)
        elif choice == '4':
            show_basic_info(df)
        elif choice == '5':
            break
        else:
            print("Opção inválida. Tente novamente.")
