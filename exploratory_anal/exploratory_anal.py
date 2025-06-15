import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os

from .distribution.distribution import distributions_menu
from .correlation.correlation import correlations_menu
from .outliers.outliers import outliers_menu
from .basic_info.basic_info import basic_info_menu

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
            outliers_menu(df)
        elif choice == '3':
            distributions_menu(df)
        elif choice == '4':
            basic_info_menu(df)
        elif choice == '5':
            break
        else:
            print("Opção inválida. Tente novamente.")
