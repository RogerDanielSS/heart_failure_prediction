import pandas as pd
import numpy as np
from exploratory_anal import exploratory_analysis_menu
from training import training_menu
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def default_preprocess(df):
    """Pré-processamento dos dados para o modelo, mantendo no DataFrame
    - Normaliza todas as variáveis numéricas
    - Transforma variáveis categóricas em índices numéricos
    """
    # Criar uma cópia para não modificar o original
    df_processed = df.copy()
    
    # Definir colunas numéricas e categóricas
    numeric_features = ['Idade', 'PressaoArterialRepouso', 'Colesterol', 
                       'FrequenciaCardiacaMaxima', 'DepressaoSegST']
    categorical_features = ['Sexo', 'TipoDorToracica', 'GlicemiaJejum', 
                           'ECGRepouso', 'DorAngina', 'InfradesnivelamentoSegST']
    
    # 1. Normalização das variáveis numéricas
    if numeric_features:
        scaler = StandardScaler()
        df_processed[numeric_features] = scaler.fit_transform(df_processed[numeric_features])
    
    # 2. Transformação das variáveis categóricas em índices numéricos
    for col in categorical_features:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
    
    return df_processed

def preprocessing_menu(df):
    """Menu para análise exploratória"""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clears terminal
        print("\n=== Pré processamento ===")
        print("1 - Padrão")
        
        choice = input("Escolha uma opção: ")
        
        if choice == '1':
            df_processed = default_preprocess(df)
            redirect_to_training_menu(df_processed)
        # elif choice == '2':
        #     analyze_outliers(df)
        # elif choice == '3':
        #     analyze_distributions(df)
        # elif choice == '4':
            break
        else:
            print("Opção inválida. Tente novamente.")

def redirect_to_training_menu(df):
    """Menu para análise exploratória"""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clears terminal
        print("\n=== Pré processamento ===")
        print("1 - Treinamento")
        print("2 - Análise exploratória do dataset processado")
        print("3 - Voltar")
        
        choice = input("Escolha uma opção: ")
        
        if choice == '1':
            training_menu(df)
        elif choice == '2':
            exploratory_analysis_menu(df)
        elif choice == '3':
            break
        else:
            print("Opção inválida. Tente novamente.")
