import pandas as pd
import numpy as np
from exploratory_anal import exploratory_analysis_menu
from training import training_menu
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, OneHotEncoder
import os

def replace_zero_cholesterol_with_mode(df):
    df_processed = df.copy()
    
    mode_value = df_processed[df_processed['Colesterol'] != 0]['Colesterol'].mode()[0]
    
    zero_count = (df_processed['Colesterol'] == 0).sum()
    
    df_processed.loc[df_processed['Colesterol'] == 0, 'Colesterol'] = mode_value

    return df_processed

def remove_zero_cholesterol_rows(df_processed):
    # Filtrar linhas onde Colesterol != 0
    df_processed = df_processed[df_processed['Colesterol'] != 0]
    
    # Resetar índice se necessário
    df_processed.reset_index(drop=True, inplace=True)
    
    return df_processed

def replace_zero_restingBP_with_mode(df):
    df_processed = df.copy()
    
    mode_value = df_processed[df_processed['PressaoArterialRepouso'] != 0]['PressaoArterialRepouso'].mode()[0]
    
    zero_count = (df_processed['PressaoArterialRepouso'] == 0).sum()
    
    df_processed.loc[df_processed['PressaoArterialRepouso'] == 0, 'PressaoArterialRepouso'] = mode_value

    return df_processed

def remove_zero_restingBP_rows(df_processed):
    # Filtrar linhas onde PressaoArterialRepouso != 0
    df_processed = df_processed[df_processed['PressaoArterialRepouso'] != 0]
    
    # Resetar índice se necessário
    df_processed.reset_index(drop=True, inplace=True)
    
    return df_processed

def normalize_numeric_columns(df_processed):
    numeric_features = ['Idade', 'PressaoArterialRepouso', 'Colesterol', 
                       'FrequenciaCardiacaMaxima', 'DepressaoSegST']

    # Normalização das variáveis numéricas
    if numeric_features:
        scaler = StandardScaler()
        df_processed[numeric_features] = scaler.fit_transform(df_processed[numeric_features])

    return df_processed

def normalize_numeric_columns_robust(df_processed):
    numeric_features = ['Idade', 'PressaoArterialRepouso', 'Colesterol', 
                       'FrequenciaCardiacaMaxima', 'DepressaoSegST']

    # Normalização das variáveis numéricas
    if numeric_features:
        scaler = RobustScaler()
        df_processed[numeric_features] = scaler.fit_transform(df_processed[numeric_features])

    return df_processed

def zeros_become_median(df_processed):
    df_processed[['PressaoArterialRepouso']] = df_processed[[ 'PressaoArterialRepouso']].replace(0, np.nan)
    median_glicemia = df_processed['PressaoArterialRepouso'].median()
    df_processed['PressaoArterialRepouso'] = df_processed['PressaoArterialRepouso'].fillna(median_glicemia)

    return df_processed

def make_negatives_positives(df_processed):
    df_processed['DepressaoSegST'] = df_processed['DepressaoSegST'].clip(lower=-2.5, upper=6)
    return df_processed


def create_indexes_for_categorical_columns(df_processed):
    categorical_features = ['Sexo', 'TipoDorToracica', 'GlicemiaJejum', 
                           'ECGRepouso', 'DorAngina', 'InfradesnivelamentoSegST']

    # Transformação das variáveis categóricas em índices numéricos
    for col in categorical_features:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
    

    return df_processed

def little_basic_preprocess(df):
    df_processed = df.copy()
    
    df_processed = create_indexes_for_categorical_columns(df_processed)
    return df_processed

def basic_preprocess(df):
    df_processed = df.copy()
    
    df_processed = normalize_numeric_columns(df_processed)
    df_processed = create_indexes_for_categorical_columns(df_processed)
    
    return df_processed

def intermediary_A_preprocess(df):
    df_processed = df.copy()
    
    df_processed = remove_zero_cholesterol_rows(df_processed)
    df_processed = remove_zero_restingBP_rows(df_processed)
    df_processed = normalize_numeric_columns(df_processed)
    df_processed = create_indexes_for_categorical_columns(df_processed)
    
    return df_processed

def intermediary_B_preprocess(df):
    df_processed = df.copy()
    
    df_processed = replace_zero_cholesterol_with_mode(df_processed)
    df_processed = replace_zero_restingBP_with_mode(df_processed)
    df_processed = normalize_numeric_columns(df_processed)
    df_processed = create_indexes_for_categorical_columns(df_processed)
    
    return df_processed

def advanced_PreProcess(df):
    df_processed = df.copy()

    df_processed = normalize_numeric_columns_robust(df_processed)
    df_processed = zeros_become_median(df_processed)
    df_processed = make_negatives_positives(df_processed)
    df_processed = create_indexes_for_categorical_columns(df_processed)

    return df_processed 

def preprocessing_menu(df):
    """Menu para análise exploratória"""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clears terminal
        print("\n=== Pré processamento ===")
        print("1 - Basiquinho: \n-> Indexa variáveis categóricas")
        print("\n2 - Básico:  \n-> Indexa variáveis categóricas \n-> Normaliza variáveis numéricas")
        print("\n3 - Intermediário A (Recomendado):  \n-> Exclui linhas que contém colesterol == 0 \n-> Indexa variáveis categóricas \n-> Normaliza variáveis numéricas")
        print("\n4 - Intermediário B (Recomendado):  \n-> Substitui pela mola linhas que têm colesterol == 0 \n-> Indexa variáveis categóricas \n-> Normaliza variáveis numéricas")
        print("\n5 - Avançado:  \n-> Linhas com zeros possuem seus valores suubstituidos pela media \n-> valores negativos ficam positivos \n-> Elimina a coluna de glicemia \n-> Indexa variáveis categóricas \n-> Normaliza variáveis numéricas")
        #print("\n6 - Avançado:  \n-> Linhas com zeros possuem seus valores substituidos pela mediana \n-> valores negativos ficam positivos \n-> Elimina a coluna de glicemia \n-> Indexa variáveis categóricas \n-> Normaliza variáveis numéricas com o robust")
        print("\n6 - Voltar")
        
        choice = input("Escolha uma opção: ")
        
        if choice == '1':
            df_processed = little_basic_preprocess(df)
            redirect_to_training_menu(df_processed)
        if choice == '2':
            df_processed = basic_preprocess(df)
            redirect_to_training_menu(df_processed)
        if choice == '3':
            df_processed = intermediary_A_preprocess(df)
            redirect_to_training_menu(df_processed)
        if choice == '4':
            df_processed = intermediary_B_preprocess(df)
            redirect_to_training_menu(df_processed)
        elif choice == '5':
            df_processed = advanced_PreProcess(df)
            redirect_to_training_menu(df_processed)
        elif choice == '6':
            break
        else:
            print("Opção inválida. Tente novamente.")

def redirect_to_training_menu(df):
    """Menu para análise exploratória"""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clears terminal
        print("\n=== Pré processamento ===")
        print("1 - Treinar")
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
