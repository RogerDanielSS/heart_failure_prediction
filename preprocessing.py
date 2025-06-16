import pandas as pd
import numpy as np
from exploratory_anal.exploratory_anal import exploratory_analysis_menu
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

def add_age_groups(df_processed, age_column='Idade'):

    bins = [20, 30, 40, 50, 60, 70, np.inf]
    labels = [0, 1, 2, 3, 4, 5]  # Ordinal encoding
    
    df_processed[age_column] = pd.cut(
        df_processed[age_column],
        bins=bins,
        labels=labels,
        right=False
    ).astype(int)  
    return df_processed

def remove_zero_restingBP_rows(df_processed): #there is no 0 values in restingBP
    # Filtrar linhas onde PressaoArterialRepouso != 0
    df_processed = df_processed[df_processed['PressaoArterialRepouso'] != 0]
    
    # Resetar índice se necessário
    df_processed.reset_index(drop=True, inplace=True)
    
    return df_processed

def remove_zero_DepressaoSegST_rows(df_processed):
    # Filtrar linhas onde DepressaoSegST != 0
    df_processed = df_processed[df_processed['DepressaoSegST'] != 0]
    
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

def normalize_columns_robust(df_processed):
    numeric_features = df_processed.select_dtypes(include=['int64', 'float64'])
    categorical_features = ['Sexo', 'TipoDorToracica', 'GlicemiaJejum', 
                           'ECGRepouso', 'DorAngina']
    ordinal_features = ['InfradesnivelamentoSegST']

    # Normalização das variáveis numéricas
    if not numeric_features.empty:
        scaler = RobustScaler()
        df_processed[numeric_features.columns] = scaler.fit_transform(df_processed[numeric_features.columns])

    if len(categorical_features) > 0:
        # Normalização das variáveis categoricas usa o One-Hot
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded_features = encoder.fit_transform(df_processed[categorical_features])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
        df_processed = pd.concat([df_processed.drop(categorical_features, axis=1), encoded_df], axis=1)

    if len(ordinal_features) > 0:
        ordem = {'Down': 0, 'Flat': 1, 'Up': 2}
        df_processed['InfradesnivelamentoSegST'] = df_processed['InfradesnivelamentoSegST'].map(ordem).fillna(1)
    
    return df_processed

def zeros_become_median(df_processed):
    df_processed[['DepressaoSegST']] = df_processed[[ 'DepressaoSegST']].replace(0, np.nan)
    median_glicemia = df_processed['DepressaoSegST'].median()
    df_processed['DepressaoSegST'] = df_processed['DepressaoSegST'].fillna(median_glicemia)

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

def handle_colesterol_MANR(df_processed):

    df_processed['ColesterolPresente'] = (df_processed['Colesterol'] == 0).astype(int)
    df_processed = df_processed.drop('Colesterol', axis=1)
    return df_processed

def handle_SegST_MANR(df_processed):

    df_processed['DepressaoSegStExiste'] = (df_processed['DepressaoSegST'] == 0).astype(int)
    df_processed = df_processed.drop('DepressaoSegST', axis=1)
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

    df_processed = handle_SegST_MANR(df_processed)
    df_processed = handle_colesterol_MANR(df_processed)
    df_processed = add_age_groups(df_processed)
    df_processed = normalize_columns_robust(df_processed)
    return df_processed 

def preprocessing_menu(df):
    """Menu para análise exploratória"""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clears terminal
        print("\n=== Pré processamento ===")
        print("1 - Basiquinho: \n-> Indexa variáveis categóricas")
        print("\n2 - Básico:  \n-> Indexa variáveis categóricas \n-> Normaliza variáveis numéricas")
        print("\n3 - Intermediário A:  \n-> Exclui linhas que contém colesterol == 0 \n-> Exclui linhas que contém pressao_arterial_repolso == 0 \n-> Indexa variáveis categóricas \n-> Normaliza variáveis numéricas")
        print("\n4 - Intermediário B:  \n-> Substitui pela moda linhas que têm colesterol == 0 \n-> Substitui pela moda linhas que têm pressao_arterial_repolso == 0 \n-> Indexa variáveis categóricas \n-> Normaliza variáveis numéricas")
        print("\n5 - Avançado(Recomendado):  \n-> Linhas com zeros possuem seus valores substituidos pela media \n-> valores negativos ficam positivos \n-> Elimina a coluna de glicemia \n-> Indexa variáveis categóricas \n-> Normaliza variáveis numéricas")
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
