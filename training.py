import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import os

def preprocess_data(df):
    """Pré-processamento dos dados para o modelo, mantendo no DataFrame"""
    # Criar uma cópia para não modificar o original
    df_processed = df.copy()
    
    # Definir colunas numéricas e categóricas
    numeric_features = ["Idade", "PressaoArterialRepouso", "Colesterol", 
                       "FrequenciaCardiacaMaxima", "DepressaoSegST"]
    categorical_features = ["Sexo", "TipoDorToracica", "GlicemiaJejum", 
                           "ECGRepouso", "DorAngina", "InfradesnivelamentoSegST"]
    
    # Aplicar StandardScaler nas features numéricas
    scaler = StandardScaler()
    df_processed[numeric_features] = scaler.fit_transform(df_processed[numeric_features])
    
    # Aplicar OneHotEncoder nas features categóricas
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_features = encoder.fit_transform(df_processed[categorical_features])
    
    # Obter nomes das novas colunas
    encoded_columns = []
    for i, col in enumerate(categorical_features):
        categories = encoder.categories_[i]
        for cat in categories:
            encoded_columns.append(f"{col}_{cat}")
    
    # Criar DataFrame com as features codificadas
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns, index=df_processed.index)
    
    # Concatenar com as features numéricas e remover as originais categóricas
    df_processed = pd.concat([df_processed.drop(categorical_features, axis=1), encoded_df], axis=1)
    
    return df_processed

def default_training(df_processed):
    """Treina um modelo MLP com validação cruzada de 3 folds"""
    # Separar features e target
    X = df_processed.drop("DoencaCardiaca", axis=1)
    y = df_processed["DoencaCardiaca"]
    
    # Criar modelo MLP
    mlp = MLPClassifier(hidden_layer_sizes=(50, 25), 
                        activation="relu", 
                        solver="adam", 
                        max_iter=500, 
                        random_state=42)
    
    # Configurar KFold com 3 folds
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # Listas para armazenar resultados
    accuracies = []
    
    print("\n=== Treinamento com Validação Cruzada (3 folds) ===")
    
    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Treinar modelo
        mlp.fit(X_train, y_train)
        
        # Fazer previsões
        y_pred = mlp.predict(X_test)
        
        # Calcular acurácia
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        
        print(f"\nFold {fold}:")
        print(f"Acurácia: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
    
    print("\n=== Resultados Finais ===")
    print(f"Acurácia média: {np.mean(accuracies):.4f}")
    print(f"Desvio padrão: {np.std(accuracies):.4f}")
    input("\n\nPressione Enter para continuar...")

def optimized_training(df_processed):
    """Treina um modelo MLP com otimização de hiperparâmetros usando GridSearchCV e validação cruzada"""
    X = df_processed.drop("DoencaCardiaca", axis=1)
    y = df_processed["DoencaCardiaca"]

    # Definir a pipeline para pré-processamento e MLP
    # Não é necessário ColumnTransformer aqui, pois o preprocess_data já retorna um DataFrame processado
    pipeline = Pipeline([
        ("mlp", MLPClassifier(max_iter=1000, random_state=42)) # Aumentar max_iter para convergência
    ])

    # Definir o grid de parâmetros para GridSearchCV
    param_grid = {
        "mlp__hidden_layer_sizes": [(50,), (100,), (50, 25), (100, 50, 25)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__solver": ["adam", "sgd"],
        "mlp__alpha": [0.0001, 0.001, 0.01], # Parâmetro de regularização L2
        "mlp__learning_rate": ["constant", "adaptive"],
    }

    # Configurar GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42), 
                               scoring="accuracy", n_jobs=-1, verbose=2)

    print("\n=== Treinamento Otimizado com GridSearchCV ===")
    grid_search.fit(X, y)

    print("\nMelhores parâmetros encontrados:", grid_search.best_params_)
    print("Melhor acurácia (validação cruzada):", grid_search.best_score_)

    # Avaliar o melhor modelo no conjunto de dados completo (ou em um conjunto de teste separado, se disponível)
    best_mlp = grid_search.best_estimator_
    y_pred = best_mlp.predict(X)
    print("\nRelatório de Classificação no conjunto completo de dados:")
    print(classification_report(y, y_pred))
    print(f"Acurácia no conjunto completo de dados: {accuracy_score(y, y_pred):.4f}")
    input("\n\nPressione Enter para continuar...")

def training_menu(df):
    """Menu para treinamento de modelos"""
    while True:
        os.system("cls" if os.name == "nt" else "clear")  # Clears terminal
        print("\n=== Treinamento ===")
        print("1 - Treinar MLP com Validação Cruzada (3 folds) - Padrão")
        print("2 - Treinar MLP com Otimização de Hiperparâmetros (GridSearchCV)")
        print("3 - Voltar")
        
        choice = input("Escolha uma opção: ")
        
        if choice == "1":
            default_training(df)
        elif choice == "2":
            optimized_training(df)
        elif choice == "3":
            break
        else:
            print("Opção inválida. Tente novamente.")
