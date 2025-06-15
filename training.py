import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, confusion_matrix)
from sklearn.model_selection import learning_curve
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
    plot_model_metrics(best_mlp, X, y)

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

def plot_model_metrics(model, X, y):
    unique_classes = sorted(y.unique())

    if set(unique_classes) == {-1.0, 0.0}:
        pos_label = 0.0
    elif set(unique_classes) == {0.0, 1.0}:
        pos_label = 1.0
    else:
        # Default case - use the less frequent class as positive
        pos_label = y.value_counts().idxmin()
    
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Calculate metrics with explicit pos_label
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, pos_label=pos_label)
    recall = recall_score(y, y_pred, pos_label=pos_label)
    f1 = f1_score(y, y_pred, pos_label=pos_label)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y, y_proba, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    
    # Calculate precision-recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y, y_proba, pos_label=pos_label)
    pr_auc = auc(recall_curve, precision_curve)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y, y_pred)
    class_labels = [f'Saúdavel ({-1.0 if pos_label == 0.0 else 0.0})', 
                   f'Doente ({pos_label})']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, 
                yticklabels=class_labels)
    plt.title('Matriz de Confusão')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Previsto')
    plt.tight_layout()
    plt.show()
    input("Pressione Enter para ver o próximo gráfico...")
    
    # 2. ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Positivos Falsos')
    plt.ylabel('Taxa de Positivos Verdadeiros')
    plt.title('Característica Operacional do Receptor')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    input("Pressione Enter para ver o próximo gráfico...")
    
    # 3. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, 
             label=f'Precision-Recall Curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Revocação')
    plt.ylabel('Precisão')
    plt.title('Curva de Precisão vs Revocação')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()
    input("Pressione Enter para ver o próximo gráfico...")
    
    # 4. Metrics Bar Plot
    plt.figure(figsize=(8, 6))
    metrics = ['Acuracia', 'Precisão', 'Revocação', 'F1 Score']
    values = [accuracy, precision, recall, f1]
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
    bars = plt.bar(metrics, values, color=colors)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')
    
    plt.title('Metricas de Performance do Modelo')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.show()
    input("Pressione Enter para ver o próximo gráfico...")
    
    # 5. Detailed F1 Score and PR Comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(['F1 Score'], [f1], color='gold')
    plt.text(0, f1/2, f'F1 = {f1:.4f}', ha='center', va='center', color='black', fontsize=12)
    plt.ylim(0, 1.1)
    plt.title('F1-Score Detalhado')
    
    plt.subplot(1, 2, 2)
    plt.bar(['Previsão', 'Revocação'], [precision, recall], color=['lightgreen', 'lightcoral'])
    plt.text(0, precision/2, f'{precision:.4f}', ha='center', va='center', color='black', fontsize=10)
    plt.text(1, recall/2, f'{recall:.4f}', ha='center', va='center', color='black', fontsize=10)
    plt.ylim(0, 1.1)
    plt.title('Previsão vs Revocação')
    
    plt.tight_layout()
    plt.show()
    input("Pressione Enter para finalizar...")