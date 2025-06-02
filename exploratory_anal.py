import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

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
    """Análise de correlações entre variáveis com menu interativo"""
    while True:
        print("\n=== Análise de Correlações ===")
        print("Selecione o tipo de análise:")
        print("1 - Matriz de correlação completa")
        print("2 - Correlação entre DoencaCardiaca e Idade")
        print("3 - Correlação entre DoencaCardiaca e Sexo")
        print("4 - Correlação entre DoencaCardiaca e TipoDorToracica")
        print("5 - Correlação entre DoencaCardiaca e PressaoArterialRepouso")
        print("6 - Correlação entre DoencaCardiaca e Colesterol")
        print("7 - Correlação entre DoencaCardiaca e GlicemiaJejum")
        print("8 - Correlação entre DoencaCardiaca e ECGRepouso")
        print("9 - Correlação entre DoencaCardiaca e FrequenciaCardiacaMaxima")
        print("10 - Correlação entre DoencaCardiaca e DorAngina")
        print("11 - Correlação entre DoencaCardiaca e DepressaoSegST")
        print("12 - Correlação entre DoencaCardiaca e InfradesnivelamentoSegST")
        print("13 - Voltar ao menu anterior")
        
        choice = input("Escolha uma opção (1-13): ")
        
        if choice == '1':
            # Matriz de correlação completa
            plt.figure(figsize=(12, 8))
            corr_matrix = df.corr(numeric_only=True)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
            plt.title('Matriz de Correlação Completa')
            plt.show()
            
            print("\nCorrelações com DoencaCardiaca (ordenadas):")
            print(corr_matrix['DoencaCardiaca'].sort_values(ascending=False))
            
        elif choice == '2':
            # Correlação entre DoencaCardiaca e Idade
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x='DoencaCardiaca', y='Idade')
            plt.title('Relação entre Doença Cardíaca e Idade')
            plt.show()
            
            # Cálculo do coeficiente de correlação
            corr = df['DoencaCardiaca'].corr(df['Idade'])
            print(f"\nCoeficiente de correlação: {corr:.3f}")
            
        elif choice == '3':
            # Correlação entre DoencaCardiaca e Sexo
            plt.figure(figsize=(8, 5))
            sns.countplot(data=df, x='Sexo', hue='DoencaCardiaca')
            plt.title('Distribuição de Doença Cardíaca por Sexo')
            plt.show()
            
            # Tabela de contingência
            print("\nTabela de contingência:")
            print(pd.crosstab(df['Sexo'], df['DoencaCardiaca'], normalize='index') * 100)
            
        elif choice == '4':
            # Correlação entre DoencaCardiaca e TipoDorToracica
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x='TipoDorToracica', hue='DoencaCardiaca')
            plt.title('Distribuição de Doença Cardíaca por Tipo de Dor Torácica')
            plt.xticks(rotation=0)
            plt.show()
            
            # Tabela de contingência
            print("\nTabela de contingência:")
            print(pd.crosstab(df['TipoDorToracica'], df['DoencaCardiaca'], normalize='index') * 100)
            
        elif choice == '5':
            # Correlação entre DoencaCardiaca e PressaoArterialRepouso
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x='DoencaCardiaca', y='PressaoArterialRepouso')
            plt.title('Relação entre Doença Cardíaca e Pressão Arterial em Repouso')
            plt.show()
            
            # Cálculo do coeficiente de correlação
            corr = df['DoencaCardiaca'].corr(df['PressaoArterialRepouso'])
            print(f"\nCoeficiente de correlação: {corr:.3f}")
            
        elif choice == '6':
            # Correlação entre DoencaCardiaca e Colesterol
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x='DoencaCardiaca', y='Colesterol')
            plt.title('Relação entre Doença Cardíaca e Colesterol')
            plt.show()
            
            # Cálculo do coeficiente de correlação
            corr = df['DoencaCardiaca'].corr(df['Colesterol'])
            print(f"\nCoeficiente de correlação: {corr:.3f}")
            
        elif choice == '7':
            # Correlação entre DoencaCardiaca e GlicemiaJejum
            plt.figure(figsize=(8, 5))
            sns.countplot(data=df, x='GlicemiaJejum', hue='DoencaCardiaca')
            plt.title('Distribuição de Doença Cardíaca por Glicemia em Jejum')
            plt.show()
            
            # Tabela de contingência
            print("\nTabela de contingência:")
            print(pd.crosstab(df['GlicemiaJejum'], df['DoencaCardiaca'], normalize='index') * 100)
            
        elif choice == '8':
            # Correlação entre DoencaCardiaca e ECGRepouso
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x='ECGRepouso', hue='DoencaCardiaca')
            plt.title('Distribuição de Doença Cardíaca por ECG em Repouso')
            plt.xticks(rotation=0)
            plt.show()
            
            # Tabela de contingência
            print("\nTabela de contingência:")
            print(pd.crosstab(df['ECGRepouso'], df['DoencaCardiaca'], normalize='index') * 100)
            
        elif choice == '9':
            # Correlação entre DoencaCardiaca e FrequenciaCardiacaMaxima
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x='DoencaCardiaca', y='FrequenciaCardiacaMaxima')
            plt.title('Relação entre Doença Cardíaca e Frequência Cardíaca Máxima')
            plt.show()
            
            # Cálculo do coeficiente de correlação
            corr = df['DoencaCardiaca'].corr(df['FrequenciaCardiacaMaxima'])
            print(f"\nCoeficiente de correlação: {corr:.3f}")
            
        elif choice == '10':
            # Correlação entre DoencaCardiaca e DorAngina
            plt.figure(figsize=(8, 5))
            sns.countplot(data=df, x='DorAngina', hue='DoencaCardiaca')
            plt.title('Distribuição de Doença Cardíaca por Dor de Angina')
            plt.show()
            
            # Tabela de contingência
            print("\nTabela de contingência:")
            print(pd.crosstab(df['DorAngina'], df['DoencaCardiaca'], normalize='index') * 100)
            
        elif choice == '11':
            # Correlação entre DoencaCardiaca e DepressaoSegST
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x='DoencaCardiaca', y='DepressaoSegST')
            plt.title('Relação entre Doença Cardíaca e Depressão do Segmento ST')
            plt.show()
            
            # Cálculo do coeficiente de correlação
            corr = df['DoencaCardiaca'].corr(df['DepressaoSegST'])
            print(f"\nCoeficiente de correlação: {corr:.3f}")
            
        elif choice == '12':
            # Correlação entre DoencaCardiaca e InfradesnivelamentoSegST
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x='InfradesnivelamentoSegST', hue='DoencaCardiaca')
            plt.title('Distribuição de Doença Cardíaca por Infradesnivelamento do Segmento ST')
            plt.xticks(rotation=0)
            plt.show()
            
            # Tabela de contingência
            print("\nTabela de contingência:")
            print(pd.crosstab(df['InfradesnivelamentoSegST'], df['DoencaCardiaca'], normalize='index') * 100)
            
        elif choice == '13':
            break
            
        else:
            print("Opção inválida. Tente novamente.")


def analyze_correlations(df):
    """Análise de correlações entre variáveis com menu interativo"""
    while True:
        print("\n=== Análise de Correlações ===")
        print("Selecione o tipo de análise:")
        print("1 - Matriz de correlação completa")
        print("2 - Correlação entre DoencaCardiaca e Idade")
        print("3 - Correlação entre DoencaCardiaca e Sexo")
        print("4 - Correlação entre DoencaCardiaca e TipoDorToracica")
        print("5 - Correlação entre DoencaCardiaca e PressaoArterialRepouso")
        print("6 - Correlação entre DoencaCardiaca e Colesterol")
        print("7 - Correlação entre DoencaCardiaca e GlicemiaJejum")
        print("8 - Correlação entre DoencaCardiaca e ECGRepouso")
        print("9 - Correlação entre DoencaCardiaca e FrequenciaCardiacaMaxima")
        print("10 - Correlação entre DoencaCardiaca e DorAngina")
        print("11 - Correlação entre DoencaCardiaca e DepressaoSegST")
        print("12 - Correlação entre DoencaCardiaca e InfradesnivelamentoSegST")
        print("13 - Voltar ao menu anterior")
        
        choice = input("Escolha uma opção (1-13): ")
        
        if choice == '1':
            # Matriz de correlação completa
            plt.figure(figsize=(12, 8))
            corr_matrix = df.corr(numeric_only=True)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
            plt.title('Matriz de Correlação Completa')
            plt.show()
            
            print("\nCorrelações com DoencaCardiaca (ordenadas):")
            print(corr_matrix['DoencaCardiaca'].sort_values(ascending=False))
            
        elif choice == '2':
            # Correlação entre DoencaCardiaca e Idade
            plt.figure(figsize=(12, 6))
            sns.histplot(data=df, x='Idade', hue='DoencaCardiaca', kde=True, bins=20,
                        palette=['#1f77b4', '#ff7f0e'], alpha=0.6)
            plt.title('Distribuição de Idade por Status de Doença Cardíaca')
            plt.xlabel('Idade (anos)')
            plt.ylabel('Contagem')
            plt.show()
            
            corr = df['DoencaCardiaca'].corr(df['Idade'])
            print(f"\nCoeficiente de correlação: {corr:.3f}")
            
        elif choice == '3':
            # Correlação entre DoencaCardiaca e Sexo
            plt.figure(figsize=(8, 5))
            sns.countplot(data=df, x='Sexo', hue='DoencaCardiaca', 
                         palette=['#1f77b4', '#ff7f0e'])
            plt.title('Distribuição de Doença Cardíaca por Sexo')
            plt.show()
            
            print("\nTabela de contingência:")
            print(pd.crosstab(df['Sexo'], df['DoencaCardiaca'], normalize='index') * 100)
            
        elif choice == '4':
            # Correlação entre DoencaCardiaca e TipoDorToracica
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x='TipoDorToracica', hue='DoencaCardiaca',
                         palette=['#1f77b4', '#ff7f0e'])
            plt.title('Distribuição de Doença Cardíaca por Tipo de Dor Torácica')
            plt.xticks(rotation=0)
            plt.show()
            
            print("\nTabela de contingência:")
            print(pd.crosstab(df['TipoDorToracica'], df['DoencaCardiaca'], normalize='index') * 100)
            
        elif choice == '5':
            # Correlação entre DoencaCardiaca e PressaoArterialRepouso
            plt.figure(figsize=(12, 6))
            sns.histplot(data=df, x='PressaoArterialRepouso', hue='DoencaCardiaca', kde=True, bins=20,
                        palette=['#1f77b4', '#ff7f0e'], alpha=0.6)
            plt.title('Distribuição de Pressão Arterial por Status de Doença Cardíaca')
            plt.xlabel('Pressão Arterial em Repouso (mmHg)')
            plt.ylabel('Contagem')
            plt.show()
            
            corr = df['DoencaCardiaca'].corr(df['PressaoArterialRepouso'])
            print(f"\nCoeficiente de correlação: {corr:.3f}")
            
        elif choice == '6':
            # Correlação entre DoencaCardiaca e Colesterol
            plt.figure(figsize=(12, 6))
            sns.histplot(data=df, x='Colesterol', hue='DoencaCardiaca', kde=True, bins=20,
                        palette=['#1f77b4', '#ff7f0e'], alpha=0.6)
            plt.title('Distribuição de Colesterol por Status de Doença Cardíaca')
            plt.xlabel('Colesterol (mg/dL)')
            plt.ylabel('Contagem')
            plt.show()
            
            corr = df['DoencaCardiaca'].corr(df['Colesterol'])
            print(f"\nCoeficiente de correlação: {corr:.3f}")
            
        elif choice == '7':
            # Correlação entre DoencaCardiaca e GlicemiaJejum
            plt.figure(figsize=(8, 5))
            sns.countplot(data=df, x='GlicemiaJejum', hue='DoencaCardiaca',
                         palette=['#1f77b4', '#ff7f0e'])
            plt.title('Distribuição de Doença Cardíaca por Glicemia em Jejum')
            plt.show()
            
            print("\nTabela de contingência:")
            print(pd.crosstab(df['GlicemiaJejum'], df['DoencaCardiaca'], normalize='index') * 100)
            
        elif choice == '8':
            # Correlação entre DoencaCardiaca e ECGRepouso
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x='ECGRepouso', hue='DoencaCardiaca',
                         palette=['#1f77b4', '#ff7f0e'])
            plt.title('Distribuição de Doença Cardíaca por ECG em Repouso')
            plt.xticks(rotation=0)
            plt.show()
            
            print("\nTabela de contingência:")
            print(pd.crosstab(df['ECGRepouso'], df['DoencaCardiaca'], normalize='index') * 100)
            
        elif choice == '9':
            # Correlação entre DoencaCardiaca e FrequenciaCardiacaMaxima
            plt.figure(figsize=(12, 6))
            sns.histplot(data=df, x='FrequenciaCardiacaMaxima', hue='DoencaCardiaca', kde=True, bins=20,
                        palette=['#1f77b4', '#ff7f0e'], alpha=0.6)
            plt.title('Distribuição de Frequência Cardíaca Máxima por Status de Doença')
            plt.xlabel('Frequência Cardíaca Máxima (bpm)')
            plt.ylabel('Contagem')
            plt.show()
            
            corr = df['DoencaCardiaca'].corr(df['FrequenciaCardiacaMaxima'])
            print(f"\nCoeficiente de correlação: {corr:.3f}")

        elif choice == '10':
            # Correlação entre DoencaCardiaca e DorAngina
            plt.figure(figsize=(8, 5))
            sns.countplot(data=df, x='DorAngina', hue='DoencaCardiaca',
                         palette=['#1f77b4', '#ff7f0e'])
            plt.title('Distribuição de Doença Cardíaca por Dor de Angina')
            plt.show()
            
            print("\nTabela de contingência:")
            print(pd.crosstab(df['DorAngina'], df['DoencaCardiaca'], normalize='index') * 100)
            
        elif choice == '11':
            # Correlação entre DoencaCardiaca e DepressaoSegST
            plt.figure(figsize=(12, 6))
            sns.histplot(data=df, x='DepressaoSegST', hue='DoencaCardiaca', kde=True, bins=20,
                        palette=['#1f77b4', '#ff7f0e'], alpha=0.6)
            plt.title('Distribuição de Depressão do Segmento ST por Status de Doença')
            plt.xlabel('Depressão do Segmento ST')
            plt.ylabel('Contagem')
            plt.show()
            
            corr = df['DoencaCardiaca'].corr(df['DepressaoSegST'])
            print(f"\nCoeficiente de correlação: {corr:.3f}")
            
        elif choice == '12':
            # Correlação entre DoencaCardiaca e InfradesnivelamentoSegST
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x='InfradesnivelamentoSegST', hue='DoencaCardiaca',
                         palette=['#1f77b4', '#ff7f0e'])
            plt.title('Distribuição de Doença Cardíaca por Infradesnivelamento do Segmento ST')
            plt.xticks(rotation=0)
            plt.show()
            
            print("\nTabela de contingência:")
            print(pd.crosstab(df['InfradesnivelamentoSegST'], df['DoencaCardiaca'], normalize='index') * 100)
            
        elif choice == '13':
            break
            
        else:
            print("Opção inválida. Tente novamente.")

           
# def analyze_outliers(df):
#     """Identificação e análise de outliers"""
#     print("\n=== Análise de Outliers ===")
    
#     num_cols = ['Idade', 'PressaoArterialRepouso', 'Colesterol', 'FrequenciaCardiacaMaxima', 'DepressaoSegST']
    
#     # Boxplots para visualização
#     plt.figure(figsize=(12, 6))
#     df[num_cols].boxplot()
#     plt.xticks(rotation=0)
#     plt.title('Boxplots para Identificação de Outliers')
#     plt.show()
    
#     # Identificação usando Z-score
#     z_scores = np.abs(stats.zscore(df[num_cols]))
#     outliers = (z_scores > 3).any(axis=1)
#     print(f"\nNúmero de outliers (Z-score > 3): {outliers.sum()}")
#     print("\nRegistros considerados outliers:")
#     print(df[outliers])

def analyze_outliers(df):
    """Identificação e análise de outliers"""
    num_cols = ['Idade', 'PressaoArterialRepouso', 'Colesterol', 'FrequenciaCardiacaMaxima', 'DepressaoSegST']
    cat_cols = [col for col in df.columns if col not in num_cols and df[col].dtype == 'object']
    
    while True:
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
            if not cat_cols:
                print("\nNão há colunas textuais no dataframe.")
                return
                
            print("\nValores únicos em colunas textuais:")
            for col in cat_cols:
                unique_vals = df[col].unique()
                print(f"\nColuna '{col}':")
                print(unique_vals)
            
        elif opcao == '3':
            break
                
        else:
            print("Opção inválida. Por favor escolha 1 ou 2.")

def analyze_distributions(df):
    """Análise das distribuições das variáveis com menu interativo"""
    while True:
        print("\n=== Análise de Distribuições ===")
        print("Selecione a variável para análise:")
        print("1 - Idade")
        print("2 - Sexo")
        print("3 - TipoDorToracica")
        print("4 - PressaoArterialRepouso")
        print("5 - Colesterol")
        print("6 - GlicemiaJejum")
        print("7 - ECGRepouso")
        print("8 - FrequenciaCardiacaMaxima")
        print("9 - DorAngina")
        print("10 - DepressaoSegST")
        print("11 - InfradesnivelamentoSegST")
        print("12 - DoencaCardiaca")
        print("13 - Voltar ao menu anterior")
        
        choice = input("Escolha uma opção (1-13): ")
        
        if choice == '1':
            # Idade
            plt.figure(figsize=(10, 5))
            sns.histplot(data=df, x='Idade', kde=True, bins=20)
            plt.title('Distribuição de Idade')
            plt.show()
            
        elif choice == '2':
            # Sexo
            plt.figure(figsize=(8, 4))
            sns.countplot(data=df, x='Sexo')
            plt.title('Distribuição por Sexo')
            plt.show()
            
        elif choice == '3':
            # TipoDorToracica
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x='TipoDorToracica', order=df['TipoDorToracica'].value_counts().index)
            plt.title('Distribuição por Tipo de Dor Torácica')
            plt.xticks(rotation=0)
            plt.show()
            
        elif choice == '4':
            # PressaoArterialRepouso
            plt.figure(figsize=(10, 5))
            sns.histplot(data=df, x='PressaoArterialRepouso', kde=True, bins=20)
            plt.title('Distribuição de Pressão Arterial em Repouso')
            plt.show()
            
        elif choice == '5':
            # Colesterol
            plt.figure(figsize=(10, 5))
            sns.histplot(data=df, x='Colesterol', kde=True, bins=20)
            plt.title('Distribuição de Colesterol')
            plt.show()
            
        elif choice == '6':
            # GlicemiaJejum
            plt.figure(figsize=(8, 4))
            sns.countplot(data=df, x='GlicemiaJejum')
            plt.title('Distribuição de Glicemia em Jejum')
            plt.show()
            
        elif choice == '7':
            # ECGRepouso
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x='ECGRepouso', order=df['ECGRepouso'].value_counts().index)
            plt.title('Distribuição de ECG em Repouso')
            plt.xticks(rotation=0)
            plt.show()
            
        elif choice == '8':
            # FrequenciaCardiacaMaxima
            plt.figure(figsize=(10, 5))
            sns.histplot(data=df, x='FrequenciaCardiacaMaxima', kde=True, bins=20)
            plt.title('Distribuição de Frequência Cardíaca Máxima')
            plt.show()
            
        elif choice == '9':
            # DorAngina
            plt.figure(figsize=(8, 4))
            sns.countplot(data=df, x='DorAngina')
            plt.title('Distribuição de Dor de Angina')
            plt.show()
            
        elif choice == '10':
            # DepressaoSegST
            plt.figure(figsize=(10, 5))
            sns.histplot(data=df, x='DepressaoSegST', kde=True, bins=20)
            plt.title('Distribuição de Depressão do Segmento ST')
            plt.show()
            
        elif choice == '11':
            # InfradesnivelamentoSegST
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x='InfradesnivelamentoSegST', order=df['InfradesnivelamentoSegST'].value_counts().index)
            plt.title('Distribuição de Infradesnivelamento do Segmento ST')
            plt.xticks(rotation=0)
            plt.show()
            
        elif choice == '12':
            # DoencaCardiaca
            plt.figure(figsize=(8, 4))
            sns.countplot(data=df, x='DoencaCardiaca')
            plt.title('Distribuição de Doença Cardíaca')
            plt.show()
            
        elif choice == '13':
            break
            
        else:
            print("Opção inválida. Tente novamente.")


def exploratory_analysis_menu(df):
    """Menu para análise exploratória"""
    while True:
        print("\n=== Análise Exploratória ===")
        print("1 - Correlações")
        print("2 - Outliers")
        print("3 - Distribuição das variáveis")
        print("4 - Voltar ao menu principal")
        
        choice = input("Escolha uma opção: ")
        
        if choice == '1':
            analyze_correlations(df)
        elif choice == '2':
            analyze_outliers(df)
        elif choice == '3':
            analyze_distributions(df)
        elif choice == '4':
            break
        else:
            print("Opção inválida. Tente novamente.")
