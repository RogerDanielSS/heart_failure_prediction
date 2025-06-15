import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def numeric_menu(df):
    """Análise das distribuições das variáveis com menu interativo"""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clears terminal
        print("\n=== Análise de Distribuições ===")
        print("Selecione a variável para análise:")
        print("1 - Correlação entre DoencaCardiaca e Idade")
        print("2 - Correlação entre DoencaCardiaca e PressaoArterialRepouso")
        if 'Colesterol' in df.columns:
            print("3 - Correlação entre DoencaCardiaca e Colesterol")
        elif 'ColesterolPresente' in df.columns: 
            print("3 - Correlação entre DoencaCardiaca e ColesterolPresente")
        print("4 - Correlação entre DoencaCardiaca e FrequenciaCardiacaMaxima")
        if 'DepressaoSegST' in df.columns:
            print("5 - Correlação entre DoencaCardiaca e DepressaoSegST")
        elif 'DepressaoSegStExiste' in df.columns: 
            print("5 - Correlação entre DoencaCardiaca e DepressaoSegStExiste")
        print("6 - Voltar")
        choice = input("Escolha uma opção (1-6): ")
        
        if choice == '1':
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

        elif choice == '2':
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

        elif choice == '3':
            # Colesterol
            if 'Colesterol' in df.columns:
                plt.figure(figsize=(12, 6))
                sns.histplot(data=df, x='Colesterol', hue='DoencaCardiaca', kde=True, bins=20,
                            palette=['#1f77b4', '#ff7f0e'], alpha=0.6)
                plt.title('Distribuição de Colesterol por Status de Doença Cardíaca')
                plt.xlabel('Colesterol (mg/dL)')
                plt.ylabel('Contagem')
                plt.show()
                
                corr = df['DoencaCardiaca'].corr(df['Colesterol'])
                print(f"\nCoeficiente de correlação: {corr:.3f}")
            elif 'ColesterolPresente' in df.columns: 
                plt.figure(figsize=(12, 6))
                sns.histplot(data=df, x='ColesterolPresente', hue='DoencaCardiaca', kde=True, bins=20,
                            palette=['#1f77b4', '#ff7f0e'], alpha=0.6)
                plt.title('Distribuição de ColesterolPresente por Status de Doença Cardíaca')
                plt.xlabel('ColesterolPresente (mg/dL)')
                plt.ylabel('Contagem')
                plt.show()
                
                corr = df['DoencaCardiaca'].corr(df['ColesterolPresente'])
                print(f"\nCoeficiente de correlação: {corr:.3f}")

        elif choice == '4':
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
            
        elif choice == '5':
            # DepressaoSegST
            if 'DepressaoSegST' in df.columns:
                plt.figure(figsize=(12, 6))
                sns.histplot(data=df, x='DepressaoSegST', hue='DoencaCardiaca', kde=True, bins=20,
                            palette=['#1f77b4', '#ff7f0e'], alpha=0.6)
                plt.title('Distribuição de Depressão do Segmento ST por Status de Doença')
                plt.xlabel('Depressão do Segmento ST')
                plt.ylabel('Contagem')
                plt.show()
                
                corr = df['DoencaCardiaca'].corr(df['DepressaoSegST'])
                print(f"\nCoeficiente de correlação: {corr:.3f}")

            elif 'DepressaoSegSTExiste' in df.columns: 
                plt.figure(figsize=(12, 6))
                sns.histplot(data=df, x='DepressaoSegSTExiste', hue='DoencaCardiaca', kde=True, bins=20,
                            palette=['#1f77b4', '#ff7f0e'], alpha=0.6)
                plt.title('Distribuição de Depressão do Segmento ST por Status de Doença')
                plt.xlabel('Depressão do Segmento ST')
                plt.ylabel('Contagem')
                plt.show()
                
                corr = df['DoencaCardiaca'].corr(df['DepressaoSegSTExiste'])
                print(f"\nCoeficiente de correlação: {corr:.3f}")
            
        elif choice == '6':
            break
            
        else:
            print("Opção inválida. Tente novamente.")

