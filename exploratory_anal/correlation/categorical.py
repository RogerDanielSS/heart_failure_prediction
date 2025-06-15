import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def categorical_menu(df):
    """Análise das distribuições das variáveis com menu interativo"""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clears terminal
        print("\n=== Análise de Distribuições ===")
        print("Selecione a variável para análise:")
        print("1 - Correlação entre DoencaCardiaca e Sexo")
        print("2 - Correlação entre DoencaCardiaca e TipoDorToracica")
        print("3 - Correlação entre DoencaCardiaca e GlicemiaJejum")
        print("4 - Correlação entre DoencaCardiaca e ECGRepouso")
        print("5 - Correlação entre DoencaCardiaca e DorAngina")
        print("6 - Voltar")
        choice = input("Escolha uma opção (1-13): ")
        
        if choice == '1':
            # Correlação entre DoencaCardiaca e Sexo
            plt.figure(figsize=(8, 5))
            sns.countplot(data=df, x='Sexo', hue='DoencaCardiaca', 
                         palette=['#1f77b4', '#ff7f0e'])
            plt.title('Distribuição de Doença Cardíaca por Sexo')
            plt.show()
            
            print("\nTabela de contingência:")
            print(pd.crosstab(df['Sexo'], df['DoencaCardiaca'], normalize='index') * 100)
            
            
        elif choice == '2':
            # Correlação entre DoencaCardiaca e TipoDorToracica
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x='TipoDorToracica', hue='DoencaCardiaca',
                         palette=['#1f77b4', '#ff7f0e'])
            plt.title('Distribuição de Doença Cardíaca por Tipo de Dor Torácica')
            plt.xticks(rotation=0)
            plt.show()
            
            print("\nTabela de contingência:")
            print(pd.crosstab(df['TipoDorToracica'], df['DoencaCardiaca'], normalize='index') * 100)
            

        elif choice == '3':
            # Correlação entre DoencaCardiaca e GlicemiaJejum
            plt.figure(figsize=(8, 5))
            sns.countplot(data=df, x='GlicemiaJejum', hue='DoencaCardiaca',
                         palette=['#1f77b4', '#ff7f0e'])
            plt.title('Distribuição de Doença Cardíaca por Glicemia em Jejum')
            plt.show()
            
            print("\nTabela de contingência:")
            print(pd.crosstab(df['GlicemiaJejum'], df['DoencaCardiaca'], normalize='index') * 100)
            
            
        elif choice == '4':
            # Correlação entre DoencaCardiaca e ECGRepouso
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x='ECGRepouso', hue='DoencaCardiaca',
                         palette=['#1f77b4', '#ff7f0e'])
            plt.title('Distribuição de Doença Cardíaca por ECG em Repouso')
            plt.xticks(rotation=0)
            plt.show()
            
            print("\nTabela de contingência:")
            print(pd.crosstab(df['ECGRepouso'], df['DoencaCardiaca'], normalize='index') * 100)
            
        elif choice == '5':
            # Correlação entre DoencaCardiaca e DorAngina
            plt.figure(figsize=(8, 5))
            sns.countplot(data=df, x='DorAngina', hue='DoencaCardiaca',
                         palette=['#1f77b4', '#ff7f0e'])
            plt.title('Distribuição de Doença Cardíaca por Dor de Angina')
            plt.show()
            
            print("\nTabela de contingência:")
            print(pd.crosstab(df['DorAngina'], df['DoencaCardiaca'], normalize='index') * 100)
        
        elif choice == '6':
            break
            
        else:
            print("Opção inválida. Tente novamente.")
