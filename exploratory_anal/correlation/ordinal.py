import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def ordinal_menu(df):
    """Análise das distribuições das variáveis com menu interativo"""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clears terminal
        print("\n=== Análise de Distribuições ===")
        print("Selecione a variável para análise:")
        print("1 - Correlação entre DoencaCardiaca e InfradesnivelamentoSegST")
        print("2 - Voltar")
        choice = input("Escolha uma opção (1-13): ")
        
        if choice == '1':
            # Correlação entre DoencaCardiaca e InfradesnivelamentoSegST
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x='InfradesnivelamentoSegST', hue='DoencaCardiaca',
                         palette=['#1f77b4', '#ff7f0e'])
            plt.title('Distribuição de Doença Cardíaca por Infradesnivelamento do Segmento ST')
            plt.xticks(rotation=0)
            plt.show()
            
            print("\nTabela de contingência:")
            print(pd.crosstab(df['InfradesnivelamentoSegST'], df['DoencaCardiaca'], normalize='index') * 100)
        
        elif choice == '2':
            break
            
        else:
            print("Opção inválida. Tente novamente.")

