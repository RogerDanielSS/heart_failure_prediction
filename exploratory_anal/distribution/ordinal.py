import os
import matplotlib.pyplot as plt
import seaborn as sns

def ordinal_menu(df):
    """Análise das distribuições das variáveis com menu interativo"""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clears terminal
        print("\n=== Análise de Distribuições ===")
        print("Selecione a variável para análise:")
        print("1 - InfradesnivelamentoSegST")
        print("2 - Voltar")
        choice = input("Escolha uma opção (1-13): ")
        
        if choice == '1':
            # InfradesnivelamentoSegST
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x='InfradesnivelamentoSegST', order=df['InfradesnivelamentoSegST'].value_counts().index)
            plt.title('Distribuição de Infradesnivelamento do Segmento ST')
            plt.xticks(rotation=0)
            plt.show()
        
        elif choice == '2':
            break
            
        else:
            print("Opção inválida. Tente novamente.")

