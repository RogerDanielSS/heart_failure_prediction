import os
import matplotlib.pyplot as plt
import seaborn as sns

def categorical_menu(df):
    """Análise das distribuições das variáveis com menu interativo"""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clears terminal
        print("\n=== Análise de Distribuições ===")
        print("Selecione a variável para análise:")
        print("1 - Sexo")
        print("2 - TipoDorToracica")
        print("3 - GlicemiaJejum")
        print("4 - ECGRepouso")
        print("5 - DorAngina")
        print("6 - Voltar")
        choice = input("Escolha uma opção (1-13): ")
        
        if choice == '1':
            # Sexo
            plt.figure(figsize=(8, 4))
            sns.countplot(data=df, x='Sexo')
            plt.title('Distribuição por Sexo')
            plt.show()
            
        elif choice == '2':
            # TipoDorToracica
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x='TipoDorToracica', order=df['TipoDorToracica'].value_counts().index)
            plt.title('Distribuição por Tipo de Dor Torácica')
            plt.xticks(rotation=0)
            plt.show()

        elif choice == '3':
            # GlicemiaJejum
            plt.figure(figsize=(8, 4))
            sns.countplot(data=df, x='GlicemiaJejum')
            plt.title('Distribuição de Glicemia em Jejum')
            plt.show()
            
        elif choice == '4':
            # ECGRepouso
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x='ECGRepouso', order=df['ECGRepouso'].value_counts().index)
            plt.title('Distribuição de ECG em Repouso')
            plt.xticks(rotation=0)
            plt.show()
            
        elif choice == '5':
            # DorAngina
            plt.figure(figsize=(8, 4))
            sns.countplot(data=df, x='DorAngina')
            plt.title('Distribuição de Dor de Angina')
            plt.show()
        
        elif choice == '6':
            break
            
        else:
            print("Opção inválida. Tente novamente.")
