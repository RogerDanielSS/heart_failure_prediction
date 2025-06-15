import os
import matplotlib.pyplot as plt
import seaborn as sns

def numeric_menu(df):
    """Análise das distribuições das variáveis com menu interativo"""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clears terminal
        print("\n=== Análise de Distribuições ===")
        print("Selecione a variável para análise:")
        print("1 - Idade")
        print("2 - PressaoArterialRepouso")
        if 'Colesterol' in df.columns:
            print("3 - Colesterol")
        elif 'ColesterolPresente' in df.columns: 
            print("3 - ColesterolPresente")
        print("4 - FrequenciaCardiacaMaxima")
        if 'DepressaoSegST' in df.columns:
            print("5 - DepressaoSegST")
        elif 'DepressaoSegStExiste' in df.columns: 
            print("5 - DepressaoSegStExiste")
        print("6 - DoencaCardiaca")
        print("\n")
        if 'Colesterol' in df.columns:
          print("7 - Analise de MNAR em Colesterol")
        if 'DepressaoSegST' in df.columns:
          print("8 - Analise de MNAR em ST_SLOPE")
        print("9 - Voltar")
        choice = input("Escolha uma opção (1-13): ")
        
        if choice == '1':
            # Idade
            plt.figure(figsize=(10, 5))
            sns.histplot(data=df, x='Idade', kde=True, bins=20)
            plt.title('Distribuição de Idade')
            plt.show()

        elif choice == '2':
            # PressaoArterialRepouso
            plt.figure(figsize=(10, 5))
            sns.histplot(data=df, x='PressaoArterialRepouso', kde=True, bins=20)
            plt.title('Distribuição de Pressão Arterial em Repouso')
            plt.show()
            
        elif choice == '3':
            # Colesterol
            if 'Colesterol' in df.columns:
                plt.figure(figsize=(10, 5))
                sns.histplot(data=df, x='Colesterol', kde=True, bins=20)
                plt.title('Distribuição de Colesterol')
                plt.show()
            elif 'ColesterolPresente' in df.columns: 
                plt.figure(figsize=(8, 4))
                sns.countplot(data=df, x='ColesterolPresente')
                plt.title('Distribuição por ColesterolPresente')
                plt.show()

        elif choice == '4':
            # FrequenciaCardiacaMaxima
            plt.figure(figsize=(10, 5))
            sns.histplot(data=df, x='FrequenciaCardiacaMaxima', kde=True, bins=20)
            plt.title('Distribuição de Frequência Cardíaca Máxima')
            plt.show()
            
        elif choice == '5':
            # DepressaoSegST
            if 'DepressaoSegST' in df.columns:
              plt.figure(figsize=(10, 5))
              sns.histplot(data=df, x='DepressaoSegST', kde=True, bins=20)
              plt.title('Distribuição de Depressão do Segmento ST')
              plt.show()

            elif 'DepressaoSegSTExiste' in df.columns: 
                plt.figure(figsize=(8, 4))
                sns.countplot(data=df, x='DepressaoSegSTExiste')
                plt.title('Distribuição por DepressaoSegSTExiste')
                plt.show()
            
        elif choice == '6':
            # DoencaCardiaca
            plt.figure(figsize=(8, 4))
            sns.countplot(data=df, x='DoencaCardiaca')
            plt.title('Distribuição de Doença Cardíaca')
            plt.show()

        elif choice == '7' and 'Colesterol' in df.columns:
            # colesterol MNAR
            df['chol_zero'] = df['Colesterol'] == 0

            resultado = df.groupby('chol_zero')['DoencaCardiaca'].value_counts(normalize=True).unstack()
            print("Distribuição de doença com base na ausência de colesterol:")
            print(resultado)

            # Convert the hue values to strings explicitly
            plot_df = df.copy()
            plot_df['DoencaCardiaca'] = plot_df['DoencaCardiaca'].map({0: 'Não', 1: 'Sim'})
            
            sns.countplot(x='chol_zero', hue='DoencaCardiaca', data=plot_df)
            plt.title("Doença vs Ausência de Colesterol")
            plt.xlabel("Colesterol ausente?")
            plt.ylabel("Contagem")
            plt.xticks([0, 1], ['Não', 'Sim'])
            plt.legend(title='Doença')
            plt.show()

        elif choice == '8' and 'InfradesnivelamentoSegST' in df.columns:
            # colesterol MNAR
            df['St_slope_zero'] = df['InfradesnivelamentoSegST'] == 0

            resultado = df.groupby('St_slope_zero')['DoencaCardiaca'].value_counts(normalize=True).unstack()
            print("Distribuição de doença com base na ausência de InfradesnivelamentoSegST:")
            print(resultado)

            # Convert the hue values to strings explicitly
            plot_df = df.copy()
            plot_df['DoencaCardiaca'] = plot_df['DoencaCardiaca'].map({0: 'Não', 1: 'Sim'})
            
            sns.countplot(x='St_slope_zero', hue='DoencaCardiaca', data=plot_df)
            plt.title("Doença vs Ausência de InfradesnivelamentoSegST")
            plt.xlabel("InfradesnivelamentoSegST ausente?")
            plt.ylabel("Contagem")
            plt.xticks([0, 1], ['Não', 'Sim'])
            plt.legend(title='Doença')
            plt.show()
        
        elif choice == '9':
            break
            
        else:
            print("Opção inválida. Tente novamente.")

