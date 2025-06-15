import os

def basic_info_menu(df):
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clears terminal
        print("\n=== Análise Exploratória ===")
        print("1 - Informações sobre as colunas")
        print("2 - Descrição dos dados")
        print("3 - Voltar")
        
        choice = input("Escolha uma opção: ")
        
        if choice == '1':
            os.system('cls' if os.name == 'nt' else 'clear')  # Clears terminal
            print(df.info())
            input("\n\nPressione Enter para continuar...")
        elif choice == '2':
            os.system('cls' if os.name == 'nt' else 'clear')  # Clears terminal
            print(df.describe(include='all'))
            input("\n\nPressione Enter para continuar...")
        elif choice == '3':
            break
        else:
            print("Opção inválida. Tente novamente.")

    print("\n=== Primeiras linhas do dataset ===")
    print(df.head())
    
    print("\n=== Informações do dataset ===")
    print(df.info())
    
    print("\n=== Estatísticas descritivas ===")
    print(df.describe(include='all'))
