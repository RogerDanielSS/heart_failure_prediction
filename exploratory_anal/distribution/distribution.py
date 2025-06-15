import os
from .categorical import categorical_menu
from .numeric import numeric_menu
from .ordinal import ordinal_menu


def distributions_menu(df):
    """Análise das distribuições das variáveis com menu interativo"""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clears terminal
        print("\n=== Análise de Distribuições ===")
        print("Selecione o tipo de variável:")
        print("1 - Variáveis numéricas")
        print("2 - Variáveis categóricas")
        print("3 - Variáveis ordinais")
        print("4 - Voltar")
        choice = input("Escolha uma opção (1-4): ")

        if choice == '1':
            numeric_menu(df)
            
        elif choice == '2':
           categorical_menu(df)

        elif choice == '3':
           ordinal_menu(df)
        
        elif choice == '4':
            break
            
        else:
            print("Opção inválida. Tente novamente.")