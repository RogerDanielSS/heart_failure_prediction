from exploratory_anal import exploratory_analysis_menu
from training import training_menu
from loader import set_dataset
from preprocessing import preprocessing_menu

def main():
    """Função principal"""
    df = set_dataset()

    while True:
        print("\n=== Menu Principal ===")
        print("1 - Análise exploratória")
        print("2 - Pré-processamento")
        # print("3 - Treinamento")
        print("3 - Sair")
        
        choice = input("Escolha uma opção: ")
        
        if choice == '1':
            exploratory_analysis_menu(df)
        elif choice == '2':
            preprocessing_menu(df)
        # elif choice == '3':
        #     training_menu(df)
        elif choice == '3':
            print("Encerrando o programa...")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()