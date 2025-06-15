import os
import matplotlib.pyplot as plt

def outliers_menu(df):
    """Identificação e análise de outliers"""
    num_cols = ['Idade', 'PressaoArterialRepouso', 'FrequenciaCardiacaMaxima']
    if 'Colesterol' in df.columns:
        num_cols.append("Colesterol")
    if 'DepressaoSegST' in df.columns:
        num_cols.append("DepressaoSegST")

    tex_cols = ['InfradesnivelamentoSegST']


    if 'ColesterolPresente' in df.columns:
        tex_cols.append("ColesterolPresente")
    if 'DepressaoSegStExiste' in df.columns:
        tex_cols.append("DepressaoSegStExiste")

    if 'Sexo' in df.columns:
        tex_cols.append("Sexo")
    if 'Sexo_Masculino' in df.columns:
        tex_cols.append("Sexo_Masculino")

    if 'TipoDorToracica' in df.columns:
        tex_cols.append("TipoDorToracica")
    if 'TipoDorToracica_Angina' in df.columns:
        tex_cols.append("TipoDorToracica_Angina")
    if 'TipoDorToracica_Assintomático' in df.columns:
        tex_cols.append("TipoDorToracica_Assintomático")

    if 'GlicemiaJejum' in df.columns:
        tex_cols.append("GlicemiaJejum")
    if 'GlicemiaJejum_1.0' in df.columns:
        tex_cols.append("GlicemiaJejum_1.0")

    if 'DorAngina' in df.columns:
        tex_cols.append("DorAngina")

    if 'ECGRepouso_Hipertrofia Ventricular Esquerda ' in df.columns:
        tex_cols.append("ECGRepouso_Hipertrofia Ventricular Esquerda ")
    if 'ECGRepouso_Normal' in df.columns:
        tex_cols.append("ECGRepouso_Normal")

    # cat_cols = [col for col in df.columns if col not in num_cols and df[col].dtype == 'object']
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clears terminal
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
            # if not cat_cols:
            #     print("\nNão há colunas textuais no dataframe.")
            #     return
                
            print("\nValores únicos em colunas textuais:")
            for col in tex_cols:
                unique_vals = df[col].unique()
                print(f"\nColuna '{col}':")
                print(unique_vals)

            input("\n\nPressione Enter para continuar...")
            
        elif opcao == '3':
            break
                
        else:
            print("Opção inválida. Por favor escolha 1 ou 2.")
 