import pandas as pd

from exploratory_anal import exploratory_analysis_menu

def load_data(filepath):
    """Carrega o dataset"""
    return pd.read_csv(filepath)


def translate_dataset(df):
    """Traduz colunas e valores categóricos para português"""

    # Dicionário de tradução das colunas
    traducoes_colunas = {
      'Age': 'Idade',
      'Sex': 'Sexo',
      'ChestPainType': 'TipoDorToracica',
      'RestingBP': 'PressaoArterialRepouso',
      'Cholesterol': 'Colesterol',
      'FastingBS': 'GlicemiaJejum',
      'RestingECG': 'ECGRepouso',
      'MaxHR': 'FrequenciaCardiacaMaxima',
      'ExerciseAngina': 'DorAngina',
      'Oldpeak': 'DepressaoSegST',
      'ST_Slope': 'InfradesnivelamentoSegST',
      'HeartDisease': 'DoencaCardiaca'
    }
    
    # Renomear colunas
    df = df.rename(columns=traducoes_colunas)
    
    # Dicionários para traduzir valores categóricos
    traducoes_valores = {
        'Sexo': {'M': 'Masculino', 'F': 'Feminino'},
        'TipoDorToracica': {
            'ATA': 'Angina Típica',
            'NAP': 'Angina Atípica',
            'ASY': 'Assintomático',
            'TA': 'Angina Típica'  # Caso exista esta categoria
        },
        'ECGRepouso': {
            'Normal': 'Normal',
            'ST': 'Anormalidade onda ST-T',
            'LVH': 'Hipertrofia Ventricular Esquerda'
        },
        'DorAngina': {'Y': 'Sim', 'N': 'Não'},
        'InfradesnivelamentoSegST': {
            'Up': 'Ascendente',
            'Flat': 'Plano',
            'Down': 'Descendente'
        }
    }
    
    # Aplicar tradução aos valores categóricos
    for col, traducoes in traducoes_valores.items():
        if col in df.columns:
            df[col] = df[col].replace(traducoes)
    
    return df


def main():
    """Função principal"""
    filepath = './dataset/heart.csv'
    df = load_data(filepath)
    df =  translate_dataset(df)
    
    while True:
        print("\n=== Menu Principal ===")
        print("1 - Análise exploratória")
        print("2 - Sair")
        
        choice = input("Escolha uma opção: ")
        
        if choice == '1':
            exploratory_analysis_menu(df)
        elif choice == '2':
            print("Encerrando o programa...")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()