# heart_failure_prediction

Usando conda para configurar o ambiente

conda create --name ml python=3.9 numpy pandas matplotlib seaborn

colunas:
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

      'Idade',
      'Sexo',
      'TipoDorToracica',
      'PressaoArterialRepouso',
      'Colesterol',
      'GlicemiaJejum',
      'ECGRepouso',
      'FrequenciaCardiacaMaxima',
      'DorAngina',
      'DepressaoSegST',
      'InfradesnivelamentoSegST',
      'DoencaCardiaca'


    # traduções diretas
    # age -> idade
    # sex -> sexo
    # ChestPainType -> Tipo de dor toráxica
    # Cholesterol -> Colesterol
    
    # abreviações
    # FastingBS -> fasting blood sugar -> Glicemia em jejum
    # RestingBP -> resting blood presure -> Pressão sanguínia em repouse
    # RestingECG -> resting electrocardiogram -> Eletrocardiograma em repouse
    # maxHR -> max heart rate -> Frequência cardíaca máxima

    # termos de domínio
    # ExerciseAngina -> Angina é uma dor no peito temporária ou uma sensação de pressão que ocorre quando o músculo cardíaco não está recebendo oxigênio suficiente
    # Oldpeak -> Depressão do segmento ST induzida pelo exercício em relação ao repouso ('ST' refere-se às posições no gráfico de ECG. Ver mais em <https://litfl.com/st-segment-ecg-library/>)
    # ST_slope ->  infradesnivelamento do segmento ST (caracterizado pela depressão desse segmento abaixo da linha de base no eletrocardiograma (ECG))
