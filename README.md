# heart_failure_prediction

## 1 Configuração de ambiente com conda

### 1.1 instalar conda com apt
    sudo apt update

    sudo apt install anaconda

### 1.2 Criar e iniciar ambiente

    conda create --name ml python=3.9 numpy pandas matplotlib seaborn

    conda activate ml

### 1.3 Rodar programa

    python menu.py

## 2 Análise exploratória

### 2.1 Traduções das colunas

#### 2.1.1 Traduções diretas
Traduções simples, sem necessidade de busca por contexto

- age -> idade
- sex -> sexo
- ChestPainType -> Tipo de dor toráxica
- Cholesterol -> Colesterol
- HeartDisease -> DoencaCardiaca

  
#### 2.1.2 Abreviações
Traduções mais elaboradas, foi necessário buscar o significado das abreviações para então fazer traduções diretas

- FastingBS -> fasting blood sugar -> Glicemia em jejum
- RestingBP -> resting blood presure -> Pressão sanguínia em repouse
- RestingECG -> resting electrocardiogram -> Eletrocardiograma em repouse
- maxHR -> max heart rate -> Frequência cardíaca máxima

#### 2.1.3 Termos de domínio
Termos da área de saúde, precisam de uma busca maior pra entender do que se trata, mesmo traduzido

- ExerciseAngina: Angina é uma dor no peito temporária ou uma sensação de pressão que ocorre quando o músculo cardíaco não está recebendo oxigênio suficiente. Traduzido para "DorAngina"
- Oldpeak: Depressão do segmento ST induzida pelo exercício em relação ao repouso ('ST' refere-se às posições no gráfico de ECG. Ver mais [aqui](https://litfl.com/st-segment-ecg-library/)). Traduzido para "DepressaoSegST"
- ST_slope: infradesnivelamento do segmento ST (caracterizado pela depressão desse segmento abaixo da linha de base no eletrocardiograma (ECG)). Traduzido para "InfradesnivelamentoSegST"
