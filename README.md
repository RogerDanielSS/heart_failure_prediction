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
- RestingBP -> resting blood presure -> Pressão arterial em repouso
- RestingECG -> resting electrocardiogram -> Eletrocardiograma em repouso
- maxHR -> max heart rate -> Frequência cardíaca máxima

#### 2.1.3 Termos de domínio
Termos da área de saúde, precisam de uma busca maior pra entender do que se trata, mesmo traduzido

- ExerciseAngina: Angina é uma dor no peito temporária ou uma sensação de pressão que ocorre quando o músculo cardíaco não está recebendo oxigênio suficiente. Traduzido para "DorAngina"
- Oldpeak: Depressão do segmento ST induzida pelo exercício em relação ao repouso ('ST' refere-se às posições no gráfico de ECG. Ver mais [aqui](https://litfl.com/st-segment-ecg-library/)). Traduzido para "DepressaoSegST"
- ST_slope: infradesnivelamento do segmento ST (caracterizado pela depressão desse segmento abaixo da linha de base no eletrocardiograma (ECG)). Traduzido para "InfradesnivelamentoSegST"

### 2.2 Identificação de outliers

Foram utilizadas duas estratégias para identificação de outliers.. Para 

#### 2.2.1 Variáveis escritas
Para variáveis escritas, foram imprimidas no terminal todos os valores possíveis. O objetivo disso é identificar valores que "não deveriam estar ali". Não foi encontrado nenhum outlier aqui.

![Variáveis escritas para identificação de outliers](/assets/Variaveis%20escritas.png)

#### 2.2.2 Variáveis numéricas
Para variáveis numéricas, foi criado um gráfico do tipo box plot com todas as variáveis numéricas.

Num boxplot, os dados são representados de modo que mostre:

- Mediana (valor central)

- Quartis (dispersão dos dados)

- Possíveis outliers (valores extremos)

![Boxplot para identificação de outliers](/assets/Boxplot.png)

Deste modo, verifica-se a possibilidade de outliers em: Colesterol PressaoArterialRepouso, FrequenciaCardiacaMaxima, DepressaoSegST

##### 2.2.2.1 Colesterol

Visualização da distribuição dos dados

![Distribuição de colesterol](/assets/dist_colesterol.png)

Observando tanto o gráfico de barras, quanto o boxplot, é possível verificar valores zerados ou extremamente altos.


###### 2.2.2.1.1 Valores zerados
interpretamos que não são valores possíveis, por tanto representam ausência de dados ou não preenchimento. Os preenchimentos possíveis são dois: Excluir as linhas que não possuem esses dados ou substituir os valores zarados pela média ou mediana dos valores da coluna.

###### 2.2.2.1.2 Valores acima de 400
Segundo [essa matéria do telemedicida Morsch](https://telemedicinamorsch.com.br/blog/colesterol-alto?srsltid=AfmBOoo1VX7_ERXKow-IqOWyFhwVE9fQU6aA82t8tDxyav6zodjMflQ3), o colestetol total é considerado saudável quando abaixo de 190. 

No entanto, a própria matéria cita casos de colesterol acima de 500. Assim sendo, esses valores não devem ser interpretados como irreais ou erros de preenchimento.

Porém esses valores extramente altos podem causar overfitting, então é necessário que façamos a normalização desses valores.


##### 2.2.2.2 Pressão sanguínea 

Para a pressão sanguínea, também é possível observar alguns valores zerados e alguns valores acima da média. Mas primeiro, para entender os dados, é preciso saber de que tipo de colesterol se trata. [Essa matéria do telemedicina Morsch](https://telemedicinamorsch.com.br/blog/tabela-de-pressao-arterial?srsltid=AfmBOooJWKinr68IJn031KjYffJNxsqZsAfGmvL5t24zitmYtQsIlraX) mostra valores entre 57 e 110 mmHg para pressão diatólica e valores entre 103 e 180 mmHg para pressão sistólica. Fazendo um exercício de bom senso, chega-se á conclusão de que os dados no dataset se tratam da pressão sistólica.

![Distribuição de colesterol](/assets/dist_pressao.png)


Agora analisando os possíveis outliers

###### 2.2.2.2.1 Valores zerados

Zero está bem distante dos valores do intervalo mostrado [nessa matéria mesma do telemedicina Morsch](https://telemedicinamorsch.com.br/blog/tabela-de-pressao-arterial?srsltid=AfmBOooJWKinr68IJn031KjYffJNxsqZsAfGmvL5t24zitmYtQsIlraX). Chegamos à conclusão de que são erros de preenchimento ou simplesmente dados não preenchidos. Tal qual para a coluna colesterol, há dois tratamentos possíveis: Excluir as linhas que não possuem esses dados ou substituir os valores zarados pela média ou mediana dos valores da coluna.


###### 2.2.2.2.2 Valores acima de 170

Aplicando a mesma lógica, os valores mostrados na distribuição de dados não estão distantes do intervalo mostrados [nessa matéria mesma do telemedicina Morsch](https://telemedicinamorsch.com.br/blog/tabela-de-pressao-arterial?srsltid=AfmBOooJWKinr68IJn031KjYffJNxsqZsAfGmvL5t24zitmYtQsIlraX). No entanto, ainda podem causar overfitting. De novo, tal qual para a coluna colesterol, é necessário que se faça a normalização.


##### 2.2.2.3 Frequência cardíaca

![Distribuição de frequência cardíaca](/assets/dist_freq_car.png)