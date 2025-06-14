# heart_failure_prediction

## 1 Configuração de ambiente com conda

### 1.1 instalar conda com apt
    sudo apt update

    sudo apt install anaconda

### 1.2 Criar e iniciar ambiente

    conda create --name ml python=3.9 numpy pandas matplotlib seaborn scikit-learn

    conda activate ml

### 1.3 Rodar programa

    python app.py

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

Os valores mostrados como possíveis outliers estão no limite inferior. Vamos iniciar a análise somente o mínimo de 60 bpm. 

Valores tão baixos parecem estranhos para frequência máxima de um indivíduo. [Esse artigo no Scielo](https://www.scielo.br/j/abc/a/3SG3HkTkTZmg6NtrDfXmZYq/) obteve uma média global de 181,0 bpm ± 14,0bpm. Enquanto isso, [essa matéria do portal Tua Saúde](https://www.tuasaude.com/frequencia-cardiaca/) mostra entre valores 56 e 80 bpm para mulheres em repouso e valores entre 61 e 84 para homens em repouso. 

Isso levanta a suspeita que se trata de um erro na alimentação do dataset, por se tratar de valores muito distantes do esperado para indivíduos saudáveis. No entanto, após visualizar o gráfico de correlação entre frequência cardíaca máxima e doença cardíaca, chegamos à conclusão de que não se trata de erros na alimentação do dataset.


![Correlação entre frequência cardíaca máxima e doença cardíaca](/assets/correlacao_freq_max_doeca_card.png)

##### 2.2.2.4 Depressão segmento ST


![Distribuição de depressão no segmento ST](/assets/dist_dep_seg_st.png)

Segundo [essa matéria do portal Cardiovascular Medicine](https://ecgwaves.com/st-segment-normal-abnormal-depression-elevation-causes/), são considerados valores normais até 0.5mm de depressão no segmento ST, "porque indivíduos saudáveis ​​raramente apresentam depressão nesse segmento" (tradução livre). 

Neste caso, fazemos uma análise parecida com a de frequência cardíaca: são valores muito distantes do que aparentemente é considerado normal para indivíduos saudáveis, então existe alguma possibilidade de erro de preenchimento. Como o gráfico de correlação mostra uma correlação forte entre os valores tidos como anormais e a ocorrência de doença cardíaca, concluímos que não se trata de um erro de preenchimento.


![Correlação de depressão no segmento ST com doença cardíaca](/assets/correlacao_dist_seg_st_doeca.png)

## 3    Pre - Processamento

#### 3.1.1 Analises

##### 3.1.1.1 Analise de Colesterol
Ao fazer a analise de distribuição de variaveis em Colesterol se percebeu a presença de um grande número de valores zerados, o que levou ao questionamento do que fazer com essa coluna. Com o objetivo de descobrir se esses valores tinham algum significado importante foi decidido a utilização da analise de números ausentes MNAR (Missing Not at Random) onde foi averiguado que 89% dos valores zerados tinham problema cardiaco. Por meio disso percebemos que não seria benefico para a melhor acuracia do ML(Machine Learning) a exclusão desses dados ou da coluna já que eles eram uma variavel valiosa para a predição da presença de doenças.

![Correlação da ausencia de valores em colesterol com doençcas cardiacas](/assets/correlacao_entre_doenca_e_colesterol_0.png)

Por causa dessa conexão entre a presença de zeros e a presença de doenças cardiacas resolvemos focar o predicamento envolvendo colesterol nela. Por isso, criamos uma (flag) binaria que evidencia a presença do 0 na coluna e removemos os outros valores de colesterol os substituindo pelos da flag binaria., ja que os mesmos como pode se observar na analise 2.2.2.1 não são interessantes para o aprendizado do modelo.

##### 3.1.1.2 Analise de Depressão SegST
Similarmente com a analise de colesterol percebemos que ha um grande número de 0 na coluna, assim utilizamos novamento a tecnica de MNAR para averiguar a importancia desses valores e percebemos que 66% dos dados zerados não possuem doença cardiaca, tornando-os muito valiosos para o aprendizado da nossa maquina. Por causa disso resolvemos criar uma flag binario que evidencia a presença do 0 na coluna e removemos os outros valores de Depressão SegST os substituindo pelos da bandeira binaria.

![Correlação da ausencia de valores em Depressão SegST com doençcas cardiacas](/assets/correlacao_entre_doenca_e_SegSt_0.png)

##### 3.1.1.3 Analise de Idade
No caso de idade a presença de inumeros valores unicos é prejudicial para o aprendizado da maquina já que eles causam uma menor capacidade de previsão. Para evitar esse prejuijo resolvemos agrupar esses valores em cinco grupos de dezenas, indo de jovem adulto(20-29) até os idosos (70+). Esse agrupamento vacilita no aprendizado da maquina já que elimina a existencia desses valores unicos aumentando a acuracia e previsibilidade do aprendizado.

![Distribuição de idade Por Agrupamento](/assets/distribuicao_de_idade_por_agrupamento.png)

#### 3.1.2 Normatização dos dados
Após verificar as informaçoes dos dados temos que as colunas do banco de dados verificamos que elas são divididas em 2 tipos de dados os númericos e os categoricos do tipo objeto.

##### 3.1.2.1 Dados Númericos
Para lidar com os dados númericos foi utilizado o RobustScaler da biblioteca Scikit-learn. O processo de normalização é muito usado no preprocessamento de muitas maquinas de apredizados que focam em previsões, esse processo é feito retirando a media e redimencionando para variação unitaria. Entretanto outliers podem influenciar esse calculo de forma negativa se utilizado tecnicas mais simples. Por causa disso que escolhemos usar o RobustScaler já que ele utilizad de IQR(intervalo quartil) e da mediana para o dimencionamentom gerando resultados melhores.

##### 3.1.2.2 Dados Categóricos
Ao analisar os dados categoricos percebemos que existem dois tipos de dados categoricos no Banco de Dados, os dados categoricos nominais e os ordinais.

###### 3.1.2.2.1 Dados Categóricos Nominais
Para lidar com os dados Categóricos Nominais foi utilizado o One-Hot Encoding do Scikit-learn que serve para transformar dados não numericos em um formato que os algoritmos de aprendizagem de maquina consigam entender colunas binárias. Variáveis nominais não tem uma ordem matemática e se fosse atribuido numeros a elas o modelo poderia interpretar esses valores como tendo uma hierarquia, assim a tecnica do One-Hot Encoding remove a suposição de que numeros maiores tenham mais peso, tratando cada categoria de forma independente.

###### 3.1.2.2.2 Dados Categóricos Ordinais
Diferentemente das categorias nominais, aqui ocorre uma hierarquia maatemática então a sua normalização não pode ocorrer com o uso do One-Hot Encoding. Assim a normalização não precisou de usos de bibliotecas especificas podendo ser feitas com bibliotecas naturais do python. Na coluna Infra desnivelamento SegST os valores ({'Down': 0, 'Flat': 1, 'Up': 2}) foram mapeados manualmente e essas categorias foram transformados em numeros que preservão uma regra de magnitude 0 < 2.