Em aprendizado de máquina, outliers (ou valores atípicos) são pontos de dados que se desviam significativamente do restante da distribuição. Eles podem ser causados por erros de medição, variabilidade natural ou anomalias genuínas no fenômeno observado.
Características dos Outliers:

    Estão distantes da maioria dos dados.

    Podem afetar negativamente o desempenho do modelo.

    Nem sempre são erros—às vezes representam informações importantes (como fraudes em transações financeiras).

Impactos no Aprendizado de Máquina:

    Modelos Sensíveis a Outliers:

        Algoritmos baseados em distâncias (KNN, K-means) ou otimização por mínimos quadrados (Regressão Linear) podem ser prejudicados.

        Árvores de decisão e modelos baseados em ensembles (Random Forest, XGBoost) são geralmente mais robustos.

    Distorção de Métricas:

        Média e variância podem ser altamente influenciadas.

        Métricas como RMSE (Root Mean Squared Error) podem piorar devido a poucos outliers extremos.

Como Lidar com Outliers:

    Detecção:

        Métodos Estatísticos: Z-score (valores além de ±3 desvios padrão), IQR (Intervalo Interquartil: dados fora de *Q1 - 1,5×IQR* ou *Q3 + 1,5×IQR*).

        Visualização: Boxplots, histogramas ou scatter plots.

        Algoritmos: Isolation Forest, DBSCAN (para detecção automática).

    Tratamento:

        Remoção: Se forem claramente erros.

        Transformação: Aplicar log, sqrt ou winsorização (limitar valores extremos).

        Imputação: Substituir pela mediana ou média truncada.

        Modelos Robustos: Usar algoritmos menos sensíveis, como Regressão de Huber ou SVR (Support Vector Regression).