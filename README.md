[![author](https://img.shields.io/badge/author-mathmeza-red.svg)](https://www.linkedin.com/in/matheus-meza-26956bb6) [![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-365/) [![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html) [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/carlosfab/data_science/issues)

# Projeto: Credit Card Fraud Detection
 ### _Detecção de fraude em cartão de crédito_

<p align="center">
  <img src="imagem-tc.jpeg" >
</p>

O pagamento com cartão de crédito obteve um aumento exponencial de utilização em decorrência de 3 principais pontos: 
- Tecnologia;
- Facilidade na obtenção;
- Maior variedade de empresas que disponibilizam esse método de pagamento em conjunto com empresas que aceitam o método de pagamento. 

Ao mesmo tempo, os fraudadores tendo essa visão de crescimento deste método de pagamento, começaram a impactar cada vez mais o mercado com a realização de fraude em pessoas inocentes. 
  
Devido este problema que tende a aumentar, é de suma importância que as empresas de cartão de crédito sejam capazes de reconhecer transações fraudulentas para que os clientes não sejam cobrados por itens que não compraram.
  
**A ideia deste projeto é trazer métodos de identificação de fraude em cartão de crédito com Machine Learning (Aprendizado de Máquina) com intuito de diminuir o nível de fraude.**

---

Para a obtenção dos dados, utilizamos o [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) pois trata-se de um site super rico, onde há diversos datasets para trabalharmos e tendo nosso problema que queremos resolver detalhado, aproveitamos o dataset de fraude nos cartões de crédito.

Em relação aos próximos passos, precisamos importar uma série de bibliotecas para trabalharmos neste nootebook. Seguem as bibliotecas:

```
# Importando bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# pré-processamento
from sklearn.preprocessing import StandardScaler 
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Modelagem
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from xgboost import XGBRFClassifier

# Validação
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

```
Segue abaixo a linha de código utilizada para obter esses dados:
```
base = pd.read_csv('creditcard.csv')
```

---

Após a extração, verificamos cerca de **284.807 linhas** e **31 colunas**. Ao analisar as colunas detalhadamente, tivemos:
- `Time` - Número de segundos decorridos entre a transação atual e a primeira transação no conjunto de dados;
- `Colunas de "V1" até "V28"` - São features (características) dos usuários nas quais são resultados de um processo de PCA (Redução de dimensionalidade) para proteger a identidade e a sensibilidade dos dados.
- `Amount` - Valor em USD da transação;
- `Class` - Identificador se foi realmente fraude ou não.


