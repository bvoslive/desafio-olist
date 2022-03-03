"""

reviews representam uma informação a mais
Num mundo tão acelerado que vivemos hoje
Dedicar seu tempo para fazer um elogio ou uma reclamação
Representa um peso considerável a respeito da venda


Quando não há comnetários
Nesse quesito, os dados nulos podem significar algo

Sem pontuação dos produtos, as pessoas não ficam confiantes

Medindo qualidade da pessoa que fez o review (Ele é apenas um hater? Está lá apenas para reclamar?)
quantidade de compras
quantidade reclamações positivas e negativas
ele costuma ranquear os produtos que compra?

Medindo confiança no vendedor
ranqueamento e reviews

Devido as emoções humanas serem diversas
não fiz a Análise de Sentimento positiva ou negativa
mas sim um compilado de emoções humanas
* Insulto
* ameaça
* obsceno
* descritivo do produto
* review equivocado
* elogio
    * entrega (tempo)
    * vendedor
    * preço
    * produto

A conquista do primeiro cliente
O cliente comprou pela primeira vez e nunca mais comprou

Churn, consumia bastante, depois parou de consumir
o que esse cliente disse?

quando um vendedor faz uma publicação
a partir dos comentários
como a venda se sucedeu depois?

PROBLEMAS:
A partir de que momento o atraso se torna um problema?
TF-IDF

Bigram
Onde demora mais para ser entregue a ponto que isso se torne um problema
Qual a faixa de preço mais atraente dependendo da região

Estimativa de tempo de entrega

Em que região está ocorrendo mais atraso

NLP e Time Series

"""


# IMPORTANDO BIBLIOTECAS
from cgitb import enable
import numpy as np
import pandas as pd
import os
import glob
import io
import string
import spacy
from tqdm import tqdm

from unidecode import unidecode


import matplotlib.pyplot as plt
import plotly
import plotly.express as px

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import CountVectorizer

componentes_excluir = ['ner', 'textcat', 'lemmatizer', 'senter', 'sentencizer', 'tagger', 'entity_linker', 'parser', 'entity_ruler', 'textcat_multilabel']
nlp = spacy.load("pt_core_news_sm", exclude=componentes_excluir)


# EXPORTANDO FEATURES DE TODOS OS DATASETS
datasets_arquivos = glob.glob('./datasets/raw/*.csv')
for dataset_arquivo in datasets_arquivos:

    df = pd.read_csv(dataset_arquivo)
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    descricao_acc = dataset_arquivo + '\n' + s + '\n\n'

    with open("datasets_features.txt", "a",
            encoding="utf-8") as f:  
        f.write(descricao_acc)
    f.close()
    


# -----> EXPLORANDO REVIEWS <-----

df_textos = pd.read_csv('./datasets/raw/olist_order_reviews_dataset.csv', encoding='utf-8')
df_textos.dropna(subset=['review_comment_message'], inplace=True)
review_comment_message = df_textos['review_comment_message']


# ELIMINANDO PONTUAÇÕES E BARRAS DE ESPAÇO
pontuacoes = string.punctuation

review_comment_message = review_comment_message.apply(lambda x: ''.join([letra for letra in x if letra not in pontuacoes]))

#ELIMINANDO BARRAS DE ESPAÇO LATERAIS
review_comment_message = review_comment_message.str.strip()
review_comment_message = review_comment_message.str.replace('\n', ' ')
review_comment_message = review_comment_message.str.lower()


# ELIMINANDO STOPWORDS
def tipo_tokens_permitidos(token: str, tipos_permitidos = ['PROPN', 'NOUN', 'VERB', 'ADJ', 'INTJ', 'ADV']):

    apenas_barras = [True if letra == ' ' else False for letra in token]
    if all(apenas_barras):
        return False

    token = token.strip()
    palavra_confere = nlp(token)[0].pos_  

    for tipo_permitido in tipos_permitidos:
        if palavra_confere == tipo_permitido:
            #ENCONTROU O TIPO PERMITIDO
            return True
    #NÃO ENCONTROU O TPO PERMITIDO
    return False


def elimina_stopwords(doc):

    doc = [token for token in doc.split(' ') if token != '']
    doc = [token for token in doc if tipo_tokens_permitidos(token)]
    doc = ' '.join(doc)

    return doc

review_comment_message_list = review_comment_message.tolist()

lista_stopwords_eliminados = []
for review in tqdm(review_comment_message_list):
    lista_stopwords_eliminados.append(elimina_stopwords(review))

df_lista_stopwords_eliminados = pd.DataFrame({'DOCS': lista_stopwords_eliminados})
df_lista_stopwords_eliminados.index = review_comment_message.index.tolist()
df_lista_stopwords_eliminados.to_csv('tratamento_stopwords.csv')

# -----> TFIDF COM GEOLOCALIZAÇÃO <-----

# MOSTRAR NO MAPA O QUE ESTÃO FALANDO POR REVIEW

np.random.seed(42)

df_geo = pd.read_csv('./datasets/raw/olist_geolocation_dataset.csv')
reviews = pd.read_csv('./datasets/processed/tratamento_stopwords.csv')
df_reviews = pd.read_csv('./datasets/raw/olist_order_reviews_dataset.csv', dtype='str')

#LIMPANDO DADOS
reviews.rename(columns={'Unnamed: 0': 'INDEX_ORIGINAL'}, inplace=True)
reviews.set_index('INDEX_ORIGINAL', inplace=True)
reviews = reviews['DOCS'].str.lower()
reviews.dropna(inplace=True)

#FILTRANDO REVIEWS
reviews_index = reviews.index.tolist()

#LIMPKPEZA 
reviews = reviews.apply(lambda x: ' '.join(pd.Series(x.split(' ')).unique().tolist()))
reviews = [unidecode(review) for review in reviews]


df_reviews = df_reviews.loc[reviews_index]


#++++++++++++++++ aqui ta errado
df_geo = df_geo.iloc[reviews_index]
df_geo['geolocation_city'].unique()

lat_long = df_geo[['geolocation_lat', 'geolocation_lng']]
sc = StandardScaler()
X = sc.fit_transform(lat_long)

#DBSCAN

model = DBSCAN().fit(lat_long)
labels = model.labels_
pd.Series(list(labels)).unique()
pd.Series(model.labels_).value_counts().loc[-1]
lat_long.iloc[0]

# KMEANS
from sklearn.cluster import KMeans

# NÚMERO DE ZONAS ATÉ A QUANTIDADE DE BAIRROS EM SÃO PAULO
distortions = []
K = range(5,95)
for k in tqdm(K):
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(distortions, 'bx-')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


kmeanModel = KMeans(n_clusters=15)
kmeanModel.fit(X)

coordenadas_ajustadas = kmeanModel.cluster_centers_
centroides = sc.inverse_transform(coordenadas_ajustadas)


# TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf(reviews):

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(reviews)

    termos = vectorizer.get_feature_names()

    pontuacao = pd.DataFrame(X.sum(axis=0))
    pontuacao = pontuacao.transpose()
    pontuacao.columns = ['TFIDF']
    pontuacao = pontuacao['TFIDF'].tolist()

    tfidf_analise = pd.DataFrame({'termo': termos, 'tfidf': pontuacao})
    tfidf_analise = tfidf_analise.sort_values('tfidf', ascending=False).iloc[:20]

    return tfidf_analise

# TFIDF COM PONTUAÇÃO

index_nota = df_reviews[df_reviews['review_score'] == '2'].index.tolist()
reviews_nota = reviews.loc[index_nota]

# BIGRAM COM PONTUAÇÃO

vectorizer = CountVectorizer(ngram_range=(2, 3))
X = vectorizer.fit_transform(reviews_nota)
bigram = vectorizer.get_feature_names_out()
pontuacao = X.sum(axis=0).tolist()
pontuacao = pontuacao[0]

df_bigram = pd.DataFrame({'BIGRAM': bigram, 'PONTUACAO': pontuacao})
df_bigram.sort_values('PONTUACAO', ascending=False, inplace=True)



# -----> EM QUE MOMENTO A ENTREGA NO ATRASO PASSA A SER UM PROBLEMA <-----

df_orders = pd.read_csv('./datasets/raw/olist_orders_dataset.csv')

df_order_reviews = df_reviews.merge(df_orders, on='order_id', how='left')

df_order_reviews['order_delivered_customer_date'] = pd.to_datetime(df_order_reviews['order_delivered_customer_date'])
df_order_reviews['order_purchase_timestamp'] = pd.to_datetime(df_order_reviews['order_purchase_timestamp'])
df_order_reviews['order_estimated_delivery_date'] = pd.to_datetime(df_order_reviews['order_estimated_delivery_date'])

df_order_reviews_delivered_dropna = df_order_reviews.dropna(subset=['order_delivered_customer_date'])

#RELACIONANDO COMPRA E ENTREGA
tempo_compra_entrega = df_order_reviews_delivered_dropna['order_delivered_customer_date'] - df_order_reviews_delivered_dropna['order_purchase_timestamp']
dias_entrega = tempo_compra_entrega.apply(lambda x: x.days)


#RELACIONANDO TEMPO ESTIMADO
tempo_compra_entrega = df_order_reviews_delivered_dropna['order_delivered_customer_date'] - df_order_reviews_delivered_dropna['order_estimated_delivery_date']
dias_estimado = tempo_compra_entrega.apply(lambda x: x.days)


review_score = df_order_reviews_delivered_dropna['review_score']
review_score = review_score.astype('int')

dias_entrega = dias_entrega.tolist()
review_score = review_score.tolist()


#DIAS ENTREGA
np.corrcoef(dias_entrega, review_score)

#DIAS ESTIMADO
np.corrcoef(dias_estimado, review_score)










# -----> MERGE DATAFRAMES <-----



df_geo = pd.read_csv('./datasets/raw/olist_geolocation_dataset.csv')
df_customer = pd.read_csv('./datasets/raw/olist_customers_dataset.csv')
df_orders = pd.read_csv('./datasets/raw/olist_orders_dataset.csv')
df_items = pd.read_csv('./datasets/raw/olist_order_items_dataset.csv')
df_reviews = pd.read_csv('./datasets/raw/olist_order_reviews_dataset.csv')

df_reviews = df_reviews[['order_id', 'review_score']]

df_geo.drop_duplicates(subset=['geolocation_zip_code_prefix'], inplace=True)

df_geo.rename(columns={'geolocation_zip_code_prefix': 'zip_code_prefix'}, inplace=True)
df_customer.rename(columns={'customer_zip_code_prefix': 'zip_code_prefix'}, inplace=True)

df_merged = df_geo.merge(df_customer, on='zip_code_prefix', how='right')
df_merged = df_merged[['geolocation_lat', 'geolocation_lng', 'customer_id']]

df_merged.drop_duplicates(inplace=True)


df_merged = df_merged.merge(df_orders, on='customer_id', how='right')
df_merged = df_merged.merge(df_items, on='order_id', how='right')
df_merged = df_merged.merge(df_reviews, on='order_id', how='right')



df_merged.info()

df_merged.dropna(subset=['order_delivered_customer_date'], inplace=True)
df_merged



df_merged['order_delivered_customer_date'] = pd.to_datetime(df_merged['order_delivered_customer_date'])
df_merged['order_purchase_timestamp'] = pd.to_datetime(df_merged['order_purchase_timestamp'])
df_merged['order_estimated_delivery_date'] = pd.to_datetime(df_merged['order_estimated_delivery_date'])



df_merged['tempo_entrega'] = df_merged['order_delivered_customer_date'] - df_merged['order_purchase_timestamp']




df_merged.iloc[0]

df_merged

df_merged.iloc[0]


df_orders

df_orders.iloc[0]



df_merged