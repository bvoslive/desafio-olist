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

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

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

# -----> KMEANS <-----

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


df_bigram








