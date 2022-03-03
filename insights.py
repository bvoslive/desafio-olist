# -----> IMPORTANDO BIBLIOTECAS <-----
import pandas as pd
from unidecode import unidecode
import matplotlib.pyplot as plt
from tqdm import tqdm

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# -----> TFIDF COM GEOLOCALIZAÇÃO <-----

# MOSTRAR NO MAPA O QUE ESTÃO FALANDO POR REVIEW

df_geo = pd.read_csv('./datasets/raw/olist_geolocation_dataset.csv')
reviews = pd.read_csv('./datasets/processed/tratamento_stopwords.csv')
reviews.rename(columns={'Unnamed: 0': 'INDEX_REFERENCIA'}, inplace=True)
df_reviews = pd.read_csv('./datasets/raw/olist_order_reviews_dataset.csv', dtype='str')


df_orders = pd.read_csv('./datasets/raw/olist_orders_dataset.csv')





df_grouped = pd.read_csv('./datasets/processed/data_grouped.csv')
df_grouped = df_grouped[['customer_id', 'geolocation_lat', 'geolocation_lng']]

df_grouped['customer_id'].unique()


customer_reviews = df_orders.merge(df_grouped, on='customer_id', how='left')
customer_reviews = customer_reviews[['order_id', 'geolocation_lat', 'geolocation_lng']]



df_reviews = df_reviews.loc[reviews['INDEX_REFERENCIA'].tolist()]
df_reviews = df_reviews[['order_id']]
df_reviews['REVIEW'] = reviews['DOCS'].tolist()
df_reviews = df_reviews[['order_id', 'REVIEW']]

customer_reviews = customer_reviews.merge(df_reviews, on='order_id', how='right')
customer_reviews.dropna(inplace=True)


#LIMPEZA EXTRA
customer_reviews['REVIEW'] = customer_reviews['REVIEW'].apply(lambda x: ' '.join(pd.Series(x.split(' ')).unique().tolist()))
customer_reviews['REVIEW'] = [unidecode(review) for review in customer_reviews['REVIEW']]













lat_long = customer_reviews[['geolocation_lat', 'geolocation_lng']]
sc = StandardScaler()
X = sc.fit_transform(lat_long)

# -----> KMEANS <-----

# NÚMERO DE ZONAS ATÉ A QUANTIDADE DE BAIRROS EM SÃO PAULO
distortions = []
K = range(5,40)
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

