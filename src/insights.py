# -----> IMPORTANDO BIBLIOTECAS <-----
import pandas as pd
from unidecode import unidecode
import matplotlib.pyplot as plt
from tqdm import tqdm

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from utils import *

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
df_reviews = df_reviews[['order_id', 'review_score']]
df_reviews['REVIEW'] = reviews['DOCS'].tolist()
df_reviews = df_reviews[['order_id', 'REVIEW', 'review_score']]

customer_reviews = customer_reviews.merge(df_reviews, on='order_id', how='right')
customer_reviews.dropna(inplace=True)

#LIMPEZA EXTRA
customer_reviews['REVIEW'] = customer_reviews['REVIEW'].apply(lambda x: ' '.join(pd.Series(x.split(' ')).unique().tolist()))
customer_reviews['REVIEW'] = [unidecode(review) for review in customer_reviews['REVIEW']]
customer_reviews.reset_index(drop=True, inplace=True)

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

lat_long = customer_reviews[['geolocation_lat', 'geolocation_lng']]
sc = StandardScaler()
X = sc.fit_transform(lat_long)

kmeanModel = KMeans(n_clusters=15)
kmeanModel.fit(X)

coordenadas_ajustadas = kmeanModel.cluster_centers_
centroides = sc.inverse_transform(coordenadas_ajustadas)

cat_labels_unicos = kmeanModel.labels_.tolist()
cat_labels = pd.Series(cat_labels_unicos).value_counts().sort_index()

lat_long_centroide = pd.DataFrame(centroides, columns=['LATITUDE', 'LONGITUDE'])
lat_long_centroide['QNT_REVIEWS'] = cat_labels

customer_reviews['CATEGORIA'] = cat_labels_unicos
customer_reviews['review_score'] = customer_reviews['review_score'].astype('int')
customer_reviews = customer_reviews[customer_reviews['review_score'] < 4]
customer_reviews['REVIEW'] = customer_reviews['REVIEW'].str.lower()
categorias_unicas = customer_reviews['CATEGORIA'].unique().tolist()

bigrams_seq = []
for categoria_unica in categorias_unicas:
    print(categoria_unica)
    docs = customer_reviews[customer_reviews['CATEGORIA'] == categoria_unica]['REVIEW']
    bigrams_seq.append(bigrams(docs))

lat_long_centroide['BI-GRAMS'] = bigrams_seq
lat_long_centroide.to_csv('mapa_bi_grams.csv', index=False)
