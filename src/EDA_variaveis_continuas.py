from unidecode import unidecode
import pandas as pd
import numpy as np

# -----> IMPORTANDO E FILTRANDO DADOS <-----

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

#LIMPEZA 
reviews = reviews.apply(lambda x: ' '.join(pd.Series(x.split(' ')).unique().tolist()))
reviews = [unidecode(review) for review in reviews]
df_reviews = df_reviews.loc[reviews_index]

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
# possui maior correlação

#DIAS ESTIMADO
np.corrcoef(dias_estimado, review_score)