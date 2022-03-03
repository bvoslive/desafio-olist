# -----> IMPORTANDO BIBLIOTECAS <-----

import numpy as np
import pandas as pd

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

df_merged.dropna(subset=['order_delivered_customer_date'], inplace=True)

df_merged['order_delivered_customer_date'] = pd.to_datetime(df_merged['order_delivered_customer_date'])
df_merged['order_purchase_timestamp'] = pd.to_datetime(df_merged['order_purchase_timestamp'])
df_merged['order_estimated_delivery_date'] = pd.to_datetime(df_merged['order_estimated_delivery_date'])

df_merged['tempo_entrega'] = df_merged['order_delivered_customer_date'] - df_merged['order_purchase_timestamp']
df_merged['tempo_estimado_entrega'] = df_merged['order_estimated_delivery_date'] - df_merged['order_delivered_customer_date']

df_merged['tempo_entrega'] = df_merged['tempo_entrega'].apply(lambda x: x.days)
df_merged['tempo_estimado_entrega'] = df_merged['tempo_estimado_entrega'].apply(lambda x: x.days)

df_merged_filtro = df_merged[['customer_id', 'geolocation_lat', 'geolocation_lng', 'price', 'freight_value', 'review_score', 'tempo_entrega', 'tempo_estimado_entrega']]

df_merged_filtro_agrupado = \
df_merged_filtro.groupby(['customer_id', 'geolocation_lat', 'geolocation_lng']) \
                    .agg({'price': np.sum, 'freight_value':np.mean, 'review_score': np.mean, 'tempo_entrega': np.mean, 'tempo_estimado_entrega': np.mean})

df_merged_filtro_agrupado.reset_index(inplace=True)
df_merged_filtro_agrupado.to_csv('data_grouped.csv', index=False)