# IMPORTANDO BIBLIOTECAS
import numpy as np
import pandas as pd
import glob
import io
import string
import spacy
from tqdm import tqdm

from utils import *

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

# ELIMNANDO PONTUAÇÕES E BARRAS DE ESPAÇO
pontuacoes = string.punctuation

# ELIMINANDO PONTUAÇÕES
review_comment_message = review_comment_message.apply(lambda x: ''.join([letra for letra in x if letra not in pontuacoes]))

#ELIMINANDO BARRAS DE ESPAÇO LATERAIS
review_comment_message = review_comment_message.str.strip()
review_comment_message = review_comment_message.str.replace('\n', ' ')
review_comment_message = review_comment_message.str.lower()
review_comment_message_list = review_comment_message.tolist()

#ELIMINANDO STOPWORDS
lista_stopwords_eliminados = []
for review in tqdm(review_comment_message_list):
    lista_stopwords_eliminados.append(elimina_stopwords(review))
df_lista_stopwords_eliminados = pd.DataFrame({'DOCS': lista_stopwords_eliminados})
df_lista_stopwords_eliminados.index = review_comment_message.index.tolist()
df_lista_stopwords_eliminados.to_csv('tratamento_stopwords.csv')
