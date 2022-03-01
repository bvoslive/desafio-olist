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
    * entrega
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


"""


# IMPORTANDO BIBLIOTECAS
from cgitb import enable
import pandas as pd
import os
import glob
import io
import string
import spacy
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


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
df_textos.iloc[300]['review_comment_message']
review_comment_message = df_textos['review_comment_message']


# ELIMINANDO PONTUAÇÕES E BARRAS DE ESPAÇO
pontuacoes = string.punctuation

review_comment_message = review_comment_message.apply(lambda x: ''.join([letra for letra in x if letra not in pontuacoes]))

#ELIMINANDO BARRAS DE ESPAÇO LATERAIS
review_comment_message = review_comment_message.str.strip()

review_comment_message = review_comment_message.str.replace('\n', ' ')



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
df_geo = pd.read_csv('./datasets/raw/olist_geolocation_dataset.csv')
reviews = pd.read_csv('./datasets/processed/tratamento_stopwords.csv')

reviews.rename(columns={'Unnamed: 0': 'INDEX_ORIGINAL'}, inplace=True)
reviews.set_index('INDEX_ORIGINAL', inplace=True)

reviews_index = reviews.index.tolist()

df_geo = df_geo.iloc[reviews_index]

df_geo.iloc[10]