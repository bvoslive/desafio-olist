"""

reviews representam uma informação a mais
Num mundo tão acelerado que vivemos hoje
Dedicar seu tempo para fazer um elogio ou uma reclamação
Representa um peso considerável a respeito da venda


A partir de que momento o atraso se torna um problema?

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


"""


# IMPORTANDO BIBLIOTECAS
import pandas as pd
import os
import glob
import io

# EXPORTANDO FEATURES DE TODOS OS DATASETS
datasets_arquivos = glob.glob('./datasets/*.csv')
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
    




#EXPLORANDO TEXTOS
df_textos = pd.read_csv('./datasets/olist_order_reviews_dataset.csv')


df_textos.dropna(subset=['review_comment_message'], inplace=True)
df_textos.iloc[300]['review_comment_message']
