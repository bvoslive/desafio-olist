from typing import List
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
componentes_excluir = ['ner', 'textcat', 'lemmatizer', 'senter', 'sentencizer', 'tagger', 'entity_linker', 'parser', 'entity_ruler', 'textcat_multilabel']
nlp = spacy.load("pt_core_news_sm", exclude=componentes_excluir)

def tfidf(reviews: List[str]) -> pd.DataFrame:

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

# BIGRAM COM PONTUAÇÃO
def bigrams(review_textos: List[str]) -> pd.DataFrame:

    vectorizer = CountVectorizer(ngram_range=(2, 3))
    X = vectorizer.fit_transform(review_textos)
    bigram = vectorizer.get_feature_names_out()
    pontuacao = X.sum(axis=0).tolist()
    pontuacao = pontuacao[0]

    df_bigram = pd.DataFrame({'BIGRAM': bigram, 'PONTUACAO': pontuacao})
    df_bigram.sort_values('PONTUACAO', ascending=False, inplace=True)
    df_bigram = df_bigram.iloc[10:21]

    return df_bigram['BIGRAM'].tolist()

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
