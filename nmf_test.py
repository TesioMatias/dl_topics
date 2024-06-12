# %%
import pandas as pd
from utils import plot_top_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, MiniBatchNMF
from utils import SPANISH_STOPWORDS
# %%
df = pd.read_parquet('data/df_joined_2024-04-01 00:00:00.paquet')
df
# %%
# Asset Name | in__title : título de la nota
# in__text: Texto de la nota
# Author Name: Medio
# out__entities: Listado de entidades
# out__potential_entities: Listado de entidades no validadas
# out__keywords_sorted: Listado de keywords
# title_and_text: Titulo y texto
# truncated_text: Titulo y texto truncado a 1000 caracteres aprox
# start_time_local: Hora de publicación
# %%
from collections import Counter
entities = set(sum(list([list(e) for e in df['out__potential_entities'].values]), []))
keywords = set(sum(list([list(e) for e in df['out__keywords_sorted'].values]), []))
all_tokens = list(entities.union(keywords))
# %%
all_tokens
# %%
print(len(all_tokens))
def tokenizer(sentence):
    return sentence.split(' ')
# %%
tokenizer('Hola que tal')
# %%
tf_vectorizer = TfidfVectorizer(
    tokenizer=tokenizer,
    max_df=0.1,
    min_df=10,
    ngram_range=(1, 3),
    stop_words=SPANISH_STOPWORDS,
    lowercase=True,
    vocabulary=all_tokens,
    max_features=100_000
)
tf = tf_vectorizer.fit_transform(list(df['title_and_text']))
# %%
tf
# %%
n_topics = 20
nmf = NMF(n_topics)
# %%
doc_topics = nmf.fit_transform(tf)
# %%
doc_topics.shape
# %%
nmf.components_.shape
# %%
doc_topics.dot(nmf.components_).shape
# %%
plot_top_words(
    nmf,
    tf_vectorizer.get_feature_names_out(),
    10,
    'NMF Plot'
)
# %%
len(df)
# %%
