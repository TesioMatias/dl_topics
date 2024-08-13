# %%
#!export PYTHONPATH=.
# %%
### Importa todos los elementos para trabajar
import pandas as pd
import numpy as np
from bertopic import BERTopic
from utils import SPANISH_STOPWORDS
from sklearn.metrics.pairwise import cosine_similarity
from opensearch_data_model import Topic, TopicKeyword, os_client
from datetime import datetime
from dateutil.parser import parse
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# %%
### Inicializa Topic, esto no entiendo muy bien para que lo hace, no le veo mucho sentido
Topic.init()

# %%
### Lee un dia de data y se gauarda en una lista los documentos de ese dia
df = pd.read_parquet('data/df_joined_2024-04-01 00:00:00.paquet')
data = list(df['in__title'])

# %%
### Informacion varia acerca de la lista de documentos, en este dia hay 3104 documentos
print(data[0])
len(data)

# %%
### Obtiene del dataframe leido con toda la info del dia, las entidades y los kw de cada documento y los unifica en un set
entities = set(sum(list([list(e) for e in df['out__potential_entities'].values]), []))
keywords = set(sum(list([list(e) for e in df['out__keywords_sorted'].values]), []))
for item in keywords:
    print(item)

# %%
### Unifica las entidades posibles y los kw en una lista entera de tokens y muestra los primeros 10
all_tokens = list(entities.union(keywords))
print(all_tokens[:10])

# %%
# Se crea un vectorizador de los documentos para pasarle a Bertopic, indicando que el vocabulario son los tokens de entities y keywords
tf_vectorizer = CountVectorizer(
    # tokenizer=tokenizer,
    # max_df=0.1,
    # min_df=10,
    ngram_range=(1, 3),
    stop_words=SPANISH_STOPWORDS,
    lowercase=True,
    vocabulary=all_tokens,
    # max_features=100_000
)
tf_vectorizer.fit(data)

# %%
# Se instancia BERTopic con una config multilingual, sin calculo de probs y con el vectorizer anterior
topic_model = BERTopic(
    language='spanish',
    calculate_probabilities=False,
    vectorizer_model=tf_vectorizer
)
# %%
# Se hace el fit_transforme sobre los documentos y se obtienen topics y probs de corresponder a dicho topic de cada documento
topics, probs = topic_model.fit_transform(data)

# %%
# Se muestra algo de info para entender la salida del fit_transform
print(len(topics))
print(len(data))
print(probs.shape)
print(probs[:10])
print(topics[:10])

# %%
# Se agergan la columna "topic" y "probs" al df (ahora df2) para indicar por cada documento a que topico pertenece y con que prob
df['topic'] = topics
df2 = df.assign(probs=probs)

df2.head()
####df['probs'] = probs

# %%
# Se muestra algo de informacion para entender como queda sobre el topico 1
df2[df['topic']==1][['in__title', 'topic', 'probs']]

# %%
# Se ve que cantidad de topicos hay (80) y que cantidad de documentos (3104)
print(np.array(topics).max())
print(len(topics))

# %%
#Showing top N(10) most representative tokens
topic_model.topic_representations_[4]

# %%
# Showing len of topics in documents and df are the same
len(topics), len(df2)

# %%
# Showing that there are 84 topics and each topic has an embedding of 384 dimensions
topic_model.topic_embeddings_.shape
len(topic_model.get_topics().keys())

# %%
topic_model.get_topics().keys()
# %%
# Crea los embeddings de los documentos con el modelo de embeddings de bertopic 
# Usa esos embeddings para calcular la matriz de similitud entre cada documento y cada topic
embedings = topic_model.embedding_model.embed(data)

sim_matrix = cosine_similarity(
    topic_model.topic_embeddings_,
    embedings
)

#%%
# Se muestra la info de la sim matrix para entender como es
print(sim_matrix)
print(sim_matrix.shape)

# %%
# Se define una funcino que toma como parametro una lista de kw y retorna un string concatenando los primemros 4 elementos con una coma
def get_topic_name(keywords):
    return ', '.join([k for k, s in keywords[:4]])

# %%
# Me devuelve los topicos y las palabras que mejor represennttan dicho topico
topic_model.get_topics()
topic_model.topic_representations_[1]

# %%
# Recorre todos los topicos mappeando info del df al tipo de datos Topic y luego lo guarda.
# No entiendo bien si lo guarda en algun lado o ese Save no hace nada y ahi hay que intentar guardarlo en OpenSearch

for topic in topic_model.get_topics().keys():
    if topic > -1:
        print(topic)
        keywords = topic_model.topic_representations_[topic]
        topic_keywords = [TopicKeyword(name=k, score=s) for k, s in keywords]

        print(len(topic_keywords))

        best_doc_index = sim_matrix[topic + 1].argmax()

        best_doc = df2.iloc[best_doc_index].in__title

        topic_doc = Topic(
            vector = list(topic_model.topic_embeddings_[topic + 1]),
            similarity_threshold = 0.7,
            created_at = datetime.now(),
            to_date = parse('2024-04-02'),
            from_date = parse('2024-04-01'),
            index = topic,
            keywords = topic_keywords,
            name = get_topic_name(keywords),
            best_doc = best_doc
        )

        print(topic_doc.save())
# %%
# No entiendoo por qué esto me da 161, cuando tenia 82 topics ???
Topic.search().count()

# %%

# Obtienen un topico de los guardados
for doc in Topic.search().query().scan():
    break

print(doc)

# %%
# Mappea el objeto Topic de un topico a un dict 
doc.to_dict()

# %%
# Encuentra un nuevo documento
new_doc = 'Javier Milei se reunió con los gobernadores tras las discuciones'

# genera el embedding del nuevo documento
new_doc_embed = topic_model.embedding_model.embed(new_doc)
new_doc_embed.shape

# %%
# genera la query para buscar dentro de openSearch y hace la busqueda obteniendo el response
query = {
    "size": 5,
    "query": {
        "knn": {
        "vector": {
            "vector": list(new_doc_embed),
            "k" : 1000
        }
        }
    }
}
response = os_client.search(index='topic', body=query)

# %%
# Se arma un data frame con los (maximo 5) hits que encuentra openSearch para el documento (embedding) de la query
df_hits = pd.DataFrame(response['hits']['hits'])

# %%
# Hace una busqueda por id del topico en la posicion 0 (osea, el ganadoor, porque estan ordenados de mayor a menor score)
winning_topic = Topic.get(df_hits.iloc[0]._id)

# %%
# Se mappea el objeto del topico ganador a un dict
winning_topic.to_dict()

# %%
#esto seguramente funciona cuando se pone calculate probabilities
probs_n = probs[6]
probs_n.sort()

# %% Arma un df que tiene, por cada documento, a qué topico pertenece y con qué probabilidad
df_top_probs = pd.DataFrame([{'prob': p, 'topic': t} for t, p in zip(topics, probs)])
df_top_probs

# %%
# Obtiene una lista de indices de los documentos del df anterior que tienen prob 1 de pertenecer al topic 0
topic_0_indexes = df_top_probs[
    (df_top_probs['prob'] >= 0.99) &
    (df_top_probs['topic'] == 0)
].index
topic_0_indexes

# %%
#Obtiene los docummentos que corresponden al topic 0
topic_0_docs = list(df2.iloc[topic_0_indexes]['in__title'])
topic_0_docs

# %%
# Obtiene los embeddings de los documentos del topic 0
# topic_0_embeddings = topic_model.embedding_model.embed(topic_0_docs)
topic_0_embeddings = topic_model._extract_embeddings(topic_0_docs)
topic_0_embeddings.shape

# %%
# Valida la similitud en una matriz de similitud entre los 10 primeros documentos del topic 0 contra todos los topicos
cosine_similarity(topic_0_embeddings[:10], topic_model.topic_embeddings_).shape

# %%
# Esto probablemente funcione cuando se use calculate probabilities
doc_idx = 5
doc_topic = probs[doc_idx].argsort()
print(doc_topic)
probs[doc_idx][doc_topic][::-1]

# %%
# Esto es probablemente para cuando calculas probs con bertopic, porque termina armando el mismoo df de top probs que antes
df_top_probs = pd.DataFrame([{'prob': p, 'topic': t} for t, p in zip(topics, probs)])
df_top_probs

# %%
# Arma un counter con el topico de cada documento
Counter(topics)

# %%
# Muestra el embedding del topico en la posicion 1
topic_model.topic_embeddings_[1]

# %%
# Muestra para todos los topicos, cuales son los tokens que mejor lo representan (10 tokens)
topic_model.topic_representations_

# %%
# Arma una matriz de simulitud entre topicos
topic_similarites = cosine_similarity(topic_model.topic_embeddings_, topic_model.topic_embeddings_)
topic_similarites[:3,:3]

# %%
# Esto tambien es cuando tenes el calculate probs, porque asi como está cada prob_ tiene 1 solo valor
for idx in range(3104):
    max_idx = topic_model.probabilities_[idx].argmax()
    prob = topic_model.probabilities_[idx][max_idx]
    print(max_idx, prob, topics[idx], probs[idx][max_idx])
    
# %%
#Obtiene todos los topicos del modelo entrenado
topics = topic_model.get_topics()
print(topics)
print(len(topics))

# %%
# Obtiene el orden jerarquico de los topics
hierarchical_topics = topic_model.hierarchical_topics(data)
hierarchical_topics

# %%
# Obtiene el topic 0
topic_model.get_topic(0)

# %%
# Obtiene 3 documentos representativos del topic 1
topic_model.get_representative_docs(1)

# %%
# Realiza el embedding de cualquier string con el modelo de embeddings interno de bertopic
topic_model.embedding_model.embed('Hola que tal').shape

# %%
# Muestra el tfidf de clase para entender como mostrar los topicos
topic_model.c_tf_idf_

# %%
# Indica el lenguaje seteado arriba cuando se instancia bertopic
topic_model.language

# %%
#topic_model.merge_models()

# %%
#Indica que cada uno de los 82 topicos tiene su embedding de 384 dimensiones
topic_model.topic_embeddings_.shape

# %%
# Indica la cantidad de documentos por topico
topic_model.get_topic_freq()

# %%
# Asi se obtiene la informacion representativa de un topico
topic_model.get_topic_info(0)

# %%
# Esto te muestra el tamaño en cantidad de documentos por topico
topic_model.topic_sizes_

# %%
# Esto te devuelve un dict con el topico, las palabras mas representativas y su score
topic_model.get_topics()

# %%
# Se muestra como se asignan a los topicos correspondientes nuevos documentos
new_docs_topics, new_docs_probs = topic_model.transform(
    [
        'Novedades en la guerra de Ucrania y Rusia',
        'Echaron a participante de Gran Hermano'
    ]
)
new_docs_topics, new_docs_probs

# %%
# Esto es cuando esta el calculate probs prendido
sorted_probs = new_docs_probs.argsort(axis=1)
sorted_probs

# %%
# Esto es cuando esta el calculate probs prendido
idx = 1
new_docs_probs[idx][sorted_probs[idx]][::-1]

# %%
# Muestra el topic 34
topic_model.get_topic(34)

# %% Interesante este punto para que el entrenamiento de bertopic sea por chunks de documentos y no de una
topic_model.partial_fit()
# %%
# Me indica la cantidad de elementos del vocabulario del vectorizer
len(topic_model.vectorizer_model.get_feature_names_out())
# %%
# Muestra los topicos en un grafico 2d
topic_model.visualize_topics()

# %%
# Muestra la estructura jerarquica del los topicos
topic_model.visualize_hierarchy()

# %%
# Muestra cual similares son los topicos entre si
topic_model.visualize_heatmap()

# %%
# Esto sirve cuando está prendido calculate probs
topic_model.visualize_distribution(probs[3], min_probability=0.001)

# %%
# Visualiza utilizando los mebeddings cada uno los topicos
topic_model.visualize_documents(list(df['in__title']))

# %%
# Shows how many word are used to represent a topic and when adding more words is not useful
topic_model.visualize_term_rank([0, 1, 2, 3])

# %%
# Importa pipeline para hacer inferencias usando trasnformers de hugging face y crea el clasificador zero-shot
from transformers import pipeline
classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
)

# %%
# indica la secuencia a clasificar y las posibles labels
sequence_to_classify = "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU"
candidate_labels = ["economy", "entertainment", "environment"]

# %%
# Clasifica la secuencia en las labels indicando que puede ser multi label
output = classifier(
    sequence_to_classify, candidate_labels,
    multi_label=True
)
print(output)

# %%
# Muestra la clasificacion usando pipeline
output['labels']
output['scores']

# %%
###########################PRUEBAS CON OPENSEARCH
###os_client.search(index='topic', body=query)
response2 = os_client.get(index='topic', id="0dengue-casos-Catamarca-Lousteau")
embedding_topic_0 = response2['_source']['vector']

print(topic_model.topic_embeddings_.shape)

print(np.array(embedding_topic_0).shape)

matrix = cosine_similarity(topic_model.topic_embeddings_, np.array(embedding_topic_0).reshape(1,-1))
matrix
###################################################