# %%
# Donwloads
# !pip install nltk
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("omw-1.4")
# !pip install gensim
# !pip install scipy==1.12
# !pip install datasets
# !export PYTHONPATH=.

# %%
from network.repository import Repository
from topic_model.TopicModelManager import fit_transform
from utils.eda_utils import load_info, eda, prepare_info_for_model
from sklearn.metrics.pairwise import cosine_similarity

# %%
# 1 - Define Datasets

ds_list = [
    "jganzabalseenka/news_2024-07-09_24hs",
    "jganzabalseenka/news_2024-07-10_24hs",
    "jganzabalseenka/news_2024-07-11_24hs",
    "jganzabalseenka/news_2024-07-12_24hs",
    "jganzabalseenka/news_2024-07-13_24hs"
]

# %%
# 2 - Create repository and index
repository = Repository()

index = 'topics-index'
resp = repository.create_index(index=index)
print('\nCreating index:')
print(resp)

# %%
# 3 - Load and EDA over the dataset
day_zero_df = load_info(dataset = ds_list[0])

eda(df = day_zero_df)

data, kw, entities = prepare_info_for_model(
    df = day_zero_df,
    first_n_elements = 2000
)

# %% 
# 4 - Train the topic model with the first day dataset
topics, probs, model = fit_transform(
    data=data,
    kw=kw,
    entities=entities
)

print(len(topics))
print(len(probs))
print(topics)
print(probs)
print(model)

# %%
# 5 - Guardado de topicos en una db
embedings = model.embedding_model.embed(data)

sim_matrix = cosine_similarity(
    model.topic_embeddings_,
    embedings
)

repository.save_model(
    df=day_zero_df, 
    date_from=ds_list[0][21:-5], 
    date_to=ds_list[0][21:-5], 
    index=index,
    sim_matrix=sim_matrix,
    model= model
)

# %%
# 6 - Llega un nuevo documento solo
model.topic_representations_
# %%
# 7 - Llega el batch del dia 1

# %%
# 8 - Llega el batch del dia 2

# %%
# 9 - Llega el batch del dia 3

# %%
# 10 - Llega el batch del dia 4





# %%
q = 'miller'
query = {
    'size': 5,
    'query': {
        'multi_match': {
            'query': q,
            'fields': ['title^2', 'director']
        }
    }
}
 
response = repository.search(query=query, index=index)
print('\nSearch results:')
print(response)







# %%

topic_model_helper.generate_topic_for_an_article("articulo de a uno") -> ids de topicos a los que el texto pertenece

topic_model_helper.generate_topic_by_batch(day_x_df)

topic_model_helper.topic_representation_by_quantity() -> necesito tener cuantos documentos tiene cada topico

topic_model_helper.topic_representation_over_time() -> necesito entender de que topicos se habló cada dia

#Sentiment analisis
sentiment_analisis = SentimentAnalisis()
sentiment_analisis.analize(df): list<G, B, N> por cada doc en el df


#ejecucion normal

topic_model_helper = TopicModelHelper()
sentiment_analisis = SentimentAnalisis()
topic_model_repository = TopicModelRepository()

topics = topic_model_helper.train(
    eda_util.prepare_info_for_model(df)
)

topic_model_repository.upsert(DaoUtils.topics())





update_df_with_topics(df_general, topics)

update_df_with_sentiment_analisis(df_general, new_docs) {
    sentiment_analisis.analize(df): list<G, B, N> por cada doc en el df
}


topic_model_helper.generate_topic_for_an_article("articulo de a uno")

update_df_with_topics(doc, topic)

update_df_with_sentiment_analisis(df_general, doc) {
    sentiment_analisis.analize(df): list<G, B, N> por cada doc en el df
}

topic_model_helper.generate_topic_by_batch(day_x_df)

update_df_with_topics(df_general, topics)

update_df_with_sentiment_analisis(df_general, new_docs) {
    sentiment_analisis.analize(df): list<G, B, N> por cada doc en el df
}



topic_model_helper.topic_representation_by_quantity()

topic_model_helper.topic_representation_over_time()








# %%
config = Config()
db_client = db(config.host, config.port, credentials)

response = db_client.create_index()
print('\nCreating index:')
print(response)
 
q = 'miller'
query = {
    'size': 5,
    'query': {
        'multi_match': {
            'query': q,
            'fields': ['title^2', 'director']
        }
    }
}
 
response = db_client.search(query)
print('\nSearch results:')
print(response)
