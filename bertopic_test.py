# %%
from bertopic import BERTopic
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from opensearch_data_model import Topic, TopicKeyword, os_client
from datetime import datetime
from dateutil.parser import parse
from utils import SPANISH_STOPWORDS
# %%
Topic.init()
# %%
# entities = set(sum(list([list(e) for e in df['out__potential_entities'].values]), []))
# keywords = set(sum(list([list(e) for e in df['out__keywords_sorted'].values]), []))
# all_tokens = list(entities.union(keywords))
# %%

# %%
df = pd.read_parquet('data/df_joined_2024-04-01 00:00:00.paquet')
data = list(df['in__title'])
# %%
# %%
from collections import Counter
entities = set(sum(list([list(e) for e in df['out__potential_entities'].values]), []))
keywords = set(sum(list([list(e) for e in df['out__keywords_sorted'].values]), []))
all_tokens = list(entities.union(keywords))
# %%
from sklearn.feature_extraction.text import CountVectorizer
tf_vectorizer = CountVectorizer(
    # tokenizer=tokenizer,
    # max_df=0.1,
    # min_df=10,
    ngram_range=(1, 3),
    stop_words=SPANISH_STOPWORDS,
    lowercase=False,
    vocabulary=all_tokens,
    # max_features=100_000
)
tf_vectorizer.fit(data)
# %%
topic_model = BERTopic(
    language='spanish',
    # calculate_probabilities=True
    vectorizer_model=tf_vectorizer
)
# %%
topics, probs = topic_model.fit_transform(data)
# %%
df['topic'] = topics
df['probs'] = probs
# %%
df[df['topic']==1][['in__title', 'topic', 'probs']]
# %%
len(topics)
# %%
topic_model.topic_representations_[1]
# %%
len(topics), len(df)

# %%
topic_model.topic_embeddings_.shape
# %%
embedings = topic_model.embedding_model.embed(data)
sim_matrix = cosine_similarity(
    topic_model.topic_embeddings_,
    embedings
)
# %%
def get_topic_name(keywords):
    return ', '.join([k for k, s in keywords[:4]])
# %%

# %%
for topic in topic_model.get_topics().keys():
    if topic > -1:
        print(topic)
        keywords = topic_model.topic_representations_[topic]
        topic_keywords = [TopicKeyword(name=k, score=s) for k, s in keywords]


        best_doc_index = sim_matrix[topic + 1].argmax()

        best_doc = df.iloc[best_doc_index].in__title

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
Topic.search().count()
# %%
for doc in Topic.search().query().scan():
    break
# %%
doc.to_dict()
# %%
new_doc = 'Javier Milei se reunió con los gobernadores tras las discuciones'

new_doc_embed = topic_model.embedding_model.embed(new_doc)
# %%
new_doc_embed.shape
# %%
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
import pandas as pd
df_hits = pd.DataFrame(response['hits']['hits'])
# %%
winning_topic = Topic.get(df_hits.iloc[0]._id)
# %%
winning_topic.to_dict()
# %%

# %%
probs_n = probs[6]
probs_n.sort()
# %%
# %% Calculate probabilities
df_top_probs = pd.DataFrame([{'prob': p, 'topic': t} for t, p in zip(topics, probs)])
df_top_probs
# %%
topic_0_indexes = df_top_probs[
    (df_top_probs['prob'] >= 0.99) &
    (df_top_probs['topic'] == 2)
].index
topic_0_indexes
# %%
topic_0_docs = list(df.iloc[topic_0_indexes]['in__title'])
topic_0_docs
# %%
# topic_0_embeddings = topic_model.embedding_model.embed(topic_0_docs)
topic_0_embeddings = topic_model._extract_embeddings(topic_0_docs)
# %%
topic_0_embeddings.shape
# %%

# %%
cosine_similarity(topic_0_embeddings[:10], topic_0_embeddings[:10]).shape
# %%
cosine_similarity(topic_0_embeddings[:10], topic_model.topic_embeddings_).shape

# %%
doc_idx = 5
doc_topic = probs[doc_idx].argsort()
print(doc_topic)
probs[doc_idx][doc_topic][::-1]
# %%
df_top_probs = pd.DataFrame([{'prob': p, 'topic': t} for t, p in zip(topics, probs)])
# %%
df_top_probs
# %%
from collections import Counter
Counter(topics)
# %%
topic_model.topic_embeddings_[1]
# %%
topic_model.topic_representations_
# %%
topic_similarites = cosine_similarity(topic_model.topic_embeddings_, topic_model.topic_embeddings_)
# %%
topic_similarites[:3,:3]
# %%

# %%
for idx in range(total_topics):
    max_idx = topic_model.probabilities_[idx].argmax()
    prob = topic_model.probabilities_[idx][max_idx]
    print(max_idx, prob, topics[idx], probs[idx][max_idx])
    
# %%        
# %%
topics = topic_model.get_topics()
print(topics)
print(len(topics))
# %%
hierarchical_topics = topic_model.hierarchical_topics(data)
hierarchical_topics
# %%
topic_model.get_topic(0)
# %%
topic_model.get_representative_docs(1)
# %%
topic_model.embedding_model.embed('Hola que tal').shape
# %%
topic_model.c_tf_idf_
# %%
topic_model.language
# %%
# topic_model.merge_models()
# %%
topic_model.topic_embeddings_.shape
# %%
topic_model.get_topic_freq(2)
# %%
topic_model.get_topic_info(0)
# %%
topic_model.topic_sizes_
# %%
topic_model.get_topics()
# %%
new_docs_topics, new_docs_probs = topic_model.transform(
    [
        'Novedades en la guerra de Ucrania y Rusia',
        'Echaron a participante de Gran Hermano'
    ]
)
# %%
sorted_probs = new_docs_probs.argsort(axis=1)
sorted_probs
# %%
idx = 1
new_docs_probs[idx][sorted_probs[idx]][::-1]
# %%
topic_model.get_topic(34)

# %% Interesante
topic_model.partial_fit()
# %%
len(topic_model.vectorizer_model.get_feature_names_out())
# %%
topic_model.visualize_topics()
# %%
topic_model.visualize_hierarchy()
# %%
topic_model.visualize_heatmap()
# %%
topic_model.visualize_distribution(probs[3], min_probability=0.001)
# %%
# topic_model.visualize_documents(list(df['in__title']))
# %%
topic_model.visualize_term_rank([0, 1, 2, 3])
# %%

# %%
from transformers import pipeline
classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
)
# %%

sequence_to_classify = "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU"
candidate_labels = ["economy", "entertainment", "environment"]
# %%
output = classifier(
    sequence_to_classify, candidate_labels,
    multi_label=True
)
print(output)

# %%
output['labels']
# %%
output['scores']
# %%
