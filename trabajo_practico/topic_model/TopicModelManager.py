from topic_model import TopicModel as tm
from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.cluster import MiniBatchKMeans
from hdbscan import HDBSCAN
from sklearn.decomposition import IncrementalPCA
from umap import UMAP
from bertopic.vectorizers import OnlineCountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from utils.utils import SPANISH_STOPWORDS

ONLINE_LEARNING = False

def __get_embedding_model():
    print("setting embeddings model")
    return SentenceTransformer("all-MiniLM-L6-v2")

def __get_dim_reduction():
    print("setting UMAP model")
    if ONLINE_LEARNING :
        return IncrementalPCA(n_components=5)
    return UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')

def __get_clustering_model():
    print("setting HDBSCAN model")
    if ONLINE_LEARNING :
        return MiniBatchKMeans(n_clusters=50, random_state=0)
    return HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

def __get_tokenizer(data, kw, entities):
    ent = set.union(*(set(e) for e in entities))
    keywords = set.union(*(set(e) for e in kw))
    all_tokens = list(ent.union(keywords))

    if ONLINE_LEARNING :
        tokenizer = OnlineCountVectorizer( 
            stop_words=SPANISH_STOPWORDS, 
            decay=.01,
            ngram_range=(1, 3),
            lowercase=True,
            vocabulary=all_tokens
        )
    else:
        tokenizer = CountVectorizer(
            ngram_range=(1, 3),
            stop_words=SPANISH_STOPWORDS,
            lowercase=True,
            vocabulary=all_tokens,
        )
    print("training tokenizer")
    tokenizer.fit(data)
    print("Tokenizer trained")
    return tokenizer


def fit_transform(data, kw, entities):
    # Bertopic

    ## Step 1 - Extract embeddings
    embedding_model = __get_embedding_model()

    # Step 2 - Reduce dimensionality
    reduce_dim = __get_dim_reduction()

    # Step 3 - Cluster reduced embeddings
    cluster = __get_clustering_model()

    # Step 4 - Tokenize topics
    tokenizer = __get_tokenizer(data, kw, entities)

    # Step 5 - Extract topic words
    #topic_representation = ClassTfidfTransformer()

    # Step 6 - (Optional) Fine-tune topic represenations
    #representation_fine_tuner= KeyBERTInspired()

    model = tm.TopicModel(
        embedding_model=embedding_model,
        dimensionality_reduction_model=reduce_dim, 
        cluster_model=cluster, 
        tokenizer=tokenizer
    )

    #    topic_representation=topic_representation, 
    #   topic_rep_fine_tuner=representation_fine_tuner

    topics, probs = model.fit_transform(data)
    return topics, probs, model.get_model()






   
#def generate_topic_for_an_article(article):

#def merge_models(old_model, new_model):

#def topic_representation_by_quantity():

#def topic_representation_over_time():

