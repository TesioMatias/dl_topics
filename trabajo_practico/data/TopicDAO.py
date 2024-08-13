from opensearchpy import Float, Field, Integer, Document, Keyword, Text, Date, Object, InnerDoc

TOPIC_DIMENSIONS = 384
knn_params = {
    "name": "hnsw",
    "space_type": "cosinesimil",
    "engine": "nmslib"
}

class TopicKeyword(InnerDoc):
    name = Keyword()
    score = Float()

class SimilarTopics(Document):
    topic_id = Keyword()
    similar_to = Keyword()
    similarity = Float()
    common_keywwords = Keyword()
    keywords_not_in_similar = Keyword()
    keywords_not_in_topic = Keyword()

class KNNVector(Field):
    name = "knn_vector"
    def __init__(self, dimension, method, **kwargs):
        super(KNNVector, self).__init__(dimension=dimension, method=method, **kwargs)

class Topic(Document):
    vector = KNNVector(TOPIC_DIMENSIONS, knn_params)
    similarity_threshold = Float()
    created_at = Date()
    to_date = Date()
    from_date = Date()
    index = Integer()
    keywords = Object(TopicKeyword)
    name = Text()
    best_doc = Text()