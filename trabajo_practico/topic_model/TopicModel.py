from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer

class TopicModel():

    def __init__(
            self,
            embedding_model=SentenceTransformer("sentence-transformers/all-mpnet-base-v2"), 
            dimensionality_reduction_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine'), 
            cluster_model=HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True), 
            tokenizer=CountVectorizer(), 
            topic_representation=None, 
            topic_rep_fine_tuner=None
            ):
        
        self.topic_model = BERTopic(
            language='spanish',
            calculate_probabilities=False,
            embedding_model=embedding_model,                    # Step 1 - Extract embeddings
            umap_model=dimensionality_reduction_model,          # Step 2 - Reduce dimensionality
            hdbscan_model=cluster_model,                        # Step 3 - Cluster reduced embeddings
            vectorizer_model=tokenizer,                         # Step 4 - Tokenize topics
            ctfidf_model=topic_representation,                  # Step 5 - Extract topic words
            representation_model=topic_rep_fine_tuner           # Step 6 - (Optional) Fine-tune topic represenations
        )

    def fit_transform(self, doc_list):
        return self.topic_model.fit_transform(doc_list)
    
    def get_model(self):
        return self.topic_model
