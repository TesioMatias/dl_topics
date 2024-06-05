# %%
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
# %%
import pandas as pd
df = pd.read_parquet('data/df_joined_2024-04-01 00:00:00.paquet')
# %%

# %%
# Step 1 - Extract embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# %%
from sklearn.metrics.pairwise import cosine_similarity
# %%

# %%
# Step 2 - Reduce dimensionality
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')

# %%
# Step 3 - Cluster reduced embeddings
hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
# %%
# Step 4 - Tokenize topics
vectorizer_model = CountVectorizer(

)
# %%
# Step 5 - Create topic representation
ctfidf_model = ClassTfidfTransformer()
# %%
# Step 6 - (Optional) Fine-tune topic representations with 
# a `bertopic.representation` model
representation_model = KeyBERTInspired()
# %%

# All steps together
topic_model = BERTopic(
  embedding_model=embedding_model,          # Step 1 - Extract embeddings
  umap_model=umap_model,                    # Step 2 - Reduce dimensionality
  hdbscan_model=hdbscan_model,              # Step 3 - Cluster reduced embeddings
  vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
  ctfidf_model=ctfidf_model,                # Step 5 - Extract topic words
  representation_model=representation_model # Step 6 - (Optional) Fine-tune topic represenations
)
# %% Load data
# %%
topic_model.fit(df['in__title'])
# %%
topic_model.get_topics()
# %%
topic_model.visualize_topics()
# %%
topic_model.visualize_hierarchy()
# %%
topic_model.visualize_documents(list(df['in__title']))
# %%
topic_model.visualize_term_rank()
# %%
