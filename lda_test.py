# %%
import pandas as pd
import numpy as np
import nltk
import re
from utils import plot_top_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
# %%
df = pd.read_parquet('data/df_joined_2024-04-01 00:00:00.paquet')
# %%
nltk.download('stopwords')

#%%
df['in__title']
#%%
data = list(df['in__title'])
data = [re.sub(r'\s+', ' ', sent) for sent in data]
data = [re.sub(r"\'", "", sent) for sent in data]

# %%
tf_vectorizer = CountVectorizer(
    stop_words=stopwords.words('spanish'),       
    min_df=10,
    lowercase=True, 
    ngram_range=(1,3),
    max_features=10000
)
#tf = tf_vectorizer.fit_transform(list(df['in__title']))
tf = tf_vectorizer.fit_transform(data)
# %%
n_topics=15
lda = LatentDirichletAllocation(
    n_components=n_topics, 
    learning_decay=0.5,
    learning_method='online', 
    learning_offset=50.,
    random_state=0
)
# %%
doc_probs = lda.fit_transform(tf)
# %%
doc_probs.shape
# %%
doc_probs.sum(axis=1)
# %%
lda.components_.shape
# %%
lda.components_[:, 0]
# %%
plot_top_words(
    lda,
    np.array(tf_vectorizer.get_feature_names()),
    15,
    'LDA Plot'
)
# %%
#GRID SEARCH TO LOOK FOR HYPER PARAMS
search_params = {'n_components': [5 ,10, 15, 20, 25, 30], 'learning_decay': [.3, .5, .7, .9]}

lda = LatentDirichletAllocation(
    max_iter=5, 
    learning_method='online', 
    learning_offset=50.,
    random_state=0)

model = GridSearchCV(lda, param_grid=search_params)

model.fit(tf)

GridSearchCV(
    cv=None, 
    error_score='raise',
    estimator=LatentDirichletAllocation(),
    fit_params=None, iid=True, n_jobs=1,
    param_grid={'n_topics': [5, 10, 15, 20, 25, 30], 'learning_decay': [0,3, 0.5, 0.7, 0.9]},
    pre_dispatch='2*n_jobs', 
    refit=True, 
    return_train_score='warn',
    scoring=None, 
    verbose=0
)
# %%
# Best Model
best_lda_model = model.best_estimator_# Model Parameters
print("Best Model's Params: ", model.best_params_)# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(tf))
# %%
