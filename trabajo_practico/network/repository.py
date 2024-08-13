
from network.database import Database
from datetime import datetime
from dateutil.parser import parse
from data.TopicDAO import Topic, TopicKeyword

class Repository():
    def __init__(self):
        self.client = Database()

    def __get_topic_name(keywords):
        return ', '.join([k for k, s in keywords[:4]])

    def create_index(self, index):
        return self.client.create_index(index)
    
    def search(self, query, index):
        return self.client.search(
            body=query,
            index=index
        )
    
    def save_model(self, model, sim_matrix, df, date_from, date_to, index):

        for topic in model.get_topics().keys():
            if topic > -1:
                keywords = model.topic_representations_[topic]
                print(keywords)
                topic_keywords = [TopicKeyword(name=k, score=s) for k, s in keywords]

                best_doc_index = sim_matrix[topic + 1].argmax()

                best_doc = df.iloc[best_doc_index].text

                topic_doc = Topic(
                    vector = list(model.topic_embeddings_[topic + 1]),
                    similarity_threshold = 0.7,
                    created_at = datetime.now(),
                    to_date = parse(date_from),
                    from_date = parse(date_to),
                    index = topic,
                    keywords = topic_keywords,
                    name = self.__get_topic_name(keywords),
                    best_doc = best_doc
                )

                self.client.save_topic(topic_doc, index)
    
    #Just for testing
    def get_client(self):
        return self.client.get_client()