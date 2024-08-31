
from network.database import Database

class Repository():
    def __init__(self):
        self.client = Database()

    def create_index(self, index):
        print('\nCreating index:')
        resp = self.client.create_index(index)
        print(resp)

    
    def search(self, query, index, size=10):
        return self.client.search(
            query=query,
            index=index,
            size=size
        )
    
    def get_client(self):
        return self.client.get_client()