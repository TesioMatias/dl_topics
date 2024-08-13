from datasets import load_dataset
import pandas as pd

def __sanitize_info(df):
    resut_list = [df[col].hasnans for col in df]
    for result in resut_list:
        if result != False:
            return False
    return True 


def __get_kws(df, first_n_elements=3000):
    return set.union(*(set(e) for e in df['keywords'][:first_n_elements]))

def __get_entities(df, first_n_elements):
    return set.union(*(set(e) for e in df['entities'][:first_n_elements]))

def load_info(dataset):
    ds = load_dataset(dataset, split="train")
    df = pd.DataFrame(ds)
    if __sanitize_info(df):
        return df    
    return "The dataset contain nulls"

def eda(df):
    print("Kw: " +str(df.iloc[0]['keywords']))
    print("Entities: " + str(df.iloc[0]['entities_transformers']))
    print("canidad de documentos: " + str(len(df)))
    print("canidad de documentos en lista: " + str(len(df)))
    df.head()

def prepare_info_for_model(df, first_n_elements):
    data = data = list(df['text'][:first_n_elements])
    kw = __get_kws(df, first_n_elements)
    entities = __get_entities(df, first_n_elements)
    return data , kw, entities