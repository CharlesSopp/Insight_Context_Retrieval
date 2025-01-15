import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
CHROMA_PATH = "../chromadbs/"

def get_eventName_embedding(emb_model, event_data):
    part1, part2, part3 = set(), set(), set()

    for alert in event_data:
        parts = alert.split(".")

        part1.add(parts[0])
        part2.add(parts[1])
        part3.add(parts[2])

    sorted_part1 = sorted(part1, key=str.lower)
    sorted_part2 = sorted(part2, key=str.lower)
    sorted_part3 = sorted(part3, key=str.lower)

    parts_encodings = []

    for part in [sorted_part1,sorted_part2,sorted_part3]:
        encoded_part = emb_model.encode(", ".join(part))
        parts_encodings.append(encoded_part)

    insight_encoded = np.mean(parts_encodings, axis=0)

    return insight_encoded

def query_collection(collection, event_data, filter_ids = None, k=100):
    if collection.name == 'incident_eventName':
        embedded_event_data = get_eventName_embedding(embedding_model, event_data)
    else:
        sorted_event_data = sorted(list(set(event_data)), key=str.lower)
        embedded_event_data = embedding_model.encode(", ".join(sorted_event_data))

    if filter_ids:
        eventName_query_results = collection.query(
            query_embeddings=embedded_event_data,
            n_results=k,
            where={"id":{"$in" : filter_ids}}
        )
    else:
        eventName_query_results = collection.query(
            query_embeddings=embedded_event_data,
            n_results=k
        )

    return eventName_query_results

def overall_cos_distance(row):
    return ((2*row['eventName_cos_distance']) + row['eventObject_cos_distance'] + row['eventCI_cos_distance']) / 4

def query_insights(inc_eventNames, inc_eventObjects, inc_eventCIs, threshold, chromadb_path=CHROMA_PATH):

    client = chromadb.PersistentClient(chromadb_path)
    eventName_collection = client.get_collection(name='incident_eventName')
    eventObject_collection = client.get_collection(name='incident_eventObject')
    eventCI_collection = client.get_collection(name='incident_eventCI')

    ids_matched_on_eventName = query_collection(eventName_collection, inc_eventNames, k=100)
    eventName_results_df = pd.DataFrame(ids_matched_on_eventName['metadatas'][0]).rename(columns={'alerts' : 'eventNames'})
    eventName_results_df['eventName_cos_distance'] = ids_matched_on_eventName['distances'][0]

    ids_matched_on_eventName_eventObject = query_collection(eventObject_collection, inc_eventObjects, filter_ids = ids_matched_on_eventName['ids'][0], k=100)
    eventObject_results_df = pd.DataFrame(ids_matched_on_eventName_eventObject['metadatas'][0]).rename(columns={'alerts' : 'eventObjects'})
    eventObject_results_df['eventObject_cos_distance'] = ids_matched_on_eventName_eventObject['distances'][0]

    ids_matched_on_eventName_eventCI = query_collection(eventCI_collection, inc_eventCIs, filter_ids = ids_matched_on_eventName['ids'][0], k=100)
    eventCI_results_df = pd.DataFrame(ids_matched_on_eventName_eventCI['metadatas'][0]).rename(columns={'alerts' : 'eventCIs'})
    eventCI_results_df['eventCI_cos_distance'] = ids_matched_on_eventName_eventCI['distances'][0]

    merged_df = pd.merge(eventName_results_df, eventObject_results_df, on='id')
    merged_df = pd.merge(merged_df, eventCI_results_df, on='id')

    merged_df['overall_distance_score'] = merged_df.apply(overall_cos_distance, axis=1)
    merged_df = merged_df[merged_df['overall_distance_score'] < threshold]

    return merged_df.sort_values(by=['overall_distance_score'], ascending=True)