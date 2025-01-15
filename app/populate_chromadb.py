import os
import shutil
import os
import json
import numpy as np
import logging
import chromadb
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = "../data/Incoming_Incident_Data"
CHROMA_PATH = "../chromadbs/"

def main():

    clear_chromadb_directory()

    eventName_processor = EventDataProcessor(event_data_name="eventName", folder_path=DATA_PATH, chromadb_path=CHROMA_PATH)
    eventName_processor.construct_chromadb_collection()

    eventObject_processor = EventDataProcessor(event_data_name = "eventObject", folder_path=DATA_PATH, chromadb_path=CHROMA_PATH)
    eventObject_processor.construct_chromadb_collection()

    eventCI_processor = EventDataProcessor(event_data_name = "eventCI", folder_path=DATA_PATH, chromadb_path=CHROMA_PATH)
    eventCI_processor.construct_chromadb_collection()

def clear_chromadb_directory():

    if len(os.listdir(CHROMA_PATH)) > 0:
        for filename in os.listdir(CHROMA_PATH):
            file_path = os.path.join(CHROMA_PATH, filename)

            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)

class EventDataProcessor:

    def __init__(self, event_data_name, folder_path, chromadb_path, embedding_model = 'all-MiniLM-L6-v2'):
        self.event_data_name = event_data_name
        self.folder_path = folder_path
        self.chromadb_path = chromadb_path
        self.event_data = {
            "incident_id" : [],
            self.event_data_name : []
            }
        self.embedding_model = SentenceTransformer(embedding_model)
        self.collection_name = f"incident_{self.event_data_name}"

        if self.event_data_name == 'eventName':
            for i in range(1,4):
                self.event_data[f'eventName_part{i}'] = []

    def _extract_event_data(self):
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(self.folder_path, filename)
                self._process_file(file_path, filename)

    def _process_file(self, file_path, filename):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
        except Exception as e:
            logging.error(f"Error processing file {filename} : {e}")
            return
        
        incident_id = filename.split(".")[0]
        self.event_data["incident_id"].append(incident_id)

        alerts = self._get_unique_alerts(data)

        if self.event_data_name == 'eventName':
            self._process_event_name_alerts(alerts)
        
        self.event_data[self.event_data_name].append(", ".join(sorted(alerts, key=str.lower)))

    def _get_unique_alerts(self, data):
        alerts = set()
        for alert in data:
            alert_value = alert["cf"].get(self.event_data_name)
            if alert_value:
                alerts.add(alert_value)
        return sorted(alerts, key=str.lower)
    
    def _process_event_name_alerts(self, alerts):
        alert_parts = [alert.split(".") for alert in alerts]

        if alert_parts:
            sep_alerts = [sorted(list(set(part_list)), key=str.lower) for part_list in zip(*alert_parts)]

            for i, part_list in enumerate(sep_alerts, start=1):
                part_name = f'eventName_part{i}'
                self.event_data[part_name].append(", ".join(part_list))

        else:
            for i in range(1,4):
                self.event_data[f'eventName_part{i}'].append('')
    
    def _construct_event_data_embeddings(self):

        self._extract_event_data()

        if self.event_data_name == 'eventName':
            alerts_encodings = []
            for alerts in [self.event_data['eventName_part1'], self.event_data['eventName_part2'], self.event_data['eventName_part3']]:
                alerts_encodings.append(self.embedding_model.encode(alerts).tolist())
            insight_encoded = np.mean(alerts_encodings, axis=0)

        else:
            alerts_encoded = self.embedding_model.encode(self.event_data[self.event_data_name])
            insight_encoded = alerts_encoded.tolist()

        return insight_encoded
    
    
    def construct_chromadb_collection(self):

        client = chromadb.PersistentClient(self.chromadb_path)

        embeddings = self._construct_event_data_embeddings()
        metadatas = [{"alerts" : str(x), "id" : y} for x,y in zip(self.event_data[self.event_data_name], self.event_data['incident_id'])]
        ids = self.event_data['incident_id']


        collection = client.create_collection(
            name = self.collection_name,
            metadata={"hnsw:space":"cosine", "hnsw:M":256}
        )
        collection.add(
            embeddings = embeddings,
            metadatas = metadatas,
            ids = ids
        )

        return collection

if __name__ == "__main__":
    main()