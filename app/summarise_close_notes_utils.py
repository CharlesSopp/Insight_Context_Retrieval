from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import chromadb
from dotenv import load_dotenv
import pandas as pd
import time
import math
import os

load_dotenv("../.env")
AZURE_API_KEY = os.getenv('AZURE_API_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT')

DATA_PATH = "../data/Closed_Incident_Data/"
DATA_PATH_W_SUMMARY = "../data/Closed_Incident_Data_w_Summaries/"
CHROMA_PATH = "../chromadbs/"

llm = AzureChatOpenAI(
    azure_deployment='gpt-4o',
    api_version="2024-08-01-preview",
    temperature=0.3,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
)

SYS_PROMPT = """You are an expert at extracting and summarising relevant information from the close notes of a recently closed network incident.
                Follow these instructions for extracting and summarising relevant information:
                  - If the close notes contains information which could possibly answer the user's question, summarise this information in as few words as possible.
                  - If the close notes do not contain any relevant information for the user's question, respond with 'No relevant information provided'."""

summarise_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("human", """Close notes:
                     {close_notes}
                     User question:
                     How was the network incident resolved?
                  """),
    ]
)

close_notes_summariser = (summarise_prompt
                            |
                            llm
                            |
                            StrOutputParser())

def get_inc_ids():
    client = chromadb.PersistentClient(CHROMA_PATH)
    eventName_collection = client.get_collection(name='incident_eventName')

    inc_ids = []
    for inc_dict in eventName_collection.get()['metadatas']:
        if len(inc_dict['alerts']) > 0:
            inc_ids.append(inc_dict['id'])

    return inc_ids

def get_close_notes():
    if len(os.listdir(DATA_PATH)) == 1:
        inc_data_path = os.path.join(DATA_PATH, os.listdir(DATA_PATH)[0])

        if ".xlsx" in inc_data_path:
            df_close = pd.read_excel(inc_data_path)
        elif ".csv" in inc_data_path:
            df_close = pd.read_csv(inc_data_path)
        else:
            raise ValueError("Invalid file format. Please upload a file with either a .csv or .xlsx extension.")
        
        inc_ids = get_inc_ids()
        df_close = df_close[df_close['Number'].isin(inc_ids)]
        df_close = df_close.dropna(subset=['Close notes'])
        return df_close
    else:
        raise ValueError("There should only be one .csv or .xlsx file inside '/data/Closed_Incident_Data' directory")

def summarise_close_notes(df):
    batch = [{'close_notes' : x} for x in list(df['Close notes'])]
    responses = batched_close_notes_summariser(input_list = batch, batch_size = 20)
    return responses

def batched_close_notes_summariser(input_list, batch_size, chain = close_notes_summariser, cached_results_dir = "./cached_batched_responses.csv"):

    responses = []
    counter = 1
    batches_required = math.ceil(len(input_list) / batch_size)

    for i in range(0, len(input_list), batch_size):
        batch = input_list[i : i + batch_size]
        complete = False
        end_for_loop = False
        try_counter = 0

        if end_for_loop:
            break

        while not complete:
            if try_counter < 5:
                try:
                    try_counter += 1
                    response = chain.batch(batch, config={"max_concurrency": 10})
                    responses.extend(response)
                    complete = True
                    print(f"-- Finished batch {counter} / {batches_required} --")
                    counter += 1
                except:
                    print(f"-- Failed batch {counter} / {batches_required}, re-trying after 60 seconds")
                    time.sleep(30)
            else:
                end_for_loop = True
                complete = True
                pd.DataFrame(data={'response' : responses}).to_csv(cached_results_dir)
                print(f"Failed for input with index {counter} 5 times, so ending for loop and saving current progress to csv file")
    
    return responses