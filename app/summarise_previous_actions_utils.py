from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import os

SUMMARY_DATA_PATH = "../data/Closed_Incident_Data_w_Summaries/summarised_data.csv"

load_dotenv("../.env")
AZURE_API_KEY = os.getenv('AZURE_API_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT')

llm = AzureChatOpenAI(
    azure_deployment='gpt-4o',
    api_version="2024-08-01-preview",
    temperature=0.3,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
)

OVERALL_SUMMARY_PROMPT = """
You are an AI assistant tasked with analyzing network incident close note summaries which describe the resolution steps taken to fix a network incident.
Your task is to summarise these close note summaries into one piece of text.
You should clearly identify the different methods taken to resolve the incident, giving more emphasis on the methods which are often repeated.
Your response should be a concise paragraph without any formatting.

Close Note Summaries: {summaries}
Answer:
"""

overall_summary_prompt_template = ChatPromptTemplate.from_template(OVERALL_SUMMARY_PROMPT)

def format_close_note_summaries(summaries):
    return "\n ----- \n".join(f"{y}" for x,y in enumerate(summaries))

overall_summary_chain = (
    {
        "summaries" : (itemgetter('summaries')
                           |
                     RunnableLambda(format_close_note_summaries)),
    }
      |
   overall_summary_prompt_template
      |
   llm
      |
   StrOutputParser()
)

def summarise_previous_actions(actions):
    response = overall_summary_chain.invoke({'summaries' : actions})
    return response

def get_close_note_summary(insight_ids):
    df_close = pd.read_csv(SUMMARY_DATA_PATH, index_col=0)

    df_relevant_close = df_close[df_close['Number'].isin(insight_ids)]
    df_useful_relevant_close = df_relevant_close[~df_relevant_close['close_notes_summary'].str.lower().str.contains('no relevant information provided')]

    assigned_to_counts = df_relevant_close['Assigned to'].value_counts()
    most_commonly_assigned_to = ", ".join(list(assigned_to_counts[assigned_to_counts == assigned_to_counts.max()].index))
    most_recently_assigned_to = df_relevant_close.sort_values(by=['Resolved'], ascending=False).iloc[0]['Assigned to']

    most_recently_resolved_date = df_relevant_close.sort_values(by=['Resolved'], ascending=False).iloc[0]['Resolved']
    most_recently_resolved_date = datetime.strptime(most_recently_resolved_date, "%Y-%m-%d %H:%M:%S").strftime("%B %d, %Y")
    least_recently_resolved_date = df_relevant_close.sort_values(by=['Resolved'], ascending=True).iloc[0]['Resolved']
    least_recently_resolved_date = datetime.strptime(least_recently_resolved_date, "%Y-%m-%d %H:%M:%S").strftime("%B %d, %Y")

    if len(df_useful_relevant_close) > 0:
        summarised_close_notes = summarise_previous_actions(list(df_useful_relevant_close['close_notes_summary']))
        response = f'''Found {len(df_relevant_close)} previously closed similar incidents, resolved between {least_recently_resolved_date} to {most_recently_resolved_date}. These similar insights are most commonly assigned to {most_commonly_assigned_to} ({assigned_to_counts.max()} time(s)), and most recently assigned to {most_recently_assigned_to}. Of these previously closed similar incidents, there are {len(df_useful_relevant_close)} close notes detailing actions taken to resolve the incidents. A summary of these close notes is provided in the following text: '{summarised_close_notes}'.'''
    else:
        response = f'''Found {len(df_relevant_close)} previously closed similar incidents, resolved between {least_recently_resolved_date} to {most_recently_resolved_date}. These similar insights are most commonly assigned to {most_commonly_assigned_to} ({assigned_to_counts.max()} time(s)), and most recently assigned to {most_recently_assigned_to}. There are no detailed close notes from these incidents.'''

    return response

def get_all_close_notes(query_df):
    df_close = pd.read_csv(SUMMARY_DATA_PATH, index_col=0)
    return query_df[['id', 'eventNames', 'eventObjects', 'eventCIs', 'overall_distance_score']].merge(df_close, left_on='id', right_on='Number', how='left')[['Number', 'eventNames', 'eventObjects', 'eventCIs', 'Assigned to', 'Resolved', 'close_notes_summary', 'Close notes', 'overall_distance_score']]
