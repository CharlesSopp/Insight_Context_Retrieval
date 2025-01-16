from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from query_chromadb_utils import query_insights
from summarise_previous_actions_utils import get_close_note_summary, get_all_close_notes
from clustering_utils import get_summary_from_clustering
from detailed_cluster_summary_utils import get_detailed_summary
from langchain_community.callbacks import get_openai_callback
import optuna
from context_chatbot import close_notes_chatbot

SIMILARITY_THRESHOLD = 0.1

app = FastAPI()

class QuerySimilarInsights(BaseModel):
    incoming_insight: List[dict]

class ChatModel(BaseModel):
    insight_summary: str
    message: str

def extract_insight_info(incoming_insight):
    inc_eventNames = []
    inc_eventObjects = []
    inc_eventCIs = []
    for cf in incoming_insight:
        inc_eventNames.append(cf['cf']['eventName'])
        inc_eventObjects.append(cf['cf']['eventObject'])
        inc_eventCIs.append(cf['cf']['eventCI'])
    return inc_eventNames, inc_eventObjects, inc_eventCIs

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI App for retrieving insight context from similar, previously closed, network incidents"}

@app.post("/get_similar_insight_ids")
def get_similar_insight_ids_endpoint(request: QuerySimilarInsights):
    inc_eventNames, inc_eventObjects, inc_eventCIs = extract_insight_info(request.incoming_insight)
    result = list(query_insights(inc_eventNames, inc_eventObjects, inc_eventCIs, threshold=SIMILARITY_THRESHOLD)['id'])
    return {"similar_insight_ids": result}

@app.post("/get_summary_of_incidents")
def get_summary_of_incidents_endpoint(request: QuerySimilarInsights):
    inc_eventNames, inc_eventObjects, inc_eventCIs = extract_insight_info(request.incoming_insight)
    similar_ids = list(query_insights(inc_eventNames, inc_eventObjects, inc_eventCIs, threshold=SIMILARITY_THRESHOLD)['id'])
    with get_openai_callback() as cb:
        response = get_close_note_summary(similar_ids)
    return {"summary_of_incidents" : response, "cost_to_generate" : cb.total_cost}

@app.post("/get_summary_of_clustered_incidents")
def get_summary_of_clustered_incidents_endpoint(request: QuerySimilarInsights):
    inc_eventNames, inc_eventObjects, inc_eventCIs = extract_insight_info(request.incoming_insight)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    similar_ids = list(query_insights(inc_eventNames, inc_eventObjects, inc_eventCIs, threshold=SIMILARITY_THRESHOLD)['id'])
    with get_openai_callback() as cb:
        response = get_summary_from_clustering(similar_ids)
    return {"summary_of_incidents" : response, "cost_to_generate" : cb.total_cost}

@app.post("/get_all_similar_closed_incidents")
def get_all_similar_closed_incidents_endpoint(request: QuerySimilarInsights):
    inc_eventNames, inc_eventObjects, inc_eventCIs = extract_insight_info(request.incoming_insight)
    query_df = query_insights(inc_eventNames, inc_eventObjects, inc_eventCIs, threshold=SIMILARITY_THRESHOLD)
    response = get_all_close_notes(query_df)
    return {"summary_of_incidents" : response.to_dict(orient="records")}

@app.post("/get_detailed_summary_of_clustered_incidents_for_chat")
def get_detailed_summary_of_clustered_incidents_endpoint(request: QuerySimilarInsights):
    global summary
    inc_eventNames, inc_eventObjects, inc_eventCIs = extract_insight_info(request.incoming_insight)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    similar_ids = list(query_insights(inc_eventNames, inc_eventObjects, inc_eventCIs, threshold=SIMILARITY_THRESHOLD)['id'])
    with get_openai_callback() as cb:
        summary = get_detailed_summary(similar_ids)
    return {"summary_of_incidents" : summary, "cost_to_generate" : cb.total_cost}

@app.post("/chat/")
def chat(user_message: ChatModel):
    with get_openai_callback() as cb:
        assistant_message = close_notes_chatbot.invoke({'summary_text':user_message.insight_summary, 'user_query':user_message.message})
    return {"response" : assistant_message, "cost_to_generate" : cb.total_cost}
