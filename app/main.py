from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from query_chromadb_utils import query_insights
from summarise_previous_actions_utils import get_close_note_summary, get_all_close_notes
from clustering_utils import get_summary_from_clustering
from langchain_community.callbacks import get_openai_callback
import optuna

SIMILARITY_THRESHOLD = 0.15

app = FastAPI()

class QuerySimilarInsights(BaseModel):
    incoming_eventNames: List[str]
    incoming_eventObjects: List[str]
    incoming_eventCIs: List[str]

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI App for retrieving insight context from similar, previously closed, network incidents"}

@app.post("/get_similar_insight_ids")
def get_similar_insight_ids_endpoint(request: QuerySimilarInsights):
    result = list(query_insights(request.incoming_eventNames, request.incoming_eventObjects, request.incoming_eventCIs, threshold=SIMILARITY_THRESHOLD)['id'])
    return {"similar_insight_ids": result}

@app.post("/get_summary_of_incidents")
def get_summary_of_incidents_endpoint(request: QuerySimilarInsights):
    similar_ids = list(query_insights(request.incoming_eventNames, request.incoming_eventObjects, request.incoming_eventCIs, threshold=SIMILARITY_THRESHOLD)['id'])
    with get_openai_callback() as cb:
        response = get_close_note_summary(similar_ids)
    return {"summary_of_incidents" : response, "cost_to_generate" : cb.total_cost}

@app.post("/get_summary_of_clustered_incidents")
def get_summary_of_clustered_incidents_endpoint(request: QuerySimilarInsights):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    similar_ids = list(query_insights(request.incoming_eventNames, request.incoming_eventObjects, request.incoming_eventCIs, threshold=SIMILARITY_THRESHOLD)['id'])
    with get_openai_callback() as cb:
        response = get_summary_from_clustering(similar_ids)
    return {"summary_of_incidents" : response, "cost_to_generate" : cb.total_cost}

@app.post("/get_all_similar_closed_incidents")
def get_all_similar_closed_incidents_endpoint(request: QuerySimilarInsights):
    query_df = query_insights(request.incoming_eventNames, request.incoming_eventObjects, request.incoming_eventCIs, threshold=SIMILARITY_THRESHOLD)
    response = get_all_close_notes(query_df)
    return {"summary_of_incidents" : response.to_dict(orient="records")}

