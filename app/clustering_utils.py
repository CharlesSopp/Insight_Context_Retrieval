from sklearn.cluster import OPTICS, AgglomerativeClustering, KMeans, DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import optuna
from optuna.samplers import TPESampler

from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from sentence_transformers import SentenceTransformer
from operator import itemgetter
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import time
import os

load_dotenv("../.env")
AZURE_API_KEY = os.getenv('AZURE_API_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT')

SUMMARY_DATA_PATH = "../data/Closed_Incident_Data_w_Summaries/summarised_data.csv"
emb_model = SentenceTransformer('all-mpnet-base-v2')

def get_summary_from_clustering(insight_ids, clustering_model = "Agglomerative", reduced_dim = False):

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

    cluster_results = summarise_signature_close_notes(df_useful_relevant_close, clustering_model, reduced_dim)
    df_sum = cluster_results.groupby(['grouped_cluster_summary'])['Resolved'].agg(['min', 'max', 'count']).reset_index().sort_values(by='count', ascending=False)
    text_summary = "\n  - ".join(f"{row['grouped_cluster_summary']} (Mentioned {row['count']} time(s), from {row['min']} to {row['max']})" for index, row in df_sum.iterrows())

    if len(df_useful_relevant_close) > 0:
        response = f'''Found {len(df_relevant_close)} previously closed similar incidents, resolved between {least_recently_resolved_date} to {most_recently_resolved_date}. These similar insights are most commonly assigned to {most_commonly_assigned_to} ({assigned_to_counts.max()} time(s)), and most recently assigned to {most_recently_assigned_to}. Of these previously closed similar incidents, there are {len(df_useful_relevant_close)} close notes detailing actions taken to resolve the incidents. A summary of these close notes is provided in the following text: \n  - {text_summary}.'''
    else:
        response = "No relevant close notes found."
    
    return response

llm = AzureChatOpenAI(
    azure_deployment='gpt-4o',
    api_version="2024-08-01-preview",
    temperature=0.3,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
)

def summarise_signature_close_notes(df, clustering_model, reduced_dim = False):
    df = df.copy()
    print("LOADING DATAFRAME")
    print(f"Dataframe contains {len(df)} close notes.")

    print("\nCLUSTERING CLOSE NOTE SUMMARIES")
    cluster_labels, cluster_summaries = cluster_close_note_summaries(df, clustering_model, reduced_dim)

    df.loc[:,'cluster_labels'] = cluster_labels
    df.loc[:,'cluster_summary'] = df.apply(cluster_summary, axis=1, cluster_summaries=cluster_summaries)

    print(f"Found {len(cluster_summaries)} clusters from {len(df)} close notes")

    print("\nCLUSTERING CLOSE NOTE SUMMARISES FURTHER USING LLM")
    summaries, movements  = llm_further_clustering(cluster_summaries)

    print(f"Number of clusters has been reduced from {len(cluster_summaries)} to {len(summaries)}")
    print("\nGETTING FINAL CLUSTER SUMMARIES USING LLM")
    final_summaries = final_cluster_summarisation(summaries)

    df.loc[:,'grouped_cluster_label'] = df.apply(grouped_cluster_label, axis=1, movements=movements)
    df.loc[:,'grouped_cluster_summary'] = df.apply(grouped_cluster_summary, axis=1, final_summaries=final_summaries)

    return df

PROMPT = """
You are an AI assistant tasked with analyzing network incident summaries. 
Given an incoming network incident summary, and a corpus of network incident summaries, determine whether any of the summaries in the corpus describe the same resolution steps to fix the issue as the incoming summary.
The descriptions may not be identical, but should refer to the same underlying actions or causes.
If there is a summary in the corpus which refers to the same actions as the incoming summary, respond with the summary number (integer only). Otherwise, respond with 'No matching summaries'.

Incoming summary: {incoming_summary}
Corpus of summaries: {summaries}
Answer:
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT)

def format_docs(summaries):
    return "\n ----- \n".join(f"{k}. {max(summaries[k], key=len)}" for k in summaries.keys())

match_summaries = (
    {
        "summaries" : (itemgetter('summaries')
                           |
                     RunnableLambda(format_docs)),
         "incoming_summary" : itemgetter('incoming_summary')
    }
      |
   prompt_template
      |
   llm
      |
   StrOutputParser()
)

PROMPT2 = """
You are an AI assistant tasked with analyzing network incident summaries which describe similar resolution steps taken to fix a network incident.
The summaries provided may not be identical, but will refer to the same underlying actions or causes.
Summarise the provided collection of network incident summaries in as few words as possible.

Incident Summaries: {summaries}
Answer:
"""

prompt_template2 = ChatPromptTemplate.from_template(PROMPT2)

def format_docs2(summaries):
    return "\n ----- \n".join(f"{s}" for s in summaries)

final_summary = (
    {
        "summaries" : (itemgetter('summaries')
                           |
                     RunnableLambda(format_docs2)),
    }
      |
   prompt_template2
      |
   llm
      |
   StrOutputParser()
)

def llm_further_clustering(cluster_summaries):
    count_clusters = len(cluster_summaries)
    summaries = {}
    movements = {}

    for original_group_no in list(cluster_summaries.keys()):
        summary = cluster_summaries[original_group_no]
        if len(summaries) == 0:
            summaries[1] = [summary]
            movements[1] = [original_group_no]
        else:
            complete = False
            while not complete:
                try:
                    print(f"Passing cluster {original_group_no} / {count_clusters} to LLM...")
                    summary_response = match_summaries.invoke({'incoming_summary':summary,'summaries':summaries})
                    print(f"Success")
                    complete = True
                except:
                    print('failed, sleeping for 60 seconds')
                    time.sleep(60)

            if summary_response == 'No matching summaries':
                summaries[len(summaries)+1] = [summary]
                movements[len(summaries)] = [original_group_no]
            else:
                summaries[int(summary_response)].append(summary)
                movements[int(summary_response)].append(original_group_no)

    return summaries, movements

def final_cluster_summarisation(summaries):
    final_summaries = {}

    for key in list(summaries.keys()):
        summary = summaries[key]
        if len(summary) == 1:
            final_summaries[key] = summary[0]
            print(f"Completed summary for cluster {key}")
        else:
            final_summaries[key] = final_summary.invoke({'summaries' : summary})
            print(f"Completed summary for cluster {key}")

    return final_summaries


def cluster_summary(row, cluster_summaries):
    return cluster_summaries[row['cluster_labels']]

def cluster_close_note_summaries(df, clustering_model, reduced_dim = False):
    df = df.copy()
    
    close_note_summaries = list(df['close_notes_summary'])
    embeddings = emb_model.encode(close_note_summaries)
    
    optimiser = ClusteringOptimiser(embeddings, reduced_dim)
    best_trial = optimiser.optimise(clustering_model, n_trials=200)
    cluster_labels = optimiser.get_labels(best_trial, clustering_model)

    df['labels'] = cluster_labels
    df['len_close_note_summaries'] = df['close_notes_summary'].str.len()

    cluster_summaries = {}
    for label in cluster_labels:
        df_label = df[df['labels'] == label]
        summary = final_summary.invoke({'summaries' : list(df_label['close_notes_summary'])})
        #summary = df_label.sort_values(by='len_close_note_summaries', ascending=True)['close_notes_summary'].iloc[int(len(df_label)/2)]
        cluster_summaries[label] = summary

    return cluster_labels, cluster_summaries

class ClusteringOptimiser():
    def __init__(self, X, reduced_dim = False):
        """
        Initializes the ClusteringOptimiser class with data and optional dimensionality reduction.
        
        Args:
            X (array-like): The embedded sentences to be clustered.
            reduced_dim (bool): Whether to apply dimensionality reduction before clustering (default: False).
        """
        self.X = X
        self.models = {
            'KMeans' : KMeans,
            'DBSCAN' : DBSCAN,
            'Agglomerative' : AgglomerativeClustering,
            'OPTICS' : OPTICS
        }
        self.tried_params = []
        self.reduced_dim = reduced_dim

    def _get_model(self, model_name):
        """
        Fetches the corresponding clustering model class based on the model name.
        
        Args:
            model_name (str): The name of the clustering model.
        
        Returns:
            model_class (class): The model class corresponding to the model_name.
        """
        return self.models.get(model_name, None)
    

    def _get_model_params(self, model_name, trial):
        """
        Retrieves hyperparameters for a given model from Optuna trials.
        
        Args:
            model_name (str): The name of the clustering model.
            trial (optuna.trial.Trial): The trial object containing suggested hyperparameters.
        
        Returns:
            params (dict): A dictionary of suggested hyperparameters for the model.
            n_components (int or None): The number of components for dimensionality reduction (if using).
        """
        if model_name == 'KMeans':
            params = {
                'n_clusters' : trial.suggest_int("n_clusters", 2,len(self.X)-1),
                'random_state' : 42
            }
        elif model_name == 'DBSCAN':
            #Using k-nearest-neigbbors to dynamically set range of eps to test against
            neighbors = NearestNeighbors(n_neighbors=2, metric='euclidean')
            neighbors_fit = neighbors.fit(self.X)
            distances, indices = neighbors_fit.kneighbors(self.X)
            distances = distances[:,1]
            nonzero_distances = distances[distances>0]
            min_eps = nonzero_distances.min()
            max_eps = nonzero_distances.max()

            params =  {
                'eps' : trial.suggest_float("eps", min_eps, max_eps),
                'min_samples' : 1
            }
        elif model_name == 'Agglomerative':
            params = {
                'n_clusters' : trial.suggest_int("n_clusters", 2,len(self.X)-1)
            }
        elif model_name == 'OPTICS':
            params = {
                'xi' : trial.suggest_float('xi', 0, 1),
                'cluster_method' : 'xi'
            }
        else:
            params = {}
        
        if self.reduced_dim == True:
            n_components = trial.suggest_int("n_components", 1,min(len(self.X),10))
        else:
            n_components = None

        return params, n_components
    
    
    def _get_eval_metric(self, labels):
        """
        Evaluates the clustering performance using a taillored metric compromising between silhouette score and the number of clusters.
        
        Args:
            labels (array): The cluster labels assigned to each point.
        
        Returns:
            score (float): A score representing the clustering performance. Larger scores correspond to better performance.
        """

        num_unclustered = (labels == -1).sum()

        if num_unclustered > 0:
            num_clusters = num_unclustered + len(set(labels)) - 1
        else:
            num_clusters = len(set(labels))

        if (len(set(labels)) > 1) & (len(set(labels)) < len(self.X)):
            sil_score = silhouette_score(self.X, labels)
        else:
            sil_score = -1

        score = sil_score - (0.5*(num_clusters / len(self.X)))
        
        return score
    
    def objective(self, trial, model_name):
        """
        The objective function for optimizing the clustering model's hyperparameters using Optuna.
        
        Args:
            trial (optuna.trial.Trial): The trial object used to suggest hyperparameters.
            model_name (str): The name of the clustering model.
        
        Returns:
            score (float): The evaluation score based on silhouette score and cluster count.
        """
        model_class = self._get_model(model_name)

        if model_class is None:
            raise ValueError(f"Model {model_name} is not supported")
        
        params, n_components = self._get_model_params(model_name, trial)

        if params in self.tried_params:
            return float('-inf')
        self.tried_params.append(params)
        
        if n_components is None:
            pipeline = model_class(**params)
        else:
            pipeline = Pipeline([
                ('pca', PCA(n_components=n_components)),
                ('cluster_model', model_class(**params))
            ])

        labels = pipeline.fit_predict(self.X)
        score = self._get_eval_metric(labels)

        return score
    
    def detailed_objective(self, trial, model_name):
        """
        A detailed objective function to return silhoutte score and the number of clusters for the best trial.
        
        Args:
            trial (optuna.trial.Trial): The trial object used to suggest hyperparameters.
            model_name (str): The name of the clustering model.
        
        Returns:
            score (float): The evaluation score based on silhouette score and cluster count.
        """
        model_class = self._get_model(model_name)

        if model_class is None:
            raise ValueError(f"Model {model_name} is not supported")
        
        params, n_components = self._get_model_params(model_name, trial)
        
        if n_components is None:
            pipeline = model_class(**params)
        else:
            pipeline = Pipeline([
                ('pca', PCA(n_components=n_components)),
                ('cluster_model', model_class(**params))
            ])

        labels = pipeline.fit_predict(self.X)

        num_unclustered = (labels == -1).sum()

        if num_unclustered > 0:
            num_clusters = num_unclustered + len(set(labels)) - 1
        else:
            num_clusters = len(set(labels))

        if len(set(labels)) > 1:
            sil_score = silhouette_score(self.X, labels)
        else:
            sil_score = -1

        return sil_score, num_clusters
    
    def get_labels(self, trial, model_name):
        """
        A detailed objective function to return silhoutte score and the number of clusters for the best trial.
        
        Args:
            trial (optuna.trial.Trial): The trial object used to suggest hyperparameters.
            model_name (str): The name of the clustering model.
        
        Returns:
            score (float): The evaluation score based on silhouette score and cluster count.
        """
        model_class = self._get_model(model_name)

        if model_class is None:
            raise ValueError(f"Model {model_name} is not supported")
        
        params, n_components = self._get_model_params(model_name, trial)
        
        if n_components is None:
            pipeline = model_class(**params)
        else:
            pipeline = Pipeline([
                ('pca', PCA(n_components=n_components)),
                ('cluster_model', model_class(**params))
            ])

        labels = pipeline.fit_predict(self.X)

        return labels
    
    def optimise(self, model_name, n_trials=50):
        """
        Runs the optimization process using Optuna to find the best hyperparameters for the specified model.
        
        Args:
            model_name (str): The name of the clustering model to optimize.
            n_trials (int): The number of optimization trials to run (default: 50).
        
        Returns:
            best_trial (optuna.trial.FrozenTrial): The best trial with the optimal hyperparameters.
        """
        study = optuna.create_study(sampler=TPESampler(seed=42), direction="maximize")
        study.optimize(lambda trial: self.objective(trial, model_name), n_trials=n_trials)
                
        return study.best_trial
    
    def optimise_with_seed(self, model_name, seed, n_trials=50):
        """
        Runs the optimization process using Optuna to find the best hyperparameters for the specified model.
        
        Args:
            model_name (str): The name of the clustering model to optimize.
            n_trials (int): The number of optimization trials to run (default: 50).
        
        Returns:
            best_trial (optuna.trial.FrozenTrial): The best trial with the optimal hyperparameters.
        """
        study = optuna.create_study(sampler=TPESampler(seed=seed), direction="maximize")
        study.optimize(lambda trial: self.objective(trial, model_name), n_trials=n_trials)
                
        return study.best_trial

def grouped_cluster_label(row, movements):
    cluster_label = row['cluster_labels']
    for key in list(movements.keys()):
        if cluster_label in movements[key]:
            return key
        
def grouped_cluster_summary(row, final_summaries):
    return final_summaries[row['grouped_cluster_label']]