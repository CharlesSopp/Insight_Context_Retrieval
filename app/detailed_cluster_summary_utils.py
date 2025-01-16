import pandas as pd
from datetime import datetime
from clustering_utils import summarise_signature_close_notes

SUMMARY_DATA_PATH = "../data/Closed_Incident_Data_w_Summaries/summarised_data.csv"

def get_detailed_summary(insight_ids, clustering_model = "Agglomerative", reduced_dim = False):

    df_close = pd.read_csv(SUMMARY_DATA_PATH, index_col=0)

    df_relevant_close = df_close[df_close['Number'].isin(insight_ids)]
    df_useful_relevant_close = df_relevant_close[~df_relevant_close['close_notes_summary'].str.lower().str.contains('no relevant information provided')]

    most_recently_resolved_date = df_relevant_close.sort_values(by=['Resolved'], ascending=False).iloc[0]['Resolved']
    most_recently_resolved_date = datetime.strptime(most_recently_resolved_date, "%Y-%m-%d %H:%M:%S").strftime("%B %d, %Y")
    least_recently_resolved_date = df_relevant_close.sort_values(by=['Resolved'], ascending=True).iloc[0]['Resolved']
    least_recently_resolved_date = datetime.strptime(least_recently_resolved_date, "%Y-%m-%d %H:%M:%S").strftime("%B %d, %Y")

    cluster_results = summarise_signature_close_notes(df_useful_relevant_close, clustering_model, reduced_dim)
    assigned_to_info = summarise_assigned_to_info(df_relevant_close)
    cluster_and_assigned_to_summaries = get_cluster_and_assigned_to_summaries(cluster_results)

    response = f'''Found {len(df_relevant_close)} previously closed similar incidents, resolved between {least_recently_resolved_date} to {most_recently_resolved_date}. 

These similar insights were assigned to the following people: {assigned_to_info}. 

Of these previously closed similar incidents, there are {len(df_useful_relevant_close)} close notes detailing actions taken to resolve the incidents. A summary of the different resolution methods mentioned in these close notes is given in the following buller points: {cluster_and_assigned_to_summaries}'''
    return response

def summarise_cluster_assigned_to_info(df_assigned_to):
    summary_list = []
    for index, row in df_assigned_to.iterrows():
        min_date = row['min']
        max_date = row['max']
        if min_date == max_date:
            date_range = f"on {min_date}"
        else:
            date_range = f"between {min_date} - {max_date}"
        summary_string = f"{row['Assigned to']} {row['count']} time(s) {date_range}"
        summary_list.append(summary_string)
    return ", ".join(summary_list)

def get_cluster_and_assigned_to_summaries(cluster_results):
    summary_list = []
    for cluster_summary in cluster_results.groupby(['grouped_cluster_summary'])['Resolved'].agg('max').sort_values(ascending=False).index:
        df_assigned_to = cluster_results[cluster_results['grouped_cluster_summary'] == cluster_summary].groupby("Assigned to")['Resolved'].agg(['max', 'min', 'count']).reset_index()
        resolvers = summarise_cluster_assigned_to_info(df_assigned_to)
        summary_list.append(f"\n  - '{cluster_summary}' was described in {df_assigned_to['count'].sum()} incident close notes. These incidents were resolved by the following people: {resolvers}")
    return "".join(summary_list)

def summarise_assigned_to_info(df_relevant_close):
    df_grouped = df_relevant_close[['Assigned to', 'Resolved']].groupby("Assigned to").agg({"Assigned to":'count', "Resolved" : ['min', 'max']}).reset_index().sort_values(by=('Assigned to', 'count'), ascending=False)
    summary_list = []
    for index, row in df_grouped.iterrows():
        min_date = row[('Resolved', 'min')]
        max_date = row[('Resolved', 'max')]
        if min_date == max_date:
            date_range = f"on {min_date}"
        else:
            date_range = f"between {min_date} - {max_date}"
        summary_string = f"\n  - {row['Assigned to', '']} {row['Assigned to', 'count']} time(s) {date_range}"
        summary_list.append(summary_string)
    return "".join(summary_list)