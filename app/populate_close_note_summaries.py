from summarise_close_notes_utils import get_close_notes, summarise_close_notes
from langchain_community.callbacks import get_openai_callback
import os

DATA_PATH_W_SUMMARY = "../data/Closed_Incident_Data_w_Summaries/"

def main():
    df_close = get_close_notes()

    with get_openai_callback() as cb:
        summaries = summarise_close_notes(df_close)
    print("Cost to summarise all provided close notes:")
    print(cb)

    df_close.loc[:,'close_notes_summary'] = summaries
    df_close.to_csv(os.path.join(DATA_PATH_W_SUMMARY, "summarised_data.csv"))

if __name__ == "__main__":
    main()