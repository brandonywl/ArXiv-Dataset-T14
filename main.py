import os
import numpy as np
import pandas as pd
import streamlit as st
from bertopic import BERTopic
import plotly.graph_objects as go
from sklearn.metrics.pairwise import pairwise_distances
from src.topic_similarity import get_intertopic_dist

st.header('Google Scholar (at home)')

model_set = set(os.listdir('models'))

@st.cache_data
def prep_df() -> pd.DataFrame:
    category_file = 'data/arxiv-metadata-ext-category.csv'
    citation_file = 'data/arxiv-metadata-ext-citation.csv'
    taxonomy_file = 'data/arxiv-metadata-ext-taxonomy.csv'
    version_file = 'data/arxiv-metadata-ext-version.csv'

    category_df = pd.read_csv(category_file, dtype={"id":object})
    citation_df = pd.read_csv(citation_file, dtype={"id":object})
    citation_count_df = citation_df.groupby('id_reference')\
                                    .count().reset_index()\
                                    .rename(columns={'id_reference':'id', 'id':'citation_count'})\
                                    .sort_values(by='citation_count', ascending=False)
    taxonomy_df = pd.read_csv(taxonomy_file)
    version_df = pd.read_csv(version_file, dtype={"id":object})
    abstract_df = pd.read_csv('data/arxiv-cs-papers-clean_abbv_casefold_punct.csv', dtype={"id":object})

    df = citation_df.merge(category_df,on="id")\
                .drop(columns=['id_reference'], axis=1)\
                .merge(taxonomy_df,on="category_id").drop_duplicates(["id","group_name"])\
                .query('`group_name`=="Computer Science"')\
                .merge(version_df[["id","year"]], on ="id")\
                .merge(abstract_df, on="id")\
                .merge(citation_count_df, how='left', on='id')

    # ids = category_df.merge(taxonomy_df, on="category_id")\
    #                 .query(f'group_name =="Computer Science"')\
    #                 .drop_duplicates(["id","group_name"], inplace=False)["id"].values
    # cits = citation_df.query('id.isin(@ids)', engine="python")\
    #                 .merge(version_df[["id","year"]], on ="id")\
    #                 .groupby(["year","id_reference"]).count()
    # cits = cits.reset_index()
    # cits.columns = ['year', 'id', 'citation_count']

    # df = cits.merge(category_df,on="id").merge(abstract_df, on="id")
                
    df['citation_count'] = df['citation_count'].fillna(0)    

    return df

def add_topic(df:pd.DataFrame, model) -> pd.DataFrame:
    df['Topic'] = model.topics_
    freq = model.get_topic_info()
    df = pd.merge(df, freq[['Topic', 'Name']], how='left', on='Topic')
    return df

base_df = prep_df()

with st.form("input_form"):
    model_name = st.selectbox(
        'Select saved model to load:',
        model_set
    )

    paper_name = st.selectbox(
        'Select arXiv paper to analyse:',
        set(base_df['title'].to_list()),
        index=None,
        placeholder='Enter arXiv research paper name'
    )

    submitted = st.form_submit_button("Submit")

## App layout
tab_list = [
    "Overview", 
    "Similar Topics", 
    "Information Retrieval"
]
overview_tab, sim_topic_tab, info_retrieval_tab = st.tabs(tab_list)

if submitted:
    # model = BERTopic.load(f"models/bertopic - 2024:03:25:00:43:23")
    model = BERTopic.load(f"models/{model_name}")
    df = add_topic(base_df, model)

with overview_tab:
    if submitted:
        query = df.query(f'`title`=="{paper_name}"')
        topic_num = query['Topic'].iloc[0]
        st.subheader(f'Topic identified:')
        st.write(query["Name"])
        st.write(query)

        st.subheader(f'Top 5 papers recommendations from identified topic:')
        paper_rank_df = df.query(f'`title`!="{paper_name}"')
        paper_rank_df = paper_rank_df.query(f'`Topic`=={topic_num}')\
                            .sort_values(['citation_count'], ascending=False).head()
        st.write(paper_rank_df)

with sim_topic_tab:
    if submitted:
        st.subheader(f'Similar topics you might be interested in:')
        intertopic_dist = get_intertopic_dist(model, topics = None, top_n_topics = None, custom_labels = False)
        
        pairwise_dist = pairwise_distances(intertopic_dist[['x','y']].to_numpy())
        ind = np.argpartition(pairwise_dist[0], 6)[:6] # top 6 argument of smallest distance
        other_topic = np.delete(ind, [0]).tolist() # remove 0-th element as it is the pairwise distance of topic against itself

        sim_paper_rank = df.query('Topic.isin(@other_topic)', engine="python")\
                            .sort_values(['citation_count'], ascending=False).head()
        st.write(sim_paper_rank)

