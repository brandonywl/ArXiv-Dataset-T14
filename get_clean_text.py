import os

from src.data_handler.data_downloader import process_papers
from src.data_handler.dataloader import load_raw_cs_papers, load_cached_tokens, load_version
from src.data_handler.file_utils import get_data_path, get_cache_path, ensure_file_folder_exists

from src.preprocessor import Preprocessor

import pickle


# If files are not processed (i.e. not available in root/data/ yet, then process them)
def check_processed_state():
    METADATA_CITATIONS = "arxiv-metadata-ext-citation.csv"
    METADATA_CATEGORY = "arxiv-metadata-ext-category.csv"
    METADATA_PAPER = "arxiv-metadata-ext-paper.csv"
    METADATA_VERSION = "arxiv-metadata-ext-version.csv"
    METADATA_TAXONOMY = "arxiv-metadata-ext-taxonomy.csv"

    files_to_check = [METADATA_CITATIONS, METADATA_CATEGORY, METADATA_PAPER, METADATA_VERSION, METADATA_TAXONOMY]
    files_to_check = [get_data_path(file) for file in files_to_check]

    processed_state = sum([os.path.exists(file) for file in files_to_check]) == len(files_to_check)
    return processed_state

# If they are, load cs_papers
def load_cs_papers(raw_suffix="", suffix=None, run_preprocessor=False, filter_2019_up=True, **kwargs):
    if "to_exclude" in kwargs:
        to_exclude = kwargs['to_exclude']
    else:
        to_exclude = ['tokenize']

    columns_interested = ['abstract'] if 'columns_interested' not in kwargs else kwargs['columns_interested']

    if "cs_papers" in kwargs:
        cs_papers = kwargs["cs_papers"]
    else:
        cs_papers = load_raw_cs_papers(raw_suffix)
        print("cs papers loaded")

    if filter_2019_up:
        print("Filtering to only papers before 2019 (not inclusive)")
        versions = load_version()
        papers_pre_2019 = versions[versions['year'] < 2019]
        cs_papers = cs_papers.merge(papers_pre_2019, how='inner', on='id')
    
    if run_preprocessor:
        preprocessor = Preprocessor(to_exclude, **kwargs)
        cs_papers = preprocessor.execute(cs_papers, columns_interested=columns_interested)

    if suffix is not None:
        file_name = f"arxiv-cs-papers-{suffix}"
        if type(cs_papers) == dict:
            file_name = file_name + ".pickle"
            file_path = get_cache_path(file_name)
            ensure_file_folder_exists(file_path)
            with open(file_path, "wb") as f:
                pickle.dump(cs_papers, f)
        else:
            file_name = file_name + ".csv"
            file_path = get_data_path(file_name)
            ensure_file_folder_exists(file_path)
            cs_papers.to_csv(file_path, index=False)

    return cs_papers

if __name__ == "__main__":
    # # Parses the dataset into a pd dataframe without normalization steps
    cs_papers = load_cs_papers("")

    # Loads the parsed/cached from "", normalizes the data, and then stores it at "-normalized"
    # load_cs_papers("", "normalized", run_preprocessor=True)

    # Another sample of loading to do preprocessing, but using the output from the previous loading
    load_cs_papers("", "clean_abbv_casefold_punct", run_preprocessor=True, cs_papers=cs_papers)

    # Load file ran at normalized if exists, else extract from the raw data. Runs preprocessor for only tokenize
    load_cs_papers("clean_abbv_casefold_punct", "clean_abbv_casefold_punct_spacytoken", run_preprocessor=True,
                   to_include=['tokenize'])

    ## Loads the tokenized output
    # cached_tokens = load_cached_tokens("clean_abbv_casefold_punct_spacytoken")