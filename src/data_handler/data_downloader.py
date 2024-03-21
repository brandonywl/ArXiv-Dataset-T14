import json
import requests

import os

import pandas as pd
from bs4 import BeautifulSoup
import re


from src.data_handler.file_utils import get_data_path, get_archive_path, ensure_file_folder_exists

"""
This file takes the raw downloads from Kaggle and GitHub and parses them into the formats required for the project.
Namely, there will be 5 outputs needed to support the project. Co-citations, Categories per paper, Paper details, Paper
versions, and Taxonomy of categories.

We expect the dataset downloaded from Kaggle to be named ``arxiv-metadata-oai-snapshot.json`` and the co-citation json
to be named ``internal-references-pdftotext.json``. These two needs to be in a folder named archive at the root of this
project. To download these two files, you can follow the links below or download them through the Kaggle API and wget
requests.   

Dataset: https://www.kaggle.com/datasets/Cornell-University/arxiv

GitHub Release: https://github.com/mattbierbaum/arxiv-public-datasets/releases/
GitHub File Download: https://github.com/mattbierbaum/arxiv-public-datasets/releases/download/v0.2.0/internal-references-v0.2.0-2019-03-01.json.gz
Rename the downloaded file to ``internal-references-pdftotext.json``.

All files will be exported to the data folder at the root of this project.
"""

CITATION_FILE = "internal-references-pdftotext.json"
PAPER_FILE = "arxiv-metadata-oai-snapshot.json"

METADATA_CITATIONS = "arxiv-metadata-ext-citation.csv"
METADATA_CATEGORY = "arxiv-metadata-ext-category.csv"
METADATA_PAPER = "arxiv-metadata-ext-paper.csv"
METADATA_VERSION = "arxiv-metadata-ext-version.csv"
METADATA_TAXONOMY = "arxiv-metadata-ext-taxonomy.csv"

def process_raw_downloads():
    process_citations()
    process_category()
    process_papers()
    process_version()
    process_taxonomy()


def process_citations():
    ref_file = get_archive_path(CITATION_FILE)
    data_file = get_data_path(METADATA_CITATIONS)
    ensure_file_folder_exists(data_file)

    if os.path.exists(data_file):
        print("Citations exists")
        return

    with open(ref_file) as f:
        citations = json.load(f)

    with open(data_file, "w+") as f_out:
        f_out.write("id,id_reference\n")
        for i, id in enumerate(citations):
            if i % 50000 == 0:
                print("Processing the citation records from {} to {}".format(i, i + 50000))

            for k in citations[id]:
                f_out.write(f'{id},{k}\n')

    print(
        "Finish processing all the internal citation and save it to {}".format(data_file))

def process_category():
    ref_file = get_archive_path(PAPER_FILE)
    data_file = get_data_path(METADATA_CATEGORY)
    ensure_file_folder_exists(data_file)

    if os.path.exists(data_file):
        print("Catgeory exists")
        return

    with open(data_file, "w+") as f_out:
        f_out.write("id,category_id\n")

        with open(ref_file) as f_in:
            for i, line in enumerate(f_in):
                if i % 100000 == 0:
                    print("Processing the categories of records from {} to {}...".format(i, i + 100000))

                row = json.loads(line)
                id = row["id"]
                categories = row["categories"].split()
                for c in categories:
                    f_out.write(f'"{id}","{c}"\n')
    print(
        "Finish processing the categories of records and save it to {}".format(data_file))

def process_papers():
    ref_file = get_archive_path(PAPER_FILE)
    data_file = get_data_path(METADATA_PAPER)
    ensure_file_folder_exists(data_file)

    if os.path.exists(data_file):
        print("Papers exists")
        return

    titles = []
    abstracts = []
    ids = []
    authors = []
    journal_refs = []
    licenses = []

    with open(ref_file) as f_in:
        for i, line in enumerate(f_in):
            if i % 50000 == 0:
                print("Start processing record from {} to {}...".format(i, i + 50000))

            row = json.loads(line)

            titles.append(row["title"])
            abstracts.append(row["abstract"])
            ids.append(row["id"])
            authors.append(row["authors"])
            journal_refs.append(row["journal-ref"])
            licenses.append(row["license"])

        print("Finish processing all the papers. Total records: {}".format(i))

    # the dataframe is too big and could not save to csv
    df_papers = pd.DataFrame({
        'id': ids,
        'title': titles,
        'abstract': abstracts,
        'authors': authors,
        'journal-ref': journal_refs,
        'license': licenses

    })
    df_papers.to_csv(data_file, index=False)

def process_version():
    ref_file = get_archive_path(PAPER_FILE)
    data_file = get_data_path(METADATA_VERSION)
    ensure_file_folder_exists(data_file)

    if os.path.exists(data_file):
        print("Version exists")
        return

    with open(data_file, "w+") as f_out:
        f_out.write("id,year,month\n")

        with open(ref_file) as f_in:
            for i, line in enumerate(f_in):
                if i % 100000 == 0:
                    print("Start processing the version of records from {} to {}".format(i, i + 100000))

                row = json.loads(line)
                id = row["id"]
                date_value = pd.to_datetime(row["versions"][0]['created'])
                month = date_value.month
                year = date_value.year

                f_out.write(f'{id},{year},{month}\n')

    print("Finish processing the versions of records and save it to {}".format(data_file))

def process_taxonomy():
    ref_file = get_archive_path(PAPER_FILE)
    data_file = get_data_path(METADATA_TAXONOMY)

    ensure_file_folder_exists(data_file)

    if os.path.exists(data_file):
        print("Taxonomy exists")
        return

    ## load taxonomy from https://arxiv.org/category_taxonomy
    website_url = requests.get('https://arxiv.org/category_taxonomy').text
    soup = BeautifulSoup(website_url, 'lxml')

    root = soup.find('div', {'id': 'category_taxonomy_list'})

    tags = root.find_all(["h2", "h3", "h4", "p"], recursive=True)

    level_1_name = ""
    level_2_code = ""
    level_2_name = ""

    level_1_names = []
    level_2_codes = []
    level_2_names = []
    level_3_codes = []
    level_3_names = []
    level_3_notes = []

    for t in tags:
        if t.name == "h2":
            level_1_name = t.text
            level_2_code = t.text
            level_2_name = t.text
        elif t.name == "h3":
            raw = t.text
            level_2_code = re.sub(r"(.*)\((.*)\)", r"\2", raw)
            level_2_name = re.sub(r"(.*)\((.*)\)", r"\1", raw)
        elif t.name == "h4":
            raw = t.text
            level_3_code = re.sub(r"(.*) \((.*)\)", r"\1", raw)
            level_3_name = re.sub(r"(.*) \((.*)\)", r"\2", raw)
        elif t.name == "p":
            notes = t.text
            level_1_names.append(level_1_name)
            level_2_names.append(level_2_name)
            level_2_codes.append(level_2_code)
            level_3_names.append(level_3_name)
            level_3_codes.append(level_3_code)
            level_3_notes.append(notes)

    df_taxonomy = pd.DataFrame({
        'group_name': level_1_names,
        'archive_name': level_2_names,
        'archive_id': level_2_codes,
        'category_name': level_3_names,
        'category_id': level_3_codes,
        'category_description': level_3_notes
    })

    df_taxonomy.to_csv(data_file, index=False)


