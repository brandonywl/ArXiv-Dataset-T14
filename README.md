# ArXiv-Dataset-T14

This project is done in fulfilment of CS5246 Text Mining. We are focusing on the open project option where we are looking to do knowledge extraction from the ArXiv Dataset provided by Cornell University.

The dataset can be found at https://www.kaggle.com/datasets/Cornell-University/arxiv

## Setup
1. Clone this repository first and download the dataset from the link above.

2. For co-citation data, download the [co-citation json](https://github.com/mattbierbaum/arxiv-public-datasets/releases/download/v0.2.0/internal-references-v0.2.0-2019-03-01.json.gz). Unzip the file and ensure it is named ```internal-references-pdftotext.json```.

3. Put the two json files into a folder ```archive``` at the root level.

4. You can run the following code snippet to just process the downloaded data into segmented chunks. It will then store it at ```./data/arxiv-cs-papers.csv```
    ```
        load_cs_papers("")
    ```

Or you could also define pre-processing steps to be run. This snippet stores it at ```./data/arxiv-cs-papers-normalized.csv```
```
    load_cs_papers("", "normalized", run_preprocessor=True)
```