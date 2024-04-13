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

## Modules ##
### KMeans classification and recommendation ###
- Due to the big size of trained data, it is not put into this repository. You could download the trained KMeans model, word2vec, and Tf-idf models from the Google Drive:<br>https://drive.google.com/drive/u/1/folders/1SkgvFmTrqfRzHRMJZQcy557MsSpkdnGl. Put those downloaded files into sub folder "model_data" under project path.

- Running KMeans classification needs to open `kmeans_classify.ipynb`.<br>
There are three parameters in the Notebook. They will control:<br>
a. `FAST_RUN_WITH_EXISTING_DATA`: whether re-train the model or just load model from existing pickle file<br>
b. `NORMALIZE_VECTOR`: whether normalize word2vec result<br>
c. `USE_TFIDF`: whether using tf-idf or word2vec to vectorize words and docs<br> 

Example:<br>
```
FAST_RUN_WITH_EXISTING_DATA = True
USE_TFIDF = True
```
The program will load the existing models and use Tf-idf to vectorize words<br><br>

```
FAST_RUN_WITH_EXISTING_DATA = False
USE_TFIDF = False
NORMALIZE_VECTOR = True
```
The program will re-train the models, using word2vec to vectorize words, and normalize the vectorized words. The newly trained model will be put into sub-folder "model_data" and replace the previous models. <br>
- After setting the above parameters, run the cells in sequence. <br>

