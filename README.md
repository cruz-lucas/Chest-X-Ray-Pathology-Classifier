# Chest X-Ray Pathology Classifier

Lucas Cruz's final project for Electronics and Computer Engineering undergraduate course at Universidade Federal do Rio de Janeiro.

## Usage
### **1. Environment** (Optional)

An exemple of building the Docker container can be found in the Makefile, and executed in command line:
```
make docker_build
```
And run the container using:
```
make docker_run
```

### **2. Installation** (If not using the environment)
```
pip install -r requirements.txt
```

### **3. Data**

To download the data, you should go to the CheXpert's [webpage](https://stanfordmlgroup.github.io/competitions/chexpert/), fillout the Use Agreement and wait for the email with the download link.

Once downloaded, unzip folder to path:
> ./data/raw/


## Components
- Data Loader

    The data loader component wraps a custom pytorch's dataset class, returning a batched pytorch's dataloader. You may choose the location where CheXpert's data was unzipped, the method/policy to handle uncertainty, the pathologies to classify, wether it's the train or validation set, and wether it's the downsampled dataset or not. The dataloader wrapper also accepts some of dataloader class parameters, as: batch size, number of workers and memory pinning. The default transformations to the data are: resize, scale pixel values between [0, 1] and normalization with dataset mean and standard deviation.
 
- Trainer
- Model

## Project Organization


    ├── LICENSE
    ├── Makefile                <- Makefile with commands like `make data` or `make train`
    ├── README.md               <- The top-level README for developers using this project.
    ├── data
    │   ├── external            <- Data from third party sources.
    │   ├── interim             <- Intermediate data that has been transformed.
    │   ├── processed           <- The final, canonical data sets for modeling.
    │   └── raw                 <- The original, immutable data dump.
    │       └── CheXpert-v1.0
    │           ├── train       <- Train data, with structure according to metadata.
    │           ├── valid       <- Valid data, with structure according to metadata.
    │           ├── train.csv
    │           └── valid.csv
    │
    ├── docs                    <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                  <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks               <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                              the creator's initials, and a short `-` delimited description, e.g.
    │                              `1.0-jqp-initial-data-exploration`.
    │
    ├── references              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures             <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
    │                              generated with `pip freeze > requirements.txt`
    │
    ├── setup.py                <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                     <- Source code for use in this project.
    │   ├── __init__.py         <- Makes src a Python module
    │   │
    │   ├── data                <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features            <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models              <- Scripts to train models and then use trained models to make
    │   │   │                      predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization       <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini                 <- tox file with settings for running tox; see tox.readthedocs.io



## Authors

- [@lucasdearaujocruz](https://github.com/lucasdearaujocruz)
