# Engine Diagnostics Project

## Python Environment Setup

Before running the data processing, model training, or the web simulator, please install the required Python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Kaggle Data

### Install the Kaggle CLI

```bash
pip install kaggle
```

You will need to download the `kaggle.json` file from your Kaggle account settings and place it in the `~/.kaggle/` directory.

```bash
# Create the .kaggle directory
mkdir -p ~/.kaggle

# Copy the kaggle.json file to the .kaggle directory
cp ~/Downloads/kaggle.json ~/.kaggle/

# Set the permissions of the kaggle.json file
chmod 600 ~/.kaggle/kaggle.json
```

### Kaggle Dataset Details

We will be using the following dataset from Kaggle for training:

- [VED Refined 20M Engine Records (Part 1/2)](https://www.kaggle.com/datasets/yashseth25/ved-refined-20m-engine-records-part-12/data)

We will use the "part 2" dataset for testing purposes.

- [VED Failure Prediction Dataset (Part 2/2)](https://www.kaggle.com/datasets/starben/ved-refined-20m-engine-records-part-22/data)

### Installing the Data

The data should be downloaded from Kaggle and placed in the `data` directory.

```bash
# Download the data
kaggle datasets download -d yashseth25/ved-refined-20m-engine-records-part-12 -p data
kaggle datasets download -d starben/ved-refined-20m-engine-records-part-22 -p data

# Unzip the data
unzip data/ved-refined-20m-engine-records-part-12.zip -d data
unzip data/ved-refined-20m-engine-records-part-22.zip -d data
```
## Run the data preprocessing

```bash
python3 data/preprocess.py --maxfiles 100
``` 

### Run training model

```bash
python3 models/train.py
```

### Export the model to ONYX format 

```bash
python3 deployment/export_model.py
```