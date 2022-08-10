# Exoplanet Exploration

This repository contains the source code for a research project regarding the detection of anomalous light curves indicative of exoplanets using an unsupervised machine learning approach. 

## Installation

Download [anaconda](https://docs.anaconda.com/anaconda/install/) with python version 3.9.12 or later.

Using [conda](https://docs.anaconda.com/anaconda/install/), create a virtual environment with the necessary libraries using the following command ```conda create --name <env> --file requirements.txt``` like so:

```bash
conda create --name tensorflow_env --file requirements.txt
```
Use the virtual environment you created by using the following command ```conda activate <env>``` or by directly select the virtual environment for usage through the [Anaconda-Navigator](https://docs.anaconda.com/anaconda/navigator/) application.

> **TODO: Show Anaconda-Navigator page and where to select a virtual environment**

## Usage

The project is separated into three primary modules:

1. `ae-download-rawdata.py`: downloads light curve data using the MAST API using the `tois_latest.csv` file. Requires a directory to download data to; defaults to local folder.
2. `ae-simulate-trainingdata.py`: various functions that work to create a two datasets for training and testing, with labels for both if supervised learning is desired. Saves all generated data locally using [pickle](https://docs.python.org/3/library/pickle.html#:~:text=%E2%80%9CPickling%E2%80%9D%20is%20the%20process%20whereby,back%20into%20an%20object%20hierarchy.).
3. `ae-test-autoencoder.py`: downloads training and testing datasets from the **pickle** file. Trains an autoencoder and returns the results.

It is recommended to run the files in order, as each program will require data from the previous program. However, since data is saved locally, preceding programs only need to be run once or twice before the current program can be run as many times as desired.

### Downloading Raw Data
Download the data to current directory:
```bash
python ae-download-rawdata.py
```
Download the data to a desired directory:
```bash
python ae-download-rawdata.py -o /usr/../dir

# or 

python ae-download-rawdata.py --outdir /usr/../dir
```

### Simulating Data
Simulating data requires the directory where the raw data was downloaded as input:
```bash
python ae-simulate-trainingdata.py -i /usr/../dir

# or

python ae-simulate-trainingdata.py --indir /usr/../dir
```

### Autoencoder
Autoencoder requires no input as it downloads from the pickle files saved locally:
```bash
python ae-test-autoencoder.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)