import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "data/"
OUTPUT_DIR = "outputs/"


def save_plot(history, filename):
    """Generates and saves the plot to output file

    Parameters
    ----------
        history: keras callbacks.History object
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df = pd.DataFrame(history)
    df.plot(subplots=True, grid=True, figsize=(10, 15))
    plt.savefig(os.path.join(OUTPUT_DIR, filename))


def save_data(dataset, filename, directory=DATA_DIR):
    """Saves the data to csv

    Parameters
    ----------
        dataset: dict

    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    pd.DataFrame(dataset).to_csv(os.path.join(directory, filename))


def get_data(filename):
    """Returns data from csv

    Parameters
    ----------
        filename: string, name of the file to sent data to

    Returns
    -------
        ret: pandas.Dataframe
    """
    ret = pd.read_csv(os.path.join(DATA_DIR, filename))
    return ret
