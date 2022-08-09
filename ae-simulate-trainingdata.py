#!/usr/bin/python3

# attempts to create an autoencoder

# used anaconda with a tensorflow env
# additional packages downloaded:
# astroquery, tqdm

import numpy as np
import pandas as pd
import math
import random

from astroquery.mast import Tesscut
from astropy.table import Table

import argparse
import json
import os
import re
import pickle
from tqdm import tqdm

# training_data_creation and test_data_creation are deprecated functions and are preserved solely for archival reasons
# use data_set_creation instead
training_data_label = []
def training_data_creation(indir, end):
    """
    Creates a testing dataset with 2% of the data being not synthetic

    Parameters
    ----------
    indir : directory containing the lightcurve data
        Of the form '/User/.../'
    end : index of file to end at

    Returns
    -------
    response : list of light curves
        Of the form `list` of `list` of `tensorflow.python.framework.ops.EagerTensor`
    training_data_label : list of labels for the light curve data (directly edits the global list)
        Of the form `list`
    """

    training_data = []
    num = 0
    max_light_curve_length = 0
    for file in tqdm(os.listdir(indir)):
        if num == end:
            break
        num = num + 1

        # grabbing light curve file and removing NaN values.
        file = indir+file
        with open(file) as json_file:
            file_content = json.load(json_file)
        detrended_light_flux = json_to_table(file_content['fields'],file_content['data'])['LC_INIT'].data.tolist()
        detrended_light_flux = list(filter(lambda x: not math.isnan(x), detrended_light_flux))

        # 2% chance to pass on an anomalous light curve (labelled 1), 98% chance to simulate the light curve (labelled 0).
        if random.randint(0, 49) == 0:
            training_data_label.append(1)
        else:
            detrended_light_flux = synthesize_data(detrended_light_flux)
            training_data_label.append(0)
        
        # normalizing the data
        first_quartile = np.quantile(detrended_light_flux, 0.25)
        third_quartile = np.quantile(detrended_light_flux, 0.75)
        iqr = third_quartile-first_quartile
        median = np.quantile(detrended_light_flux, 0.5)

        detrended_light_flux = (detrended_light_flux - median) / iqr
        
        # appending the final product to the list, checking max length
        training_data.append(detrended_light_flux)
        if (len(detrended_light_flux) > max_light_curve_length):
            max_light_curve_length = len(detrended_light_flux)

    # if the training data is not the max size, backfill the lightcurve with values
    training_data_final = []
    for light_curve in training_data:
        if (len(light_curve) == max_light_curve_length):
            continue
        else:
            # previous method was backfilling with the very last item. we are trying an alternate form here
            # light_curve = [*light_curve, *[light_curve[-1]]*(max_light_curve_length-len(light_curve))]
            for i in range(0, max_light_curve_length-len(light_curve)-1):
                np.append(light_curve, light_curve[i])
        training_data_final.append(light_curve)
    return training_data_final

test_data_label = []
def test_data_creation(indir, begin, end):
    """
    Creates a testing dataset with a third of the data being not synthetic

    Parameters
    ----------
    indir : directory containing the lightcurve data
        Of the form '/User/.../'
    begin : index of file to begin at (usually the one after the training_data_creation)
    end : index of file to end at

    Returns
    -------
    response : list of light curves
        Of the form `list` of `list` of `tensorflow.python.framework.ops.EagerTensor`
    test_data_label : list of labels for the light curve data (directly edits the global list)
        Of the form `list`
    """
    test_data = []
    num = 0
    max_light_curve_length = 0
    for file in tqdm(os.listdir(indir)):
        if num == end:
            break
        elif num < begin:
            continue
        num = num + 1

        # grabbing light curve file and removing NaN values.
        file = indir+file
        with open(file) as json_file:
            file_content = json.load(json_file)
        detrended_light_flux = json_to_table(file_content['fields'],file_content['data'])['LC_INIT'].data.tolist()
        detrended_light_flux = list(filter(lambda x: not math.isnan(x), detrended_light_flux))

        # 33% chance to pass on an anomalous light curve (labelled 1), 66% chance to simulate the light curve (labelled 0).
        if random.randint(0, 2) == 0:
            test_data_label.append(1)
        else:
            detrended_light_flux = synthesize_data(detrended_light_flux)
            test_data_label.append(0)
        
        # normalizing the data
        first_quartile = np.quantile(detrended_light_flux, 0.25)
        third_quartile = np.quantile(detrended_light_flux, 0.75)
        iqr = third_quartile-first_quartile
        median = np.quantile(detrended_light_flux, 0.5)

        detrended_light_flux = (detrended_light_flux - median) / iqr

        # appending the final product to the list, checking max length
        test_data.append(detrended_light_flux)
        if (len(detrended_light_flux) > max_light_curve_length):
            max_light_curve_length = len(detrended_light_flux)

    # if the test data is not the max size, backfill the lightcurve with values
    test_data_final = []
    for light_curve in test_data:
        if (len(light_curve) == max_light_curve_length):
            continue
        else:
            # previous method was backfilling with the very last item. we are trying an alternate form here
            # light_curve = [*light_curve, *[light_curve[-1]]*(max_light_curve_length-len(light_curve))]
            for i in range(0, max_light_curve_length-len(light_curve)-1):
                np.append(light_curve, light_curve[i])
        test_data_final.append(light_curve)
    return test_data_final

def get_curves():
    """
    Gets location of light curve files using -i
    """

    parser = argparse.ArgumentParser(
        description="Download lightcurve data from TESS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--indir", help="directory for light curve data")
    args = vars(parser.parse_args())

    if args['indir']:
        return args['indir'] + '/'
    else:
        return ''

def json_to_table(fields, data):
    """"
    Takes a json object and turns it into an astropy table.

    Parameters
    ----------
    fields : list of dicts
        Of the form [{colname:,datatype:,description:}, ...]
    data : list of dicts
       Of the form [{col1:, col2:, ...},{col1:, col2:, ...}, ...]

    Returns
    -------
    response : `astropy.table.Table`
    """

    rx = re.compile(r"varchar\((\d+)\)")
    
    data_table = Table()

    for col, atype in [(x['colname'], x['datatype']) for x in fields]:
        col = col.strip()
        if "varchar" in atype:
            match = rx.search(atype)
            atype = "U" + match.group(1)
        if atype == "real":
            atype = "float"
        data_table[col] = np.array([x.get(col, None) for x in data], dtype=atype)

    return data_table

def synthesize_data(detrended_light_curve):
    """
    Synthesizes a light curve by removing anomalous light flux

    Parameters
    ----------
    detrended_light_curve : light curve file that has been detrended
        Of the form `list`
    
    Returns
    -------
    response : light curve file with outliers removed and the holes interpolated over
        of the form `pandas.core.series.Series`
    """
    
    first_quartile = np.quantile(detrended_light_curve, 0.25)
    third_quartile = np.quantile(detrended_light_curve, 0.75)
    iqr = third_quartile-first_quartile

    high_cutoff = third_quartile + 1.5 * iqr
    low_cutoff = first_quartile - 1.5 * iqr

    median_light_curve = np.array(detrended_light_curve).astype(float)
    median_light_curve = median_light_curve[(detrended_light_curve <= high_cutoff) 
                                            & (detrended_light_curve >= low_cutoff)]
    median_light_curve = pd.Series(median_light_curve).interpolate(method="polynomial", order=1).tolist()
    
    return median_light_curve

def data_set_creation(indir, training_data_end, test_data_end, max_light_curve_length=36414, training_data_percent=0.02, test_data_percent=0.33):
    """
    Creates a training and testing dataset

    Parameters
    ----------
    indir : directory containing the lightcurve data
        Of the form `/User/.../`
    
    training_data_end : index of file to stop generating training data at (inclusive)
        Of the form `int`

    test_data_end : index of file to stop generating test data at (inclusive). Has to be bigger than training_data_end
        Of the form `int`
    
    max_light_curve_length : maximum length the light curve file is allowed. Used to reduce computation time and is set at the third quartile
        Of the form `int`
    
    training_data_percent : percentage of training data that will be anomalous (set at 2%)
        Of the form `float`

    test_data_percent : percentage of test data that will be anomalous (set at 33%)
        Of the form `float`

    Returns
    -------
    response : training data set of light curves
        Of the form `list` of `list` of `tensorflow.python.framework.ops.EagerTensor`

    response 2 : test data set of light curves
        Of the form `list` of `list` of `tensorflow.python.framework.ops.EagerTensor`

    response 3 : list of labels for training data set
        Of the form `list`

    response 4 : list of labels for test data set
        Of the form `list`
    """

    # if training data endpoint exceeds test data endpoint, switch the two
    if training_data_end > test_data_end:
        temp = test_data_end
        test_data_end = training_data_end
        training_data_end = temp

    uncorrected_training_data = []
    uncorrected_test_data = []

    training_data_label = []
    test_data_label = []

    num = 0
    TRAINING_DATA = True

    for file in tqdm(os.listdir(indir)):
        if (num == test_data_end):
            print("Done creating data")
            break
        elif (num >= training_data_end):
            TRAINING_DATA = False
        num = num + 1

        # grabbing light curve file and removing NaN values
        file = indir+file
        with open(file) as json_file:
            file_content = json.load(json_file)
        detrended_light_curve = json_to_table(file_content['fields'],file_content['data'])['LC_INIT'].data.tolist()
        detrended_light_curve = list(filter(lambda x: not math.isnan(x), detrended_light_curve))

        # chooses to synthesize the data or not depending on the percent values, and appropriately labelling as so
        # additionally chooses to synthesize all data above the third quartile in length
        if TRAINING_DATA:
            if random.random() < training_data_percent and len(detrended_light_curve) < max_light_curve_length:
                training_data_label.append(1)
            else:
                detrended_light_curve = synthesize_data(detrended_light_curve)

                if len(detrended_light_curve) > max_light_curve_length:
                    detrended_light_curve = detrended_light_curve[:max_light_curve_length]

                training_data_label.append(0)
        elif not TRAINING_DATA:
            if random.random() < test_data_percent and len(detrended_light_curve) < max_light_curve_length:
                test_data_label.append(1)
            else:
                detrended_light_curve = synthesize_data(detrended_light_curve)
                if (len(detrended_light_curve) > max_light_curve_length):
                    detrended_light_curve = detrended_light_curve[:max_light_curve_length]
                test_data_label.append(0)
        
        # normalizes the data
        first_quartile = np.quantile(detrended_light_curve, 0.25)
        third_quartile = np.quantile(detrended_light_curve, 0.75)
        iqr = third_quartile-first_quartile
        median = np.quantile(detrended_light_curve, 0.5)

        detrended_light_curve = (detrended_light_curve - median) / iqr
        
        # appending the final product to the list, checking max length
        if TRAINING_DATA:
            uncorrected_training_data.append(detrended_light_curve)
        else:
            uncorrected_test_data.append(detrended_light_curve)
        
        # deprecated code
        # if (len(detrended_light_curve) > max_light_curve_length):
        #     max_light_curve_length = len(detrended_light_curve)
    print("Max light curve length: ", max_light_curve_length)

    # if the training data is not the max size, backfill the lightcurve with values
    final_training_data = []
    print("Checking training data")
    for light_curve in tqdm(uncorrected_training_data):
        if not (len(light_curve) == max_light_curve_length):
            # previous method was backfilling with the very last item. we are trying an alternate form here
            # light_curve = [*light_curve, *[light_curve[-1]]*(max_light_curve_length-len(light_curve))]
            for i in range(0, max_light_curve_length-len(light_curve)):
                light_curve = np.append(light_curve, np.array(light_curve[i]))
        final_training_data.append(light_curve)
    
    # if the test data is not the same size as training data, backfill it as well
    final_test_data = []
    print("Checking test data")
    for light_curve in tqdm(uncorrected_test_data):
        if not (len(light_curve) == max_light_curve_length):
            # previous method was backfilling with the very last item. we are trying an alternate form here
            # light_curve = [*light_curve, *[light_curve[-1]]*(max_light_curve_length-len(light_curve))]
            for i in range(0, max_light_curve_length-len(light_curve)):
                # np.append(light_curve, light_curve[i])
                light_curve = np.append(light_curve, np.array(light_curve[i]))
        final_test_data.append(light_curve)
    
    print("Done checking training and test data")
    return final_training_data, final_test_data, training_data_label, test_data_label

def light_curve_length_counter(indir):
    """
    Measures the length of light curve files

    Parameters
    ----------
    indir : directory containing the lightcurve data
        Of the form `/User/.../`

    Returns
    -------
    response : list of all the light curves' lengths
        Of the form `list`
    """
    light_curve_lengths = []
    for file in tqdm(os.listdir(indir)):
        file = indir+file
        with open(file) as json_file:
            file_content = json.load(json_file)
        detrended_light_curve = json_to_table(file_content['fields'],file_content['data'])['LC_INIT'].data.tolist()
        detrended_light_curve = list(filter(lambda x: not math.isnan(x), detrended_light_curve)) 
        light_curve_lengths.append(len(detrended_light_curve))
    return light_curve_lengths


# TODO: Before boarding the code to github, delete this line of code and use get_curves() instead
indir = get_curves()
training_data, test_data, training_data_label, test_data_label = data_set_creation(indir, 300, 330)
print("training data length: ", len(training_data))
print("test data length: ", len(test_data))
print("training label length: ", len(training_data_label))
print("test data length: ", len(test_data_label))

"""
light_curve_lengths = light_curve_length_counter(indir)
print("Median Light Curve Length", np.quantile(light_curve_lengths, 0.5))
print("Third Quartile Light Curve Length", np.quantile(light_curve_lengths, 0.75))

import matplotlib.pyplot as plt
plt.hist(light_curve_lengths)
"""
# median: 17222.5, third quartile: 36414.0, plot max: ~400 000
# conclusion: most light curves around 20 000 in length, with a few enormous outliers
# plan: remove any light curves above 36k in length, or force them to be simulated and cut them off after 36k

# saving simulated data locally to reduce computation times and make model training more consistent
print("Dumping training data")
pickle.dump(training_data, open("train.p", "wb"))

print("Dumping test data")
pickle.dump(test_data, open("test.p", "wb"))

print("Labels for the data")
pickle.dump(training_data_label, open("train_label.p", "wb"))
pickle.dump(test_data_label, open("test_label.p", "wb"))

print("All done")

"""
There's no such thing as a final product
"""