#!/usr/bin/python3

# generate the script to download lightcurves

import argparse
import os
import pandas as pd
import requests
from time import sleep

tess_url = "https://exo.mast.stsci.edu/api/v0.1/dvdata/tess/"

def get_args():
  """
  Get user input args (i.e. output dir to hold all downloaded lightcurve json files)
  """
  parser = argparse.ArgumentParser(
    description="Download lightcurve data from TESS",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-o", "--outdir", help="directory to store meta and data output")
  args = vars(parser.parse_args())

  if args['outdir']:
    return args['outdir'] + '/'
  else:
    return ''


def download_tce(outdir, id, tce_entry):
  """
  Download TCE lightcurve data into a {id}.json file
  """
  fname = f"{outdir}{id}-{sector}-{tce}-data.json"
  if os.path.exists(fname):
    print(f"skip {fname}")
    return

  sector, tce = tce_entry.split(":")
  data_query = f"{tess_url}{id}/table/?tce={tce}&sector={sector}"
  print(f"downloading {data_query} ...")
  response = requests.get(data_query)
  open(fname, "wb").write(response.content)
  print(f"done {fname}")

outdir = get_args()

delay_count = 0
filtered = pd.read_csv('tois_latest.csv',usecols=['TIC ID'])
for i, row in filtered.iterrows():
  id = row['TIC ID'].item()
  response = requests.get(f"{tess_url}/{id}/tces/")
  tce_dict = response.json()

  if len(tce_dict['TCE']) == 0:
    continue

  # loop through each TCE array entry and download the actual json data
  arr = tce_dict['TCE']
  for ea in arr:
    download_tce(outdir, id, ea)
    delay_count = delay_count + 1
  
  if delay_count % 10 == 0:
    sleep(1)
