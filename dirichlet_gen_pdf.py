import pickle
import numpy as np
from rdkit import Chem
from datetime import datetime
import time
from molecular_dataset import MolecularDataset
from  data_smiles.drugbank import genPkl

import argparse
from trainer_debug import Trainer
import matplotlib.pyplot as plt
import copy
import pprint
import utils

import os

import difflib
import datetime

import io
from os import listdir
import torch

import traceback
import sys
import json

import git
from git import Repo
import statistics
import math
from collections import deque

from trainer_test import getArgs
from trainer_explorer import genDatasetSplits
from collections import defaultdict
import random
from trainer_explorer import create_path_if_not_exists
from check_iid import create_clients_dirichlet
from sqlitedict import SqliteDict
import dirichlet
import check_iid
from trainer_explorer import genDatasetSplits
from contextlib import redirect_stdout

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--formatted_date', type=str, default='25-02-25_01-39-30')
  parser.add_argument('--alphaKey', type=str, default='5')
  args = parser.parse_args()
  print( "sys.argv (before reset)", sys.argv )
  # sys.argv = []
  sys.argv = sys.argv[:1]

  non_iid_clients_dict = SqliteDict( "non_iid_clients_dict_by_alpha_datetime.sqlite" )
  # alphaKey = str(alpha)
  print( "alphaKey", args.alphaKey )
  print( "formatted_date", args.formatted_date )
  if args.alphaKey in non_iid_clients_dict:
    non_iid_clients_dict_alpha = non_iid_clients_dict[args.alphaKey]
    for k in non_iid_clients_dict_alpha.keys():
      if args.formatted_date in k:
        non_iid_clients = non_iid_clients_dict_alpha[ k ]
        args_trainer_test = getArgs()
        print( "", non_iid_clients.keys() )
        # uMol_data_dirs = genDatasetSplits(args_trainer_test.num_users)
        # labels, mols, smiles, atomicNumRepresentative = check_iid.check_iid(args_trainer_test)
        trap = io.StringIO()
        with redirect_stdout(trap):
          # labels, mols, smiles, atomicNumRepresentative, nMolsNaUsers, lModelsDataDataDict = \
          #   dirichlet.check_iid_uMol_data_dirs(args_trainer_test, uMol_data_dirs)
          naMolsDict = dirichlet.getNaMolsDict( non_iid_clients, args.alphaKey, args_trainer_test )
          # dirichlet.getStackedSubplots(naMolsDict, k)
          # dirichlet.getStackedSubplotsExcl78(naMolsDict, k)
          dirichlet.getStackedSubplots2xCol(naMolsDict, k)
