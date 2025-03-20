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

# https://github.com/prabu-ram/Non-IID-data-distibution-Federeated-Learning/blob/main/Non_IID_Performance_Federated/Federated_base.py
def create_clients_dirichlet(labels, mols, smiles, alpha=0.5):
  print( "labels", len(labels) )
  print( "mols", len(mols) )
  print( "smiles", len(smiles) )
  # https://numpy.org/doc/stable/reference/generated/numpy.unique.html
  num_clients = 3
  # num_classes = len( counterDict.keys() )
  label_array = np.array(labels)
  unique_labels, label_indices = np.unique(label_array, return_inverse=True)
  num_classes = len(unique_labels)
  class_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)

  print( "type( class_distribution )", type( class_distribution ) )
  print( "class_distribution.shape", class_distribution.shape )
  # print( "class_distribution", class_distribution )
  print( "len( label_indices )", len( label_indices ) )
  non_iid_clients = defaultdict(list)
  for class_idx, class_dist in enumerate(class_distribution):
    # print("class_idx", class_idx)
    # print("np.where(label_indices == class_idx)[0].shape", np.where(label_indices == class_idx)[0].shape )
    class_data_indices = np.where(label_indices == class_idx)[0]
    random.shuffle(class_data_indices)
    # print( "len(class_data_indices)", len(class_data_indices) )
    # continue
    N = len(class_data_indices)
    split_indices = (np.cumsum(class_dist) * N).astype(int)[:-1]
    # print( "class_idx, split_indices", class_idx, split_indices )
    class_split = np.array_split(class_data_indices, split_indices)
    for client_idx, indices in enumerate(class_split):
      # print( "client_idx, len(indices)", client_idx, len(indices) )
      for idx in indices:
        non_iid_clients[ client_idx ].append( smiles[idx] )
  print( "non_iid_clients.keys()", list( non_iid_clients.keys() ) )
  for cid,smiles in non_iid_clients.items():
    print( "cid, len(mols)", cid, len(smiles) )
  return non_iid_clients

def getMolDicts():
  filenames = ['data-smiles/qm8_smiles.pkl','data_smiles/drugbank/diabetes-drugbank.pkl']
  mols = []
  labels = [] # labels = na = number of atoms
  molsDictBySmile = {}
  smiles = []
  smilesByNA = {}
  counterDict = {}
  atomicNumRepresentative = {} # atom.GetAtomicNum()
  for filename in filenames:
    with open(filename, 'rb') as f:
      for i,line in enumerate( pickle.load(f) ):
        mol = Chem.MolFromSmiles(line)
        smile = Chem.CanonSmiles( Chem.MolToSmiles(mol) )
        # smiles.append( smile )
        # mols.append( mol )
        if not mol:
          print( i, filename )
        if '5' == smile:
          print( "line, smile:", line, smile )
        molsDictBySmile[ smile ] = mol
        na = mol.GetNumAtoms()
        if na < 33:
          labels.append( na )
          mols.append( mol )
          smiles.append( smile )
          for atom in mol.GetAtoms():
            an = atom.GetAtomicNum()
            if an not in atomicNumRepresentative:
              atomicNumRepresentative[an] = smile
        if na not in counterDict:
          counterDict[na] = []
        counterDict[na].append( (mol, (line,i,filename) ) )
        # for atom in mol.GetAtoms():
        #   an = atom.GetAtomicNum()
        #   if an not in atomicNumRepresentative:
        #     atomicNumRepresentative[an] = smile
  # print( "len( counterDict.keys() )", len( counterDict.keys() ) )
  # print( "counterDict.keys()", list( counterDict.keys() ) )
  counterDictKeys = list( counterDict.keys() )
  for na in counterDictKeys:
    # if mol.GetNumAtoms() < 33:
    if na >= 33:
      counterDict.pop( na )
  print( "len( smiles )", len( smiles ) )
  print( "len( counterDict.keys() )", len( counterDict.keys() ) )
  print( "counterDict.keys()", list( counterDict.keys() ) )
  print( "atomicNumRepresentative",  )
  print( pprint.pformat( atomicNumRepresentative , indent=2, sort_dicts=False) )
  print( "counterDict[1]", len( counterDict[1] ) )
  for mol,smileT in counterDict[1]: # T: tupe
    print( "mol in counterDict[1]", Chem.CanonSmiles( Chem.MolToSmiles( mol ) ), smileT, type( mol ) )
  pass
  # return counterDict, atomicNumRepresentative
  return labels, mols, smiles, atomicNumRepresentative

def check_iid(args):
  num_users = args.num_users
  # if args.isFL:
  uMol_data_dirs = genDatasetSplits(num_users)

  mols = []
  labels = [] # labels = na = number of atoms
  smiles = []
  atomicNumRepresentative = {} # atom.GetAtomicNum()

  lModelsDict = {}
  lModelsDataDataDict = {}
  for u in range(num_users):
    counterDict = {}
    lModel = Trainer(args=args, data=None, idxs=None, mol_data_dir=uMol_data_dirs[u])
    lModelsDict[u] = lModel
    print( pprint.pformat( lModel.__dict__.keys() , indent=2, sort_dicts=False) )
    # print( "lModel.D", lModel.D )
    print( "len( lModel.data.smiles )", len( lModel.data.smiles ) )
    print( "len( lModel.data.data )", len( lModel.data.data ) )
    print( "type( lModel.data.data )", type( lModel.data.data[0] ) )
    for mol in lModel.data.data:
      na = mol.GetNumAtoms()
      if na not in counterDict:
        counterDict[na] = []
      counterDict[na].append(mol)

      smile = Chem.CanonSmiles( Chem.MolToSmiles(mol) )
      labels.append( na )
      mols.append( mol )
      smiles.append( smile )
      for atom in mol.GetAtoms():
        an = atom.GetAtomicNum()
        if an not in atomicNumRepresentative:
          atomicNumRepresentative[an] = smile

    # for na,mlist in counterDict.items():
    #   print( na, len(mlist) )
    lModelsDataDataDict[u] = counterDict

  for i in range( 33 ):
    nMolsPerNaPerUsers = []
    for u in range(num_users):
      # nMolsPerNaPerUsers = []
      if i in lModelsDataDataDict[u]:
        nMolsPerNaPerUsers.append( len( lModelsDataDataDict[u][i] ) )
      else:
        nMolsPerNaPerUsers.append( 0 )
    print( i, nMolsPerNaPerUsers )
  return labels, mols, smiles, atomicNumRepresentative

# molecularGAN\GraphGANFed\trainer_explorer.py
def genDatasetSplitsNoniid(non_iid_clients, atomicNumRepresentative, mols, num_users=3):
  dataRoorFolder = 'data_smiles/noniid/'
  splits = []
  for cid,smiles in non_iid_clients.items():
    splits.append( str( len( smiles ) ) )
  splitFolderName = '-'.join( ['split'] + splits )
  splitDSFolder = os.path.join( dataRoorFolder, splitFolderName )
  splitDSFolder = os.path.join( splitDSFolder, str(num_users))
  print( splitDSFolder )
  create_path_if_not_exists( splitDSFolder )
  for mol in mols:
    if mol.GetNumAtoms() == 32:
       mol32 = Chem.CanonSmiles( Chem.MolToSmiles(mol) )
  uSmilesLists = []
  for u in range(num_users):
    uSmilesLists.append([])

  pathList = []
  for u in range(num_users):
    uDSFilename_pkl = os.path.join( splitDSFolder, str(u) + '.pkl' )
    print( "len( atomicNumRepresentative )", len( atomicNumRepresentative ) )
    for k,v in atomicNumRepresentative.items():
      uSmilesLists[u].append(v)
    uSmilesLists[u].append(mol32)
    for smile in non_iid_clients[u]:
      uSmilesLists[u].append( smile )

    print( "len(uSmilesLists[u])", len(uSmilesLists[u]) )
    with open(uDSFilename_pkl, 'wb') as f:
      pickle.dump( np.array(uSmilesLists[u]) , f)

    data = MolecularDataset()
    data.generate( uDSFilename_pkl, validation=0.1, test=0.1 ) # data_smiles\\esol_smiles.pkl
    data.save( uDSFilename_pkl + '.dataset' )
    pathList.append(uDSFilename_pkl + '.dataset')
  return pathList
  pass

if __name__ == '__main__':
  args = getArgs()
  labels, mols, smiles, atomicNumRepresentative = check_iid(args)
  print( "len( atomicNumRepresentative )", len( atomicNumRepresentative ), "check_iid original FL datasets" )
  # labels, mols, smiles, atomicNumRepresentative = getMolDicts()
  # print( "len( atomicNumRepresentative )", len( atomicNumRepresentative ), "getMolDicts from pkl numpy array of smile strings" )
  non_iid_clients = create_clients_dirichlet(labels, mols, smiles)
  pathList = genDatasetSplitsNoniid(non_iid_clients, atomicNumRepresentative, mols)
  print( "pathList", pathList )
