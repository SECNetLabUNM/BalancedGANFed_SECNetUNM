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

def check_iid_uMol_data_dirs(args, uMol_data_dirs):
  num_users = args.num_users
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

  nMolsPerNaPerUsersDict = {}
  for i in range( 33 ):
    nMolsPerNaPerUsers = []
    for u in range(num_users):
      # nMolsPerNaPerUsers = []
      if i in lModelsDataDataDict[u]:
        nMolsPerNaPerUsers.append( len( lModelsDataDataDict[u][i] ) )
      else:
        nMolsPerNaPerUsers.append( 0 )
    print( i, nMolsPerNaPerUsers )
    nMolsPerNaPerUsersDict[i] = nMolsPerNaPerUsers
  return labels, mols, smiles, atomicNumRepresentative, nMolsPerNaPerUsersDict, lModelsDataDataDict

def adjustIidNaStatisitcs(nMolsNaUsers, lModelsDataDataDict, mols, atomicNumRepresentative, args):
  mol32 = None
  for mol in mols:
    if mol.GetNumAtoms() == 32:
       # mol32 = Chem.CanonSmiles( Chem.MolToSmiles(mol) )
       mol32 = mol

  for u in range(args.num_users):
    na = mol32.GetNumAtoms()
    if not na in lModelsDataDataDict[u]:
      lModelsDataDataDict[u][na] = []
    lModelsDataDataDict[u][na].append(mol32)
    nMolsNaUsers[na][u] +=1
    for k,v in atomicNumRepresentative.items():
      print( "v, MolFromSmiles", type(v), type( Chem.MolFromSmiles(v) ) )
      mol = Chem.MolFromSmiles(v)
      na = mol.GetNumAtoms()
      # lModelsDataDataDict[u][na].append( Chem.MolFromSmiles(v) )
      if not na in lModelsDataDataDict[u]:
        lModelsDataDataDict[u][na] = []
      lModelsDataDataDict[u][na].append( mol )
      nMolsNaUsers[na][u] +=1
    print( "lModelsDataDataDict[u]", u, len(lModelsDataDataDict[u]) )
    # print( "nMolsNaUsers[u]", u, nMolsNaUsers[u] )
  print( "nMolsNaUsers", nMolsNaUsers )
  totalNumMols = 0
  for k,v in nMolsNaUsers.items():
    print( "nMolsNaUsers", k,v, sum(v) )
    totalNumMols += sum(v)
  print( "totalNumMols", totalNumMols )

def getNaStatisitcs(non_iid_clients, atomicNumRepresentative, mols, num_users=3):
  for cid,smiles in non_iid_clients.items():
    print( "cid,smiles in non_iid_clients:", cid,len(smiles) )
  pass
  for mol in mols:
    if mol.GetNumAtoms() == 32:
       mol32 = Chem.CanonSmiles( Chem.MolToSmiles(mol) )
  uSmilesLists = []
  for u in range(num_users):
    uSmilesLists.append([])

  lModelsDataDataDict = {}
  # nMolsPerNaPerUsersDict = {}

  for u in range(num_users):
    counterDict = {}
    for k,v in atomicNumRepresentative.items():
      uSmilesLists[u].append(v)
    uSmilesLists[u].append(mol32)
    for smile in non_iid_clients[u]:
      uSmilesLists[u].append( smile )
    for smile in uSmilesLists[u]:
      mol = Chem.MolFromSmiles(smile)
      na = mol.GetNumAtoms()
      if na not in counterDict:
        counterDict[na] = []
      counterDict[na].append(mol)
    lModelsDataDataDict[u] = counterDict
    pass

  nMolsPerNaPerUsersDict = {}
  for i in range( 33 ):
    nMolsPerNaPerUsers = []
    for u in range(num_users):
      # nMolsPerNaPerUsers = []
      if i in lModelsDataDataDict[u]:
        nMolsPerNaPerUsers.append( len( lModelsDataDataDict[u][i] ) )
      else:
        nMolsPerNaPerUsers.append( 0 )
    nMolsPerNaPerUsersDict[i] = nMolsPerNaPerUsers

  return [nMolsPerNaPerUsersDict, lModelsDataDataDict]

import plotUtils
#import time_series_plotter
from matplotlib import rcParams
#rcParams.update(time_series_plotter.params)
labelsizeInt = 13
params = {
   'axes.labelsize': labelsizeInt,
   'axes.labelweight': "bold",
   'axes.titlesize': labelsizeInt,
   'axes.titleweight': "bold",
   'font.size': labelsizeInt,
   'font.weight': "bold",
   'legend.fontsize': labelsizeInt,
   'xtick.labelsize': labelsizeInt,
   'ytick.labelsize': labelsizeInt,
   'text.usetex': False,
   'figure.figsize': [6*1.5, 5*1.5],
   }
rcParams.update(params)

# caller: naMolsDict = dirichlet.getNaMolsDict( non_iid_clients, args.alphaKey, args_trainer_test, args.existingdatasetid )
# args.existingdatasetid: existingdatasetid
def getNaMolsDict(non_iid_clients, alphaKey, args, existingdatasetid, num_users=3):
  uMol_data_dirs_list = {}
  uMol_data_dirs_list["iid"] = genDatasetSplits(args.num_users)
  # uMol_data_dirs_list["nonIid-0.5"] = [
  #   'data_smiles/noniid/split-9355-8744-3874/3/0.pkl.dataset', 
  #   'data_smiles/noniid/split-9355-8744-3874/3/1.pkl.dataset', 
  #   'data_smiles/noniid/split-9355-8744-3874/3/2.pkl.dataset']
  v = getEntry_uMol_data_dirs_list(splitText=existingdatasetid)
  if len(v) >0:
    uMol_data_dirs_list["nonIid-0.5"] = v
  else:
    print( "len(v)<=0", "getEntry_uMol_data_dirs_list", existingdatasetid )
    sys.exit()
  labelsEtcDict = {}
  naMolsDict = {}
  for datakey,uMol_data_dirs in uMol_data_dirs_list.items():
    labels, mols, smiles, atomicNumRepresentative, nMolsNaUsers, lModelsDataDataDict = check_iid_uMol_data_dirs(args, uMol_data_dirs)
    labelsEtcDict[ datakey ] = [labels, mols, smiles, atomicNumRepresentative]
    naMolsDict[ datakey ] = [nMolsNaUsers, lModelsDataDataDict]
    print( "len( labels )", datakey, len( labels ) )
  labels, mols, smiles, atomicNumRepresentative = labelsEtcDict[ "iid" ]
  nMolsNaUsers, lModelsDataDataDict = naMolsDict[ "iid" ]
  # alpha = 5
  # alphaKey = str(alpha)
  naMolsDict[ "nonIid-" + alphaKey ] = getNaStatisitcs(non_iid_clients, atomicNumRepresentative, mols)
  nMolsNaUsers, lModelsDataDataDict = naMolsDict[ "iid" ]
  adjustIidNaStatisitcs(nMolsNaUsers, lModelsDataDataDict, mols, atomicNumRepresentative, args)
  pass
  return naMolsDict

uSpacingCoeff = 1.1
def getStackedSubplots(naMolsDict, formatted_date, num_users=3):
  plt.clf()
  plt.close()
  fig, axs = plt.subplots( len(naMolsDict), sharex=True )
  plt.subplots_adjust(hspace=0.3)
  colors=plotUtils.getColorsCmap('tab20', 33 + 1)
  for iAx,kv in enumerate( naMolsDict.items() ):
    k,v = kv
    nMolsNaUsers, lModelsDataDataDict = v
    axs[iAx].set_title(k)
    axs[iAx].set_title(k)

    plt.sca(axs[iAx])
    u = np.arange(0, num_users)
    uActual = np.arange(0, num_users) * uSpacingCoeff
    plt.yticks( uActual, u )

    for u in range(num_users):
      axs[iAx].barh(u, nMolsNaUsers[1][u], color=colors[1])
      for i in range( 2,33 ):
        left = np.sum([nMolsNaUsers[k][u] for k in range(i)])
        axs[iAx].barh(u*uSpacingCoeff, nMolsNaUsers[i][u], left=left, color=colors[i])
  axs[num_users-1].set(xlabel='number of mols')
  axs[num_users-1].set(ylabel='client id')
  plt.savefig("fedgan5/img/StackedSubplots." + formatted_date +".pdf", format='pdf', bbox_inches='tight', dpi=720)
  pass

def getStackedSubplotsExcl78(naMolsDict, formatted_date, num_users=3):
  plt.clf()
  plt.close()
  fig, axs = plt.subplots( len(naMolsDict), sharex=True )
  plt.subplots_adjust(hspace=0.3)
  fig.tight_layout() # https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots
  colors=plotUtils.getColorsCmap('tab20', 33 + 1)
  for iAx,kv in enumerate( naMolsDict.items() ):
    k,v = kv
    nMolsNaUsers, lModelsDataDataDict = v
    axs[iAx].set_title(k)
    # https://stackoverflow.com/questions/19626530/how-to-set-xticks-in-subplots
    plt.sca(axs[iAx])
    print( "range(num_users), range(num_users*2, 2)", range(num_users), range(num_users*2, 2) )
    print( "list(range(num_users)), list(range(num_users*2, 2))", list(range(num_users)), list(range(0, num_users*2, 2)) )
    u = np.arange(0, num_users)
    uActual = np.arange(0, num_users) * uSpacingCoeff
    plt.yticks( uActual, u )
    for u in range(num_users):
      axs[iAx].barh(u, nMolsNaUsers[1][u], color=colors[1])
      nMolsCum = nMolsNaUsers[1][u]
      for i in range( 2,33 ):
        if i == 7 : continue
        if i == 8 : continue
        # print(i, colors[i])
        axs[iAx].barh(u*uSpacingCoeff, nMolsNaUsers[i][u], left=nMolsCum, color=colors[i])
        nMolsCum += nMolsNaUsers[i][u]
  axs[num_users-1].set(xlabel='number of mols')
  axs[num_users-1].set(ylabel='client id')
  plt.savefig("fedgan5/img/StackedSubplots." + formatted_date +"-Excl78.pdf", format='pdf', bbox_inches='tight', dpi=720)
  pass

def getStackedSubplots2xCol(naMolsDict, formatted_date, num_users=3):
  plt.clf()
  plt.close()
  # fig, axs = plt.subplots( len(naMolsDict), 2, sharex=True )
  hatches = ["", "*"]
  hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
  lh = len(hatches) # length of hatches
  extraTexts = ["IID", "β=0.5", "β=5"]
  handles = []
  labels = []
  fig, axs = plt.subplots( len(naMolsDict), 2 )
  plt.subplots_adjust(hspace=0.3)
  colors=plotUtils.getColorsCmap('tab20', 33 + 1)
  for iAx,kv in enumerate( naMolsDict.items() ):
    k,v = kv
    nMolsNaUsers, lModelsDataDataDict = v
    # axs[iAx,0].set_title(k)
    # axs[iAx,0].set_title(k)

    plt.sca(axs[iAx,0])
    u = np.arange(0, num_users)
    uActual = np.arange(0, num_users) * uSpacingCoeff
    plt.yticks( uActual, u )

    if not iAx ==0: axs[iAx,0].sharex(axs[0, 0])
    axs[iAx,0].set(ylabel='\n'.join( [extraTexts[iAx], 'client id' ] ) )
    for u in range(num_users):
      # axs[u,0].set(ylabel='client id')
      axs[iAx,0].barh(u, nMolsNaUsers[1][u], color=colors[1])
      for i in range( 2,33 ):
        left = np.sum([nMolsNaUsers[k][u] for k in range(i)])
        # print( "hatches[i%2]", hatches[i%2], i )
        edgecolor=colors[i]
        if i in [ 5,6,7,8] : edgecolor='white'
        l = axs[iAx,0].barh(u*uSpacingCoeff, nMolsNaUsers[i][u], left=left, hatch=hatches[i%lh],edgecolor=edgecolor, color=colors[i], label=str(i))
        if iAx ==0 and u==0 :
          handles.append(l[0])
          labels.append(str(i))
          print( type(u), u )
          print( type(l), len(l) )
          print( type(l[0]) )
          print( vars(l[0]) )
          print( "hatches[i%2]", hatches[i%2], i )
    # lh = len(hatches) # length of hatches
    # for i, bar in enumerate(axs[iAx,0].patches):
    #   bar.set_hatch(hatches[i%lh])
  axs[num_users-1,0].set(xlabel='number of mols')
  # axs[num_users-1].set(ylabel='client id')

  for iAx,kv in enumerate( naMolsDict.items() ):
    k,v = kv
    nMolsNaUsers, lModelsDataDataDict = v
    # axs[iAx,1].set_title(k)
    # https://stackoverflow.com/questions/19626530/how-to-set-xticks-in-subplots
    plt.sca(axs[iAx,1])
    print( "range(num_users), range(num_users*2, 2)", range(num_users), range(num_users*2, 2) )
    print( "list(range(num_users)), list(range(num_users*2, 2))", list(range(num_users)), list(range(0, num_users*2, 2)) )
    u = np.arange(0, num_users)
    uActual = np.arange(0, num_users) * uSpacingCoeff
    plt.yticks( uActual, u )
    if not iAx ==0: axs[iAx,1].sharex(axs[0, 1])
    for u in range(num_users):
      axs[iAx,1].barh(u, nMolsNaUsers[1][u], color=colors[1])
      nMolsCum = nMolsNaUsers[1][u]
      for i in range( 2,33 ):
        if i == 7 : continue
        if i == 8 : continue
        # print(i, colors[i])
        edgecolor=colors[i]
        if i == 5 or i == 6 : edgecolor='white'
        axs[iAx,1].barh(u*uSpacingCoeff, nMolsNaUsers[i][u], left=nMolsCum, hatch=hatches[i%lh],edgecolor=edgecolor, color=colors[i])
        nMolsCum += nMolsNaUsers[i][u]
  axs[num_users-1,1].set(xlabel='number of mols')
  # axs[num_users-1].set(ylabel='client id')
  # fig.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure, ncol=4)
  fig.legend(handles=handles, labels=labels, title='#atoms per mol',
    bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure, ncol=7,
    loc='upper center',
    )
  plt.savefig("fedgan5/img/StackedSubplots2xCol." + formatted_date +".pdf", format='pdf', bbox_inches='tight', dpi=720)
  pass

def getProbDistrNaCategories(naMolsDict, num_users=3):
  # v: [nMolsNaUsers, lModelsDataDataDict]
  uColors = ['b', 'g', 'r']
  for k,v in naMolsDict.items():
    # print( "k, len(nMolsNaUsers), len(lModelsDataDataDict)", k, len(nMolsNaUsers), len(lModelsDataDataDict) )
    nMolsNaUsers, lModelsDataDataDict = v
    print( "k, id(v)", k, id(v) )
    print( "k, id(nMolsNaUsers)", k, id(nMolsNaUsers) )
    print( "k, id(lModelsDataDataDict)", k, id(lModelsDataDataDict) )
    print( "k, len(nMolsNaUsers), len(lModelsDataDataDict)", k, len(nMolsNaUsers), len(lModelsDataDataDict) )
    plt.clf()
    plt.close()
    for i in range( 1,33 ):
      sumNMols = sum(nMolsNaUsers[i])
      cumProb = 0
      # for u in range(num_users):
      if sumNMols > 0:
        for u in range(num_users):
          # print(i, nMolsNaUsers[i][u]/sumNMols, "nMolsNaUsers[i][u], sumNMols", nMolsNaUsers[i][u], sumNMols, "left=", cumProb, "color=", uColors[u])
          plt.barh(i, nMolsNaUsers[i][u]/sumNMols, left=cumProb, color=uColors[u])
          cumProb += nMolsNaUsers[i][u]/sumNMols
    # plt.show()
    plt.savefig("fedgan5/img/" + k+".png")
  pass

def getEntry_uMol_data_dirs_list(splitText="9355-8744-3874", n_users=3):
  # v = [
  #   'data_smiles/noniid/split-9355-8744-3874/3/0.pkl.dataset', 
  #   'data_smiles/noniid/split-9355-8744-3874/3/1.pkl.dataset', 
  #   'data_smiles/noniid/split-9355-8744-3874/3/2.pkl.dataset']
  v = []
  noniidFolderName = 'data_smiles/noniid/'
  dir_list = os.listdir( noniidFolderName )
  for fn in dir_list:
    if splitText in fn:
      print( "if splitText in fn:", fn )
      splitFolderName = os.path.join( noniidFolderName, fn )
      print( "splitFolderName:", splitFolderName )
      for i in range(n_users):
        udatasetfnPrefix = os.path.join( splitFolderName, str(n_users) )
        udatasetfn = os.path.join( udatasetfnPrefix, str(i) + '.pkl.dataset' )
        print( udatasetfn, udatasetfnPrefix )
        v.append( udatasetfn )
  # v = []
  return v

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--alphaKey', type=str, default='5')
  parser.add_argument('--splitText', type=str, default='9355-8744-3874')
  argsAlpha = parser.parse_args()
  print( "argsAlpha.alphaKey", argsAlpha.alphaKey )
  alphaKey = argsAlpha.alphaKey
  splitText = argsAlpha.splitText
  alpha = float( alphaKey )
  print( "sys.argv (before reset)", sys.argv )
  # sys.argv = []
  sys.argv = sys.argv[:1]

  args = getArgs()
  uMol_data_dirs_list = {}
  uMol_data_dirs_list["iid"] = genDatasetSplits(args.num_users)
  v = getEntry_uMol_data_dirs_list(splitText=splitText)
  if len(v) >0:
    uMol_data_dirs_list["nonIid-0.5"] = v
  else:
    print( "len(v)<=0", "getEntry_uMol_data_dirs_list", splitText )
    sys.exit()
  labelsEtcDict = {}
  naMolsDict = {}
  for datakey,uMol_data_dirs in uMol_data_dirs_list.items():
    labels, mols, smiles, atomicNumRepresentative, nMolsNaUsers, lModelsDataDataDict = check_iid_uMol_data_dirs(args, uMol_data_dirs)
    labelsEtcDict[ datakey ] = [labels, mols, smiles, atomicNumRepresentative]
    naMolsDict[ datakey ] = [nMolsNaUsers, lModelsDataDataDict]
    print( "len( labels )", datakey, len( labels ) )
  labels, mols, smiles, atomicNumRepresentative = labelsEtcDict[ "iid" ]
  # nMolsNaUsers, lModelsDataDataDict = naMolsDict[ "iid" ]
  # alpha = 5
  # alphaKey = str(alpha)
  non_iid_clients = create_clients_dirichlet(labels, mols, smiles, alpha=alpha)
  naMolsDict[ "nonIid-" + alphaKey ] = getNaStatisitcs(non_iid_clients, atomicNumRepresentative, mols)
  nMolsNaUsers, lModelsDataDataDict = naMolsDict[ "iid" ]
  adjustIidNaStatisitcs(nMolsNaUsers, lModelsDataDataDict, mols, atomicNumRepresentative, args)
  getProbDistrNaCategories(naMolsDict)
  now = datetime.datetime.now() ; formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")
  getStackedSubplots(naMolsDict, formatted_date)
  getStackedSubplotsExcl78(naMolsDict, formatted_date)
  getStackedSubplots2xCol(naMolsDict, formatted_date)

  # now = datetime.datetime.now() ; formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")
  non_iid_clients_dict = SqliteDict( "non_iid_clients_dict_by_alpha_datetime.sqlite" )
  # alphaKey = str(alpha)
  print( "alphaKey", alphaKey )
  non_iid_clients_dict_alpha = {}
  if alphaKey in non_iid_clients_dict:
    non_iid_clients_dict_alpha = non_iid_clients_dict[alphaKey]
  non_iid_clients_dict_alpha[ formatted_date ] = non_iid_clients
  non_iid_clients_dict[alphaKey] = non_iid_clients_dict_alpha
  print( list( non_iid_clients_dict[alphaKey].keys() ) )
  print( list( naMolsDict.keys() ) )
  non_iid_clients_dict.commit()
  # pathList = genDatasetSplitsNoniid(non_iid_clients, atomicNumRepresentative, mols)
