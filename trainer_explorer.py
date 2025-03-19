import pickle
import numpy as np
from rdkit import Chem
from datetime import datetime
import time
from molecular_dataset import MolecularDataset
# from molecular_dataset import genDatasetSplits
from  data_smiles.drugbank import genPkl

# from Dataloader import get_loader
# from Dataloader import Molecular
from torch.utils import data
import argparse
from trainer_class_explorer import Trainer
import matplotlib.pyplot as plt
import copy
import pprint
import utils

import os
htmlStyle="""
<style>._txt_smlw {width:40px;float:left; margin:1px; padding:0px;border:solid 1px black;overflow:hidden;}</style>
<style>._txt_200 {width:200px;float:left; margin:1px; padding:0px;border:solid 1px black;overflow:hidden;}</style>
<style>._txt_400orig {color:#ff3399;font-family: "", Arial Black;height:300px;width:400px;float:left; margin:1px; padding:0px;border:solid 1px black;overflow:hidden;}</style>
"""

import difflib
import pprint
import datetime

import io
from os import listdir
import torch

import molecular_dataset_test
from splitter import getAtomicNumRepresentative

# class Trainer(object):
#   def __init__(self, mol_data_dir):
#    self.data = MolecularDataset()
#    self.data.load(mol_data_dir)
#    pass

class Molecular(data.Dataset):
  """Dataset class for the Molecular dataset"""

  def __init__(self, data_dir):
      self.data = MolecularDataset()
      self.data.load(data_dir)

  def __getitem__(self, index):
      """Return one molecule and its corresponding attribute label"""

      return index, self.data.data[index], self.data.smiles[index],\
             self.data.data_S[index], self.data.data_A[index],\
             self.data.data_X[index], self.data.data_D[index],\
             self.data.data_F[index], self.data.data_Le[index],\
             self.data.data_Lv[index]

  def __len__(self):
      """Return the number of molecules"""
      return len(self.data.data)

def str2bool(v):
  return v.lower() in ('true')

def argsToCmdline(args):
  output = io.StringIO()
  for arg in vars(args):
    print(' {} {}'.format(arg, getattr(args, arg) or ''), file=output)
  result = output.getvalue()
  output.close()
  return result

def getArgs():
  parser = argparse.ArgumentParser()

  # Model configuration.
  parser.add_argument('--z_dim', type=int, default=16, help='dimension of domain labels')
  parser.add_argument('--g_conv_dim', default=[64, 128], help='number of conv filters in the first layer of G')
  parser.add_argument('--d_conv_dim', type=int, default=[[32, 64], 32, [64, 1]], help='number of conv filters in the first layer of D') #[128, 64], 128, [128, 64]
  parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
  parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
  parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
  parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
  parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
  parser.add_argument('--post_method', type=str, default='softmax', choices=['softmax', 'soft_gumbel', 'hard_gumbel'])

  # Training configuration.
  parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size') #16
  parser.add_argument('--num_iters_local', type=int, default=1000, help='number of total iterations for training D') #200000
  parser.add_argument('--num_iters_decay', type=int, default=10, help='number of iterations for decaying lr') #100000
  parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
  parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
  parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
  parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
  parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
  parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
  parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
  parser.add_argument('--epochs_global', type=int, default=50, help="number of rounds of training")
  parser.add_argument('--num_users', type=int, default=3, help="number of users: K")
  parser.add_argument('--frac', type=float, default=1, help='the fraction of clients: C')

  # Test configuration.
  parser.add_argument('--test_iters', type=int, default=1000, help='test model from this step') #200000

  # Miscellaneous.
  parser.add_argument('--num_workers', type=int, default=1)
  parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
  parser.add_argument('--use_tensorboard', type=str2bool, default=False)
  parser.add_argument('--data_iid', type=int, default=1, help='Default set to IID. Set to 0 for non-IID.')
  # parser.add_argument('--data_noniid', type=int, default=0, help='whether to use unequal data splits for non-i.i.d setting (use 0 for equal splits)')

  # Directories.
  parser.add_argument('--mol_data_dir', type=str, default='data_smiles/qm8.dataset')
  parser.add_argument('--log_dir', type=str, default='fedgan5/logs')
  parser.add_argument('--model_save_dir', type=str, default='fedgan5/models')
  parser.add_argument('--sample_dir', type=str, default='fedgan5/samples')
  parser.add_argument('--result_dir', type=str, default='fedgan5/results')

  # Step size.
  parser.add_argument('--log_step', type=int, default=10) #10
  parser.add_argument('--sample_step', type=int, default=1000)  #1000
  parser.add_argument('--model_save_step', type=int, default=1000) #10000
  parser.add_argument('--lr_update_step', type=int, default=1000)  #1000

  # 
  parser.add_argument('--gtl_arr_fp', type=str, default="fedgan5/Gen-loss-FedAvg.txt")
  parser.add_argument('--dtl_arr_fp', type=str, default="fedgan5/Dis-loss-FedAvg.txt")
  parser.add_argument('--man_resume_filepath', type=str, default=None) # man: manual
  parser.add_argument('--cmd', type=str, default="plot")

  args = parser.parse_args()
  return args

def data_iid(dataset, num_users): # from: graphganfeddrugbank\molecularGAN\GraphGANFed\molecular_dataset.py
  num_items = int(len(dataset)/num_users)
  dict_users, all_idxs = {}, [i for i in range(len(dataset))]
  print( "len(dataset)", len(dataset) )
  # print( "len( str(dataset.data) )", len( str(dataset.data) ) )
  print( "len( str(dataset.dataset.data) )", len( str(dataset.dataset.data) ) )
  print( "str(dataset.dataset.data)", str(dataset.dataset.data) )
  print( "len(dataset.dataset.data)", len(dataset.dataset.data) )
  print( "len(dataset.dataset.data.data)", len(dataset.dataset.data.data) )
  print( "type(dataset)", type(dataset) )

  for i in range(num_users):
    dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
    all_idxs = list(set(all_idxs) - dict_users[i])
    print( "len( all_idxs )", len( all_idxs ) )
  return dict_users

def get_loader(args): # from graphganfeddrugbank\molecularGAN\GraphGANFed\Dataloader.py
  num_workers = 1
  dataset = Molecular(args.mol_data_dir)

  train_loader = data.DataLoader(dataset=dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=num_workers)
  test_loader = data.DataLoader(dataset=dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=num_workers)

  user_groups = data_iid(train_loader, args.num_users)
  return train_loader, test_loader, user_groups

def plotdgloss(args
    ): # gtl_arr_fp = g_train_loss_array file path
  g_train_loss_array = np.loadtxt(args.gtl_arr_fp)
  g_train_loss = g_train_loss_array.tolist()
  d_train_loss_array = np.loadtxt(args.dtl_arr_fp)
  d_train_loss = d_train_loss_array.tolist()

  plt.plot(g_train_loss, label="Generator")
  plt.plot(d_train_loss, label="Discriminator")
  plt.legend()
  nr = len(d_train_loss) # nr: number of global rounds
  plt.xticks(np.arange(0, nr, 20) )
  plt.xlabel("Global rounds (" + str( nr ) + ')')
  plt.ylabel("Loss")
  plt.savefig("Gen-Dis-Loss-for-FedAvg.png")

def manualResume(args, g_global_model, d_global_model):
  rDict = {} # result dict
  if args.man_resume_filepath:
    for f in listdir(args.man_resume_filepath):
      fp = os.path.join(args.man_resume_filepath, f)
      if 'G.ckpt' in f:
        rDict[fp] = g_global_model.load_state_dict(torch.load(fp, map_location=lambda storage, loc: storage))
      if 'D.ckpt' in f:
        rDict[fp] = d_global_model.load_state_dict(torch.load(fp, map_location=lambda storage, loc: storage))
  return rDict

def create_path_if_not_exists(path):
  """Creates a directory if it does not exist."""
  if not os.path.exists(path):
    os.makedirs(path)

def genDatasetSplits(num_users, dataRoorFolder = 'data_smiles/'):
  pathList = []
  # follow: https://lobogit.unm.edu/tallpik3/graphganfeddrugbank/-/commit/07add03dabe34c21f0ddebedf146a41020942102
  splitDSFolder = os.path.join( dataRoorFolder, 'split')
  splitDSFolder = os.path.join( splitDSFolder, str(num_users))
  create_path_if_not_exists( splitDSFolder )

  for u in range(num_users):
    uDSPath = os.path.join( splitDSFolder, str(u) + '.pkl.dataset' )
    if os.path.exists(uDSPath):
      pathList.append(uDSPath)
  if len(pathList) == num_users: return pathList
  fieldsList, fieldsDict, fieldsNameDict, filenameMDsDict, filenameMDsObjDict = molecular_dataset_test.getDicts()
  smilesList = []
  mol32 = None
  filenames = ['data-smiles/qm8_smiles.pkl','data_smiles/drugbank/diabetes-drugbank.pkl']
  atomicNumRepresentative = getAtomicNumRepresentative(filenames)
  for filename in filenames:
    mdata = filenameMDsObjDict[filename]
    for mol in mdata.data:
      if mol32 is None:
        if mol.GetNumAtoms() == 32:
          mol32 = Chem.CanonSmiles( Chem.MolToSmiles(mol) )
      if mol.GetNumAtoms() < 33:
        smilesList.append( Chem.CanonSmiles( Chem.MolToSmiles(mol) ) )

  uSmilesLists = []
  for u in range(num_users):
    uSmilesLists.append([])
  addedN = 0
  uCurr = 0
  while len(smilesList) > 0:
    toAdd = smilesList.pop()
    uCurr = addedN % num_users
    uSmilesLists[uCurr].append(toAdd)
    addedN += 1

  pathList = []
  for u in range(num_users):
    uDSFilename_pkl = os.path.join( splitDSFolder, str(u) + '.pkl' )
    for k,v in atomicNumRepresentative.items():
      uSmilesLists[u].append(v)
    uSmilesLists[u].append(mol32)
    with open(uDSFilename_pkl, 'wb') as f:
      pickle.dump( np.array(uSmilesLists[u]) , f)
    data = MolecularDataset()
    data.generate( uDSFilename_pkl, validation=0.1, test=0.1 ) # data_smiles\\esol_smiles.pkl
    data.save( uDSFilename_pkl + '.dataset' )
    pathList.append(uDSFilename_pkl + '.dataset')
  return pathList

def modetrain(args):
  epochs_global = 3 # https://github.com/danielmanu93/GraphGANFed/blob/85d245d6468272604bc3c985511c7d8d7f42e09c/main.py#L141
  frac = 1 # https://github.com/danielmanu93/GraphGANFed/blob/85d245d6468272604bc3c985511c7d8d7f42e09c/main.py#L143
  num_users = 3 # https://github.com/danielmanu93/GraphGANFed/blob/85d245d6468272604bc3c985511c7d8d7f42e09c/main.py#L142C28-L142C37
  mol_data_dir = 'data_smiles/qm8-diabetes-drugbank.pkl.dataset' # parser.add_argument('--mol_data_dir', type=str, default='C:\\Users\\DANIEL\\Desktop\\fedgan\\data_smiles\\esol.dataset') ; https://github.com/danielmanu93/GraphGANFed/blob/85d245d6468272604bc3c985511c7d8d7f42e09c/main.py#L156C5-L156C125

  # args = getArgs()
  args.mol_data_dir = mol_data_dir

  progressDict = {}
  progressDict['args'] = argsToCmdline(args)
  now = datetime.datetime.now() ; formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")
  progressDictFN = 'fedgan5/logs/progressDict.'+formatted_date+'.txt'
  glossnpsavetxtFN = 'fedgan5/logs/Gen-loss-FedAvg.'+formatted_date+'.txt'
  dlossnpsavetxtFN = 'fedgan5/logs/Dis-loss-FedAvg.'+formatted_date+'.txt'
  trainer = Trainer(args, data=None, idxs=None)
  g_global_model, d_global_model = trainer.build_model()
  # global model weights
  g_global_weights = g_global_model.state_dict()
  d_global_weights = d_global_model.state_dict()
  g_train_loss, d_train_loss = [], []
  g_last_local_loss, d_last_local_loss = [], []

  uMol_data_dirs = genDatasetSplits(num_users)

  progressDict['uMol_data_dirs'] = uMol_data_dirs
  progressDict['manualResume'] = manualResume(args, g_global_model, d_global_model)
  with open(progressDictFN,'w') as data:
    data.write(pprint.pformat(progressDict, sort_dicts=False))

  for ep in range(epochs_global):
    g_local_weights, g_local_losses, d_local_weights, d_local_losses = [], [], [], []
    progressDict[ep] = {}

    m = max( int(frac * num_users), 1)
    idxs_users = np.random.choice( range(num_users), m, replace=False)
    idxs_usersDict = {}
    for idx in idxs_users:
      progressDictLocal = {}
      idxs_usersDict[idx] = {}
      local_model = Trainer(args=args, data=None, idxs=None)
      local_model.data.load(uMol_data_dirs[idx])
      idxs_usersDict[idx]["local_model.data.train_idx"] = (ep, idx, local_model.data.train_idx)
      idxs_usersDict[idx]["type(local_model.data.train_idx"] = str( type(local_model.data.train_idx) )
      idxs_usersDict[idx]["len(local_model.data.train_idx"] = len(local_model.data.train_idx)

      local_model.num_iters_local = 10
      g_weights, d_weights, g_loss, d_loss = local_model.tnr(
        modeld=copy.deepcopy(d_global_model),
        modelg=copy.deepcopy(g_global_model),
        global_round=ep, progressDictLocal=progressDictLocal)
      idxs_usersDict[idx]['g_loss'] = g_loss
      idxs_usersDict[idx]['type(g_loss)'] = type(g_loss)
      idxs_usersDict[idx]['type(g_loss[0])'] = type(g_loss[0])
      progressDict[ep][idx] = progressDictLocal

      idxs_usersDict[idx]['(ep, idx)'] = (ep, idx)
      g_local_weights.append(copy.deepcopy(g_weights)) ; g_local_losses.append( g_loss[0] ) ; idxs_usersDict[idx]['g_local_losses'] = copy.deepcopy(g_local_losses)
      d_local_weights.append(copy.deepcopy(d_weights)) ; d_local_losses.append( d_loss[0] )
      g_last_local_loss.append(g_local_losses[-1]) ; d_last_local_loss.append(d_local_losses[-1])

    progressDict[ep]['g_local_losses'] = g_local_losses
    progressDict[ep]['d_local_losses'] = d_local_losses
    progressDict[ep]['idxs_usersDict'] = idxs_usersDict
    progressDict[ep]['len(g_last_local_loss)'] = len(g_last_local_loss)
    # g_local_losses = np.array(g_last_local_loss).ravel()
    progressDict[ep]['len(g_local_losses)'] = len(g_local_losses)
    # d_local_losses = np.array(d_last_local_loss).ravel()
    g_global_weights = utils.average_weights(g_local_weights)
    d_global_weights = utils.average_weights(d_local_weights)
    g_global_model.load_state_dict(g_global_weights)
    d_global_model.load_state_dict(d_global_weights)
    g_loss_avg = sum(g_local_losses) / len(g_local_losses)
    d_loss_avg = sum(d_local_losses) / len(d_local_losses)
    g_train_loss.append(g_loss_avg)
    d_train_loss.append(d_loss_avg)
    g_train_loss_array = np.array(g_train_loss)
    d_train_loss_array = np.array(d_train_loss)
    np.savetxt(glossnpsavetxtFN, g_train_loss_array)
    np.savetxt(dlossnpsavetxtFN, d_train_loss_array)

    with open(progressDictFN,'w') as data:
      data.write(pprint.pformat(progressDict, sort_dicts=False))
  pass

if __name__ == '__main__':
  args = getArgs()
  if "plot" == args.cmd : plotdgloss(args)
  if "train" == args.cmd : modetrain(args)

