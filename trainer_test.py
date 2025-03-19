import pickle
import numpy as np
from rdkit import Chem
from datetime import datetime
import time
from molecular_dataset import MolecularDataset
from  data_smiles.drugbank import genPkl

# from Dataloader import get_loader
# from Dataloader import Molecular
from torch.utils import data
import argparse
from trainer_debug import Trainer
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
from trainer_explorer import genDatasetSplits
from trainer_explorer import create_path_if_not_exists
import traceback
import sys
import json

import git
from git import Repo
import statistics
import math
from collections import deque

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
  parser.add_argument('--isFL', type=str2bool, default=False)
  parser.add_argument('--isWAvg', type=str2bool, default=False)
  parser.add_argument('--isNonIid', type=str2bool, default=False)
  parser.add_argument('--isFixedRatio', nargs='+', type=int, default=[0, 5, 1]) # 0/1, dsteps, gsteps
  parser.add_argument('--nonIidDatasets', nargs='+', type=str, default=[
    'data_smiles/noniid/split-9355-8744-3874/3/0.pkl.dataset', 
    'data_smiles/noniid/split-9355-8744-3874/3/1.pkl.dataset', 
    'data_smiles/noniid/split-9355-8744-3874/3/2.pkl.dataset'])

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

import matplotlib
def plotdgloss(args, isReturn=False
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
  if isReturn:
    return g_train_loss, d_train_loss
  else:
    plt.savefig("Gen-Dis-Loss-for-FedAvg.png")

def manualResume(args, g_global_model, d_global_model, v_global_model, progressDict):
  rDict = {} # result dict
  if args.man_resume_filepath:
    for f in listdir(args.man_resume_filepath):
      fp = os.path.join(args.man_resume_filepath, f)
      if 'G.ckpt' in f:
        rDict[fp] = str( g_global_model.load_state_dict(torch.load(fp, map_location=lambda storage, loc: storage)) )
      if 'D.ckpt' in f:
        rDict[fp] = str( d_global_model.load_state_dict(torch.load(fp, map_location=lambda storage, loc: storage)) )
      if 'V.ckpt' in f:
        rDict[fp] = str( v_global_model.load_state_dict(torch.load(fp, map_location=lambda storage, loc: storage)) )
  else:
    progressDict['saveGlobalModels'] = saveGlobalModels(args, 'init', g_global_model, d_global_model, v_global_model)
  return rDict

def dsCompareToHtml(uMol_data_dirs):
  for fp in uMol_data_dirs:
    molecular_dataset_test.filenames.append( fp.replace('.dataset', '') )
  fieldsList, fieldsDict, fieldsNameDict, filenameMDsDict, filenameMDsObjDict = molecular_dataset_test.getDicts()
  molecular_dataset_test.dsCompareToHtml(fieldsList, fieldsDict, fieldsNameDict, filenameMDsDict, filenameMDsObjDict)

def save_args(args, filename):
  with open(filename, 'w') as file:
    json.dump(vars(args), file)  # Save as a dictionary

from sqlitedict import SqliteDict
from sqlitedict import SqliteDict

# ChatGPT said: Store nested structures as values
def sqliteDictToDict( db ):
  # Convert to a standard Python dictionary
  python_dict = dict(db.items())
  return python_dict

def getGlobalValidLoss(G, D, V, local_models, args, global_model):
  vloss = {}
  for u in range(args.num_users):
    vloss[u] = {}
    mols, _, _, a, x, _, _, _, _ = local_models[u].data.next_validation_batch()
    lM = local_models[u]
    z = global_model.sample_z(a.shape[0])
    a = torch.from_numpy(a).to(lM.device).long()
    x = torch.from_numpy(x).to(lM.device).long()
    a_tensor = global_model.label2onehot(a, lM.b_dim)
    x_tensor = global_model.label2onehot(x, lM.m_dim)
    # z = torch.from_numpy(z).to(lM.device).float()
    if lM.device.type == 'cuda':
      z = torch.from_numpy(z).to(lM.device).float()
    else:
      z = torch.from_numpy(z).to(torch.float32).to(lM.device)  # Use float32 for MPS and CPU


    logits_real, features_real = D(a_tensor, None, x_tensor)
    d_loss_real = - torch.mean(logits_real)
    edges_logits, nodes_logits = G(z)
    (edges_hat, nodes_hat) = global_model.postprocess((edges_logits, nodes_logits), global_model.post_method)
    logits_fake, features_fake = D(edges_hat, None, nodes_hat)
    d_loss_fake = torch.mean(logits_fake)
    # Compute loss for gradient penalty.
    eps = torch.rand(logits_real.size(0),1,1,1).to(global_model.device)
    x_int0 = (eps * a_tensor + (1. - eps) * edges_hat).requires_grad_(True)
    x_int1 = (eps.squeeze(-1) * x_tensor + (1. - eps.squeeze(-1)) * nodes_hat).requires_grad_(True)
    grad0, grad1 = D(x_int0, None, x_int1)
    d_loss_gp = global_model.gradient_penalty(grad0, x_int0) + global_model.gradient_penalty(grad1, x_int1)
    d_loss = d_loss_fake + d_loss_real + global_model.lambda_gp * d_loss_gp
    # generator
    edges_logits, nodes_logits = G(z)
    (edges_hat, nodes_hat) = global_model.postprocess((edges_logits, nodes_logits), global_model.post_method)
    logits_fake, features_fake = D(edges_hat, None, nodes_hat)
    g_loss_fake = - torch.mean(logits_fake)
    (edges_hard, nodes_hard) = global_model.postprocess((edges_logits, nodes_logits), 'hard_gumbel')
    edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
    mols = [lM.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True) for e_, n_ in zip(edges_hard, nodes_hard)]
    value_logit_real,_ = V(a_tensor, None, x_tensor, torch.sigmoid)
    value_logit_fake,_ = V(edges_hat, None, nodes_hat, torch.sigmoid)
    g_loss_value = torch.mean((value_logit_real) ** 2 + (value_logit_fake) ** 2)
    g_loss = g_loss_fake + g_loss_value
    vloss[u]['d'] = d_loss.item()
    vloss[u]['g'] = g_loss.item()
  pass
  return vloss

def saveGlobalModels(args, global_round, G, D, V):
  G_path = os.path.join(args.model_save_dir, '{}-G.ckpt'.format(global_round))
  D_path = os.path.join(args.model_save_dir, '{}-D.ckpt'.format(global_round))
  V_path = os.path.join(args.model_save_dir, '{}-V.ckpt'.format(global_round))
  torch.save(G.state_dict(), G_path)
  torch.save(D.state_dict(), D_path)
  torch.save(V.state_dict(), V_path)
  return 'Saved model checkpoints into {}...'.format(args.model_save_dir) + str( (G_path, D_path, V_path, ) )

# molecularGAN\GraphGANFed\plotGap.py
class RatioScheduler(object):
  def __init__(self, num_users):
    self.num_users = num_users
    self.r_dgs = []
    self.window = deque()

  def getDg(self, vloss):
    # vloss[u]['d']
    # vloss[u]['g']
    # molecularGAN\GraphGANFed\balance_strategy.py
    d = statistics.mean( [ vloss[u]['d'] for u in range(self.num_users)] )
    g = statistics.mean( [ vloss[u]['g'] for u in range(self.num_users)] )
    return d,g

  # progressDict[ep]['getGlobalValidLoss']
  # r_dg_just_done : just done mean from the iteration just completed
  def update(self, glValidLosses, r_dg_just_done, ep, n_critic):
    d,g = self.getDg(glValidLosses)
    gap = d-g
    dsteps,gsteps = 1,1
    # log(  1.7737838621158528 *x +  1.757278921402407 )
    # 3/graphganfeddrugbank/-/commit/3cd314d226b39966fb8bec940c882bf3230dcc30 ; curvefit_gap_dgratio.py
    # 3/graphganfeddrugbank/-/commit/152cc9e7aec79c53afc6bc5ec7bb19123d1bf575
    if ep < 4:
      r_dg = n_critic
    else :
      if gap >= 0:
        # dsteps = int( math.log(  1.7737838621158528 *gap +  1.757278921402407 ) )
        dsteps = int( 1.6595567410746717 * math.log( gap +  math.e ) )
      else:
        # gsteps = int( math.log(  1.7737838621158528 *gap +  1.757278921402407 ) )
        gsteps = int( 1.6595567410746717 * math.log( gap*-1 +  math.e ) )
      r_dg = dsteps/gsteps

    # r_dg_int = int(r_dg)
    # if r_dg_int <= 0:
    #   r_dg_int = 1
    return r_dg, (gap, r_dg_just_done),dsteps,gsteps

def getGapsLocal(progressDictLocals, num_users): # coefficients
  gaps = {}
  for u in range(num_users):
    d = progressDictLocals[u]['d_valid_loss'][-1]
    g = progressDictLocals[u]['g_valid_loss'][-1]
    gaps[u] = 1/abs(d-g)
  return gaps

def weighted_average_weights(w, wc): # coefficients
  #  print( "type(w[0])", type(w[0]) )
  w_avg = copy.deepcopy(w[0])
  wc_sum = sum(wc.values())# Total Weight (W_total or W_sum)
  for key in w_avg.keys():
    w_avg[key] *= wc[0]
    for i in range(1, len(w)):
      w_avg[key] += w[i][key]*wc[i]
    # print( "type( w[i][key] )", type( w[i][key] ), key, w[i][key].shape )
    w_avg[key] = torch.div(w_avg[key], wc_sum)
  return w_avg

def modetrain(args):
  epochs_global = args.epochs_global # https://github.com/danielmanu93/GraphGANFed/blob/85d245d6468272604bc3c985511c7d8d7f42e09c/main.py#L141
  frac = 1 # https://github.com/danielmanu93/GraphGANFed/blob/85d245d6468272604bc3c985511c7d8d7f42e09c/main.py#L143
  num_users = args.num_users # https://github.com/danielmanu93/GraphGANFed/blob/85d245d6468272604bc3c985511c7d8d7f42e09c/main.py#L142C28-L142C37
  # mol_data_dir = 'data_smiles/qm8-diabetes-drugbank.pkl.dataset' # parser.add_argument('--mol_data_dir', type=str, default='C:\\Users\\DANIEL\\Desktop\\fedgan\\data_smiles\\esol.dataset') ; https://github.com/danielmanu93/GraphGANFed/blob/85d245d6468272604bc3c985511c7d8d7f42e09c/main.py#L156C5-L156C125

  # args = getArgs()
  # args.mol_data_dir = mol_data_dir

  now = datetime.datetime.now() ; formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")
  dbFN = "progressDict."+formatted_date+".sqlite"
  progressDict = SqliteDict( dbFN )
  # progressDict = {}
  # progressDict['args'] = args
  progressDict['args-argsToCmdline'] = argsToCmdline(args)
  # now = datetime.datetime.now() ; formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")
  progressDictFN = 'fedgan5/logs/progressDict.'+formatted_date+'.txt'
  argsJsonFN = 'fedgan5/logs/args.'+formatted_date+'.json'
  glossnpsavetxtFN = 'fedgan5/logs/Gen-loss-FedAvg.'+formatted_date+'.txt'
  dlossnpsavetxtFN = 'fedgan5/logs/Dis-loss-FedAvg.'+formatted_date+'.txt'
  trainer = Trainer(args, data=None, idxs=None)
  g_global_model, d_global_model = trainer.build_model()
  v_global_model = copy.deepcopy( d_global_model )
  # global model weights
  g_global_weights = g_global_model.state_dict()
  d_global_weights = d_global_model.state_dict()
  v_global_weights = v_global_model.state_dict()
  g_train_loss, d_train_loss = [], []
  g_last_local_loss, d_last_local_loss = [], []

  uMol_data_dirs = [
    "data_smiles/qm8-diabetes-drugbank-lt33.pkl.dataset",
    "data_smiles/qm8-diabetes-drugbank-lt33.pkl.dataset",
    "data_smiles/qm8-diabetes-drugbank-lt33.pkl.dataset",
    ]
  if args.isFL:
    uMol_data_dirs = genDatasetSplits(num_users)
  elif args.isNonIid:
    uMol_data_dirs = args.nonIidDatasets
  # dsCompareToHtml(uMol_data_dirs)

  progressDict['uMol_data_dirs'] = uMol_data_dirs
  # progressDict['manualResume'] = manualResume(args, g_global_model, d_global_model, v_global_model, progressDict)
  progressDict['sys.argv'] = sys.argv
  progressDict['epochs_global'] = epochs_global
  args.model_save_dir = os.path.join( args.model_save_dir, formatted_date )
  create_path_if_not_exists( args.model_save_dir )
  progressDict['manualResume'] = manualResume(args, g_global_model, d_global_model, v_global_model, progressDict)
  progressDict['argsJsonFN'] = argsJsonFN
  save_args(args, argsJsonFN)
  cwd = os.getcwd()
  repo = Repo(cwd, search_parent_directories=True)
  progressDict["repo.head.commit.hexsha"] = repo.head.commit.hexsha
  progressDict["statistics.mean( [1,2,3] )"] = statistics.mean( [1,2,3] )
  # progressDict.commit()
  # pyProgressDict = sqliteDictToDict( progressDict )
  local_models = {}
  local_modelsDict = {}
  ratioScheduler = RatioScheduler(num_users)
  if args.isFixedRatio[0] == 1:
    dsteps = args.isFixedRatio[1]
    gsteps = args.isFixedRatio[2]
    r_dg = dsteps / gsteps
  elif args.isFixedRatio[0] == 0:
    r_dg = args.n_critic
  else:
    print( "error: ", "args.isFixedRatio", args.isFixedRatio )
    exit()
  for u in range(num_users):
    local_models[u] = Trainer(args=args, data=None, idxs=None, mol_data_dir=uMol_data_dirs[u])
    # local_ratioSchedulers[u] = RatioScheduler()
    # r_dgDict[u] = args.n_critic
    local_modelsDict[u] = {}
    local_modelsDict[u][".data.train_idx"] = local_models[u].data.train_idx
    local_modelsDict[u][".data.validation_idx"] = local_models[u].data.validation_idx
    local_modelsDict[u][".data.test_idx"] = local_models[u].data.test_idx
  progressDict["local_models"] = local_modelsDict
  progressDict.commit()
  pyProgressDict = sqliteDictToDict( progressDict )
  with open(progressDictFN,'w') as data:
    data.write(pprint.pformat(pyProgressDict, sort_dicts=False))

  progressDictSql = progressDict
  progressDict = {}
  gl_d_steps,gl_g_steps=5, 1 # gl : global
  # local_models = {}
  # for u in range(num_users):
  #   local_models[u] = Trainer(args=args, data=None, idxs=None, mol_data_dir=uMol_data_dirs[u])
  for ep in range(epochs_global):
    g_local_weights, g_local_losses, d_local_weights, d_local_losses = [], [], [], []
    v_local_weights = []
    progressDict[ep] = {}
    # local_models = {}

    m = max( int(frac * num_users), 1)
    idxs_users = np.random.choice( range(num_users), m, replace=False)
    idxs_usersDict = {}
    for idx in idxs_users:
      progressDictLocal = {}
      idxs_usersDict[idx] = {}
      # local_model = Trainer(args=args, data=None, idxs=None, mol_data_dir=uMol_data_dirs[idx])
      # local_model.data = MolecularDataset()
      # local_model.data.load(uMol_data_dirs[idx])
      # idxs_usersDict[idx]["local_model.data.train_idx"] = str( (ep, idx, local_model.data.train_idx) )
      # idxs_usersDict[idx]["type(local_model.data.train_idx"] = str( type(local_model.data.train_idx) )
      # idxs_usersDict[idx]["len(local_model.data.train_idx"] = len(local_model.data.train_idx)

      # local_model.num_iters_local = 1220
      local_model = local_models[idx]
      try:
        g_weights, d_weights, v_weights, g_loss, d_loss = local_model.tnr_sequence_gan(
          modeld=copy.deepcopy(d_global_model),
          modelg=copy.deepcopy(g_global_model),
          modelv=copy.deepcopy(v_global_model),
          global_round=ep, d_steps=gl_d_steps, g_steps=gl_g_steps, progressDictLocal=progressDictLocal)
      except:
        progressDictLocal["traceback.format_exc()"] = str( traceback.format_exc() )
        progressDict[ep][idx] = progressDictLocal
        progressDictSql[ep] = progressDict[ep]
        progressDictSql.commit()
        pyProgressDict = sqliteDictToDict( progressDictSql )
        with open(progressDictFN,'w') as data: data.write(pprint.pformat(pyProgressDict, sort_dicts=False))
      else:
        # idxs_usersDict[idx]['g_loss'] = g_loss
        # idxs_usersDict[idx]['type(g_loss)'] = str( type(g_loss) )
        # idxs_usersDict[idx]['type(g_loss[0])'] = str( type(g_loss[0]) )
        progressDict[ep][idx] = progressDictLocal

        # idxs_usersDict[idx]['(ep, idx)'] = (ep, idx)
        g_local_weights.append(copy.deepcopy(g_weights)) ; g_local_losses.append( g_loss[0] )# ; idxs_usersDict[idx]['g_local_losses'] = copy.deepcopy(g_local_losses)
        d_local_weights.append(copy.deepcopy(d_weights)) ; d_local_losses.append( d_loss[0] )
        v_local_weights.append(copy.deepcopy(v_weights))
        g_last_local_loss.append(g_local_losses[-1]) ; d_last_local_loss.append(d_local_losses[-1])
        idxs_usersDict[idx]['glast_lr'] = local_model.g_scheduler.get_last_lr()[0]
        idxs_usersDict[idx]['dlast_lr'] = local_model.d_scheduler.get_last_lr()[0]
      finally:
        pass

    progressDict[ep]['g_local_losses'] = g_local_losses
    progressDict[ep]['d_local_losses'] = d_local_losses
    progressDict[ep]['idxs_usersDict'] = idxs_usersDict
    progressDict[ep]['len(g_last_local_loss)'] = len(g_last_local_loss)
    # g_local_losses = np.array(g_last_local_loss).ravel()
    progressDict[ep]['len(g_local_losses)'] = len(g_local_losses)
    # d_local_losses = np.array(d_last_local_loss).ravel()
    wc = getGapsLocal(progressDict[ep], num_users)
    print( "wc", wc )
    progressDict[ep]['wc_getGapsLocal'] = wc
    if args.isWAvg:
      g_global_weights = weighted_average_weights(g_local_weights, wc)
      d_global_weights = weighted_average_weights(d_local_weights, wc)
      v_global_weights = weighted_average_weights(v_local_weights, wc)
    else:
      g_global_weights = utils.average_weights(g_local_weights)
      d_global_weights = utils.average_weights(d_local_weights)
      v_global_weights = utils.average_weights(v_local_weights)
    g_global_model.load_state_dict(g_global_weights)
    d_global_model.load_state_dict(d_global_weights)
    v_global_model.load_state_dict(v_global_weights)
    g_loss_avg = sum(g_local_losses) / len(g_local_losses)
    d_loss_avg = sum(d_local_losses) / len(d_local_losses)
    g_train_loss.append(g_loss_avg)
    d_train_loss.append(d_loss_avg)
    g_train_loss_array = np.array(g_train_loss)
    d_train_loss_array = np.array(d_train_loss)
    np.savetxt(glossnpsavetxtFN, g_train_loss_array)
    np.savetxt(dlossnpsavetxtFN, d_train_loss_array)
    # print( "[idxs_usersDict[u]['glast_lr'] for u in range(num_users)]", [idxs_usersDict[u]['glast_lr'] for u in range(num_users)] )
    # print( "[idxs_usersDict[u]['dlast_lr'] for u in range(num_users)]", [idxs_usersDict[u]['dlast_lr'] for u in range(num_users)] )
    args.g_lr = statistics.mean( [idxs_usersDict[u]['glast_lr'] for u in range(num_users)] )
    args.d_lr = statistics.mean( [idxs_usersDict[u]['dlast_lr'] for u in range(num_users)] )
    progressDict[ep]['args.g_lr'] = args.g_lr
    progressDict[ep]['args.d_lr'] = args.d_lr
    progressDict[ep]['getGlobalValidLoss'] = getGlobalValidLoss(g_global_model, d_global_model, v_global_model, local_models, args, global_model=trainer)
    progressDict[ep]['saveGlobalModels'] = saveGlobalModels(args, ep, g_global_model, d_global_model, v_global_model)
    # (gap, r_dg_just_done)
    if args.isFixedRatio[0] == 1:
      dsteps = args.isFixedRatio[1]
      gsteps = args.isFixedRatio[2]
      d,g = ratioScheduler.getDg( progressDict[ep]['getGlobalValidLoss'] )
      gap = d-g
      # return r_dg, (gap, r_dg_just_done),dsteps,gsteps
      r_dg, gr_tuple, gl_d_steps, gl_g_steps = r_dg, (gap, r_dg), dsteps, gsteps
    elif args.isFixedRatio[0] == 0:
      r_dg, gr_tuple, gl_d_steps, gl_g_steps = ratioScheduler.update(progressDict[ep]['getGlobalValidLoss'],
        r_dg_just_done=r_dg,
        ep=ep,
        n_critic=args.n_critic)
    progressDict[ep]['gapratiotuple'] = gr_tuple # return r_dg_int, (gap, r_dg_just_done)
    # r_dg_int = int(r_dg)
    # if r_dg_int <= 0:
    #   r_dg_int = 1
    for u in range(num_users):
      local_models[u].n_critic = r_dg
      # local_models[u].log_step = r_dg*6

    progressDictSql[ep] = progressDict[ep]
    progressDictSql.commit()
  pass
  pyProgressDict = sqliteDictToDict( progressDictSql )
  with open(progressDictFN,'w') as data:
    data.write(pprint.pformat(pyProgressDict, sort_dicts=False))

if __name__ == '__main__':
  args = getArgs()
  if "plot" == args.cmd : plotdgloss(args)
  if "train" == args.cmd : modetrain(args)

