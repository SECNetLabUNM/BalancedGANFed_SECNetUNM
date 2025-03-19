import ast
import git
import os
from git import Repo
import inspect
import argparse
import json
import sys

from trainer_test import getArgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import plotUtils
nlinestyles = plotUtils.nlinestyles
linestyle_tuple = plotUtils.linestyle_tuple
nmarkers = plotUtils.nmarkers
markers = plotUtils.markers
import statistics
import numpy as np
import pprint
import datetime

def load_args(filename):
  with open(filename, 'r') as file:
    return json.load(file)

epochs_global, args, num_users, num_iters_local = 0,None,0,0
def getPD(fn = 'fedgan5/logs/progressDict.2024-11-18_22-11-50.txt'):
  global epochs_global, args, num_users, num_iters_local, filename, log_step, n_critic
  cwd = os.getcwd()
  repo = Repo(cwd, search_parent_directories=True)
  tree = repo.head.commit.tree
  print( "(cwd, repo.head)", (cwd, repo.head) )
  print( "type(cwd)", type(cwd) )
  print( "type(repo.head)", type(repo.head) )
  # print( "vars(repo.head.commit)", vars(repo.head.commit) )
  # print( "inspect.getmembers(repo.head.commit)", inspect.getmembers(repo.head.commit) )
  # print( "len() ; inspect.getmembers(repo.head.commit)", len( inspect.getmembers(repo.head.commit) ) )
  # print( "type() ; inspect.getmembers(repo.head.commit)", type( inspect.getmembers(repo.head.commit) ) )
  # for m in inspect.getmembers(repo.head.commit):
  #   print( "m, type(m)", m, type(m) )
  print( "repo.head.commit.hexsha", repo.head.commit.hexsha )
  print( "type: repo.head.commit.hexsha", type( repo.head.commit.hexsha ) )

  filename = fn
  with open(filename) as f: # 3/uav_iab/-/blob/main/3dPlot.py
    data = f.read()
    pD = ast.literal_eval(data) # pD: progressDict
    print( "pD.keys()", pD.keys() )
    # print( "type ; pD.keys()", type( pD.keys() ) )
    # print( "type ; pD['epochs_global']", type( pD['epochs_global'] ) )
    # print( "type ; pD[0]", type( pD[0] ) )
    # print( "pD[0]", pD[0].keys() )
    # print( "pD[0][0]", pD[0][0].keys() )
    # print( "pD[0][0][0]", pD[0][0][0].keys() )
    # print( "pD[0][0][0]['loss']", pD[0][0][0]['loss'] )
    # print( "type ; pD[0][0][0]['loss']", type( pD[0][0][0]['loss'] ) )

    epochs_global = pD['epochs_global']
    argsJsonFN = pD['argsJsonFN']
    restored_args = load_args(argsJsonFN)
    parser = argparse.ArgumentParser()
    args = parser.parse_args([], namespace=argparse.Namespace(**restored_args))
    num_users = args.num_users
    num_iters_local = args.num_iters_local
    log_step = args.log_step
    n_critic = args.n_critic
    print( "type(args)", type(args) )
    print( "args.num_iters_local", args.num_iters_local )
    print( "type ; args.num_iters_local", type(args.num_iters_local) )
    # for arg in vars(args):
    #   print( "", arg, type(arg) )
    print( "",  )
    return pD

def parseFloat(txt, ep, u, it, itK, xYDict, x):
  try:
    y = float(txt)
    pass
  except:
    evalFailCases.append( (ep, u, it, txt ) )
    pass
  else:
    xYDict['x'].append(x)
    xYDict['y'].append(y)

# itemKeys = ['d_valid_loss', 'g_valid_loss', 'D/loss_real', 'D/loss_fake', 'D/loss_gp', 'G/loss_fake', 'G/loss_value', 'QED score','logP score', 'diversity score','similarity_scores','valid score','unique score','novel score']
itemKeys = ['D/loss_real', 'D/loss_fake', 'D/loss_gp', 'G/loss_fake', 'G/loss_value', 'QED score','logP score', 'diversity score','similarity_scores','valid score','unique score','novel score']
vlossKeys = ['d_valid_loss', 'g_valid_loss']
trlossKeys = ['d_batch_loss', 'g_batch_loss'] # tr: training loss
eplossKeys = ['ep_d_valid_loss', 'ep_g_valid_loss',
  'ep_d_batch_loss', 'ep_g_batch_loss'] # batch means "training" batch, ep: epoch loss
ep2localLosskeys = {
  'ep_d_valid_loss':'d_valid_loss',
  'ep_g_valid_loss':'g_valid_loss',
  'ep_d_batch_loss':'d_batch_loss',
  'ep_g_batch_loss':'g_batch_loss',
  }
# vlossKeys = ['d_valid_loss', 'g_valid_loss']
def getXYDict(pD):
  global evalFailCases
  xYDict = {}
  # for itK in vlossKeys:
  #   xYDict[itK] = { 'x':[], 'y':[] }
  for itK in itemKeys + vlossKeys + trlossKeys + eplossKeys:
    xYDict[itK] = {}
    for u in range(num_users):
      xYDict[itK][u] = { 'x':[], 'y':[] }
  uXSteps = {}
  vLIdxSteps = {} # valida. loss index
  tLIdxSteps = {} # traini. loss index
  for u in range(num_users):
    uXSteps[u] = 0
    vLIdxSteps[u] = 0
    tLIdxSteps[u] = 0
  evalFailCases = []
  for ep in range(epochs_global):
    for u in range(num_users):
      # for itK in vlossKeys:
      vLIdxSteps[u] = 0
      tLIdxSteps[u] = 0
      for it in range(num_iters_local):
        # ast.literal_eval(pD[ep][u][it]['loss-items']
        if (it+1) % n_critic == 0:
          for itK in trlossKeys:
            txt = pD[ep][u][itK][tLIdxSteps[u]]
            parseFloat(txt, ep, u, it, itK, xYDict[itK][u], uXSteps[u])
          tLIdxSteps[u] += 1
        if (it+1) % log_step == 0:
          for itK in vlossKeys:
            # print( "vLIdxSteps[u], ep, u, it, itK, log_step", (vLIdxSteps[u], ep, u, it, itK, log_step) )
            txt = pD[ep][u][itK][vLIdxSteps[u]]
            parseFloat(txt, ep, u, it, itK, xYDict[itK][u], uXSteps[u])
          vLIdxSteps[u] += 1
        for itK in itemKeys:
          if itK in pD[ep][u][it]['loss-items']:
            txt = pD[ep][u][it]['loss-items'][itK]
            parseFloat(txt, ep, u, it, itK, xYDict[itK][u], uXSteps[u])
        uXSteps[u] += 1
    # g_epoch_loss.append(sum(g_batch_loss[-2:])/len(g_batch_loss[-2:]))
    for u in range(num_users):
      for itK in eplossKeys:
        x = uXSteps[u] -1 # -1 because uXSteps[u] is already incr-ed
        xYDict[itK][u]['x'].append( x )
        lk = ep2localLosskeys[itK]
        y = statistics.mean( xYDict[lk][u]['y'][-2:] )
        # print( '\titK, ep, u, x, y, -2:,len(y):', itK, ep, u, x, y, [ ffmt2(fln) for fln in xYDict[lk][u]['y'][-2:] ], len(xYDict[lk][u]['y']) ) # fln = floating number
        xYDict[itK][u]['y'].append( y )

  print( "len(evalFailCases)", len(evalFailCases) )
  print( "uXSteps", uXSteps )
  return xYDict

def ffmt2(f_num): # float formatting .2 ; rsrc_management/iab/stackelberg.py
  return "{:6.2f}".format(f_num)

def plotShowItemKeysGroups(xYDict, itemKeysGroups = [ ['D/loss_real', 'D/loss_fake', 'D/loss_gp', 'G/loss_fake', 'G/loss_value'] + vlossKeys, ['similarity_scores','valid score','unique score','novel score'] ] ):
  # itemKeysGroups = [ ['D/loss_real', 'D/loss_fake', 'D/loss_gp', 'G/loss_fake', 'G/loss_value'] + vlossKeys, ['similarity_scores','valid score','unique score','novel score'] ]
  for itemKeys in itemKeysGroups:
    colors=plotUtils.getGradColors('m', len(itemKeys) * num_users )
    colorStep = 0
    for itK in itemKeys:
      for u in range(num_users):
        plt.plot(xYDict[itK][u]['x'], xYDict[itK][u]['y'], label=itK+str(u), marker=markers[colorStep%nmarkers], linestyle=linestyle_tuple[colorStep%nlinestyles][1] )
        colorStep += 1
    plt.legend()
    plt.show()
    plt.clf()

def plotSaveItemKeysGroupsAvgUsers(xYDict, itemKeysGroups ):
  # itemKeysGroups = [ ['D/loss_real', 'D/loss_fake', 'D/loss_gp', 'G/loss_fake', 'G/loss_value'] + vlossKeys, ['similarity_scores','valid score','unique score','novel score'] ]
  resultImgsPaths = []
  for kv in itemKeysGroups:
    grName, itemKeys = kv
    colors=plotUtils.getGradColors('m', len(itemKeys) )
    colorStep = 0
    for itK in itemKeys:
      uValsArrs = []
      for u in range(num_users):
        uValsArrs.append( np.array( xYDict[itK][u]['y'] ) )
      mean_array = np.mean( uValsArrs, axis=0 )
      plt.plot(xYDict[itK][u]['x'], mean_array, label=itK, marker=markers[colorStep%nmarkers], linestyle=linestyle_tuple[colorStep%nlinestyles][1] )
      colorStep += 1
    plt.legend()
    # plt.show()
    oName = os.path.basename(filename)
    imgsPath = "fedgan5/img/" +oName +'-' +grName + ".png"
    plt.savefig( imgsPath ) # 3/uav_iab/-/blob/main/plotPdf.py
    resultImgsPaths.append( imgsPath )
    plt.clf()
  return resultImgsPaths

def plotSaveItemKeysGroupsUsers(xYDict):
  itemKeysGroups = { "loss":['D/loss_real', 'D/loss_fake', 'D/loss_gp', 'G/loss_fake', 'G/loss_value'] + vlossKeys, 
    "train-validation-compare": vlossKeys + trlossKeys,
    "score":['similarity_scores','valid score','unique score','novel score'],
    }
  oName = os.path.basename(filename)
  resultImgsPaths = []
  for grName,itemKeys in itemKeysGroups.items():
    for u in range(num_users):
      colors=plotUtils.getGradColors('m', len(itemKeys) )
      colorStep = 0
      #for u in range(num_users):
      for itK in itemKeys:
        # , color = colors[colorStep]
        plt.plot(xYDict[itK][u]['x'], xYDict[itK][u]['y'], label=itK+str(u), marker=markers[colorStep%nmarkers], linestyle=linestyle_tuple[colorStep%nlinestyles][1] )
        colorStep += 1
      plt.title(grName)
      plt.legend()
      imgsPath = "fedgan5/img/" +oName +'-' +grName +"-u" + str(u)+ ".png"
      # plt.savefig( imgsPath+".pdf", format='pdf', bbox_inches='tight', dpi=720 ) # 3/uav_iab/-/blob/main/plotPdf.py
      plt.savefig( imgsPath ) # 3/uav_iab/-/blob/main/plotPdf.py
      resultImgsPaths.append( imgsPath )
      plt.clf()
  return resultImgsPaths

def plotSaveItemKeysGroupsUsersItemKeys(grName,itemKeys,xYDict):
  oName = os.path.basename(filename)
  resultImgsPaths = []
  for u in range(num_users):
    colors=plotUtils.getGradColors('m', len(itemKeys) )
    colorStep = 0
    #for u in range(num_users):
    for itK in itemKeys:
      # color = colors[colorStep]
      plt.plot(xYDict[itK][u]['x'], xYDict[itK][u]['y'], label=itK+str(u), marker=markers[colorStep%nmarkers], linestyle=linestyle_tuple[colorStep%nlinestyles][1] )
      colorStep += 1
    plt.legend()
    imgsPath = "fedgan5/img/" +oName +'-' +grName +"-u" + str(u)+ ".png"
    plt.savefig( imgsPath+".pdf", format='pdf', bbox_inches='tight', dpi=720 )
    resultImgsPaths.append( imgsPath )
    plt.clf()
  return resultImgsPaths

def pairwiseMergeEpLosses(xYDictMerged, xYDict):
  for u in range(num_users):
    for itK in eplossKeys:
      offset = xYDictMerged[itK][u]['x'][-1]
      for x,y in zip(xYDict[itK][u]['x'], xYDict[itK][u]['y']):
        xYDictMerged[itK][u]['x'].append( x + offset)
        xYDictMerged[itK][u]['y'].append( y)

import copy
def mergeEpLosses(logsDict, logs):
  _,xYDict = logsDict[logs[0]]
  xYDictMerged = copy.deepcopy( xYDict )
  _,xYDict = logsDict[logs[1]]
  pairwiseMergeEpLosses(xYDictMerged, xYDict)
  for log in logs[2:]:
    _,xYDict = logsDict[log]
    pairwiseMergeEpLosses(xYDictMerged, xYDict)
  return xYDictMerged

now = datetime.datetime.now() ; formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--logs', nargs='+', type=str)
  args = parser.parse_args()
  logs = args.logs
  logsDict = {}
  for log in logs:
    pD = getPD(fn = log)
    xYDict = getXYDict(pD)
    # plotShowItemKeysGroups(xYDict, itemKeysGroups = [ trlossKeys + vlossKeys ])
    rPaths = plotSaveItemKeysGroupsAvgUsers(xYDict, itemKeysGroups = [ ("eplossKeys",eplossKeys) ] )
    rPaths = plotSaveItemKeysGroupsUsers(xYDict)
    rPaths = plotSaveItemKeysGroupsUsersItemKeys(grName='d-vs-v-output', itemKeys=['G/loss_fake', 'G/loss_value'],xYDict=xYDict)
    rPaths = plotSaveItemKeysGroupsUsersItemKeys(grName='loss-fake-d-vs-g',itemKeys=['D/loss_fake', 'G/loss_fake'],xYDict=xYDict)
    print( "len(pD[0][0]['d_valid_loss'])", len(pD[0][0]['d_valid_loss']) )
    logsDict[log] = (pD, xYDict)
  print( "logs", logs )
  print( "len(logsDict)", len(logsDict) )
  xYDictMerged = mergeEpLosses(logsDict, logs)
  filename = 'fedgan5/logs/mergeEpLosses.txt'
  rPaths = plotSaveItemKeysGroupsAvgUsers(xYDictMerged, itemKeysGroups = [ ("eplossKeys",eplossKeys) ] )
  xYDictMerged["args.logs"] = logs
  xYDictMergedFN = 'fedgan5/logs/xYDictMerged.' + formatted_date + '.txt'
  with open(xYDictMergedFN,'w') as data: data.write(pprint.pformat(xYDictMerged, sort_dicts=False))

