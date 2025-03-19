
import progress_parser
import peakfinding
import datetime
import argparse
import statistics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os
from sqlitedict import SqliteDict
import os
from progress_parser import load_args
import numpy as np
import plotUtils
linestyle_tuple = plotUtils.linestyle_tuple
from itertools import cycle
linestyleCyle = cycle(linestyle_tuple)
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter

def getArgsSqliteDict(pD):
  global epochs_global, num_users, num_iters_local, log_step, n_critic
  argsJsonFN = pD['argsJsonFN']
  restored_args = load_args(argsJsonFN)
  parser = argparse.ArgumentParser()
  args = parser.parse_args([], namespace=argparse.Namespace(**restored_args))
  epochs_global = args.epochs_global
  num_users = args.num_users
  num_iters_local = args.num_iters_local
  log_step = args.log_step
  n_critic = args.n_critic
  return args

def getFilenames(logdatetime):
  fnDict = {}
  paths = [os.getcwd(), "fedgan5/logs/"]
  for path in paths :
    dir_list = os.listdir(path)
    for fn in dir_list:
      if logdatetime in fn:
        if ".sqlite" in fn:
          fnDict["sqlite"] = os.path.join( path, fn )
        elif "Gen-loss-FedAvg" in fn:
          fnDict["gtl_arr_fp"] = os.path.join( path, fn )
        elif "Dis-loss-FedAvg" in fn:
          fnDict["dtl_arr_fp"] = os.path.join( path, fn )
  return fnDict

def getXYDict(pD):
  xYDict = {}

# def plotdgloss(args, isReturn=False @ 3\Documents\unm2h\git\graphganfeddrugbank\molecularGAN\GraphGANFed\peakfinding.py
def loadTLArrNp( fnDict ): # gtl_arr_fp = g_train_loss_array file path
  g_train_loss_array = np.loadtxt( fnDict["gtl_arr_fp"] )
  g_train_loss = g_train_loss_array.tolist()
  d_train_loss_array = np.loadtxt( fnDict["dtl_arr_fp"] )
  d_train_loss = d_train_loss_array.tolist()
  return g_train_loss, d_train_loss

# 3\Documents\unm2h\git\graphganfeddrugbank\molecularGAN\GraphGANFed\progress_zoom.py
def plotGlobalVLoss(pD, fnDict, restored_args):
  import progress_zoom
  from peakfinding import plotOriginalData
  globalValidLoss = {}
  globalLr = {}
  tToTextLabel = {'d':'Dis Val Loss', 'g':'Gen Val Loss'}
  lrToTextLabel = {'args.d_lr':'Dis LR', 'args.g_lr':'Gen LR'}
  for t in ['d', 'g']:
    globalValidLoss[t] = []
  for t in ['args.d_lr', 'args.g_lr']:
    globalLr[t] = []
  r_dg = []
  gaps = []
  n_epochPerGlRounds = []
  local_dg = {}
  wc = []
  for u in range(num_users):
    local_dg[u] = {}
    for t in ['d_valid_loss', 'g_valid_loss']:
      local_dg[u][t] = []
  for ep in range(epochs_global):
    ep = str(ep)
    epDict = {}
    if ep not in pD:
      break
    else:
      epDict = pD[ep]
    for t in ['d', 'g']:
      globalValidLoss[t].append( statistics.mean( [epDict['getGlobalValidLoss'][u][t] for u in range(num_users)] ) )
    for t in ['d_valid_loss', 'g_valid_loss']:
      for u in range(num_users):
        local_dg[u][t].append( epDict[u][t][-1] )
    for t in ['args.d_lr', 'args.g_lr']:
      globalLr[t].append( epDict[t] )
    r_dg.append( epDict['gapratiotuple'][1] )
    gaps.append( epDict['gapratiotuple'][0] )
    n_epochPerGlRounds.append( epDict[0]['n_epoch'] )
    if 'wc_getGapsLocal' in epDict:
      wc.append( epDict['wc_getGapsLocal'] )
  series = []
  fig, ax = plt.subplots()
  ax.callbacks.connect('xlim_changed', peakfinding.on_xlims_change)
  ax.callbacks.connect('ylim_changed', peakfinding.on_ylims_change)
  # peakfinding.args = args
  g_train_loss, d_train_loss = loadTLArrNp(fnDict)
  plotOriginalData(g_train_loss, d_train_loss, ax)
  series.append( (g_train_loss, "Gen Tr Loss") )
  series.append( (d_train_loss, "Dis Tr Loss") )
  series.append( (r_dg, "D/G ratio") )
  series.append( (n_epochPerGlRounds, "n_epoch") )
  series.append( (gaps, "D G val loss gap") )
  ax.plot( r_dg, label="D/G ratio" )
  ax.plot( gaps, label="D G val loss gap" )
  ax.plot( n_epochPerGlRounds, label="n_epoch" )
  for t in ['args.d_lr', 'args.g_lr']:
    ax.plot(globalLr[t], label=lrToTextLabel[t])
    series.append( (globalLr[t], lrToTextLabel[t]) )
  for t in ['d', 'g']:
    ax.plot(globalValidLoss[t], label=tToTextLabel[t])
    series.append( (globalValidLoss[t], tToTextLabel[t]) )
  print( "restored_args", type(restored_args) )
  xlableText = "Global rounds test(" + str( len(r_dg) ) + ')'
  if 'isWAvg' in restored_args:
    print( "restored_args.man_resume_filepath", restored_args.man_resume_filepath )
    # print( "restored_args.isWAvg", restored_args.isWAvg )
    if restored_args.isWAvg:
      xlableText += " WAvg"
    else :
      xlableText += " plain Avg"
  plt.xlabel(xlableText)
  plt.legend()
  plt.show()
  progress_zoom.oName = os.path.basename( fnDict["sqlite"] )
  progress_zoom.plotZooms(series, isShow=False)
  pass
  local_wc, local_wc_acc, local_gaps = plotLrVloss(wc, globalLr, lrToTextLabel, globalValidLoss, tToTextLabel, local_dg)
  plotWWAvgVloss(wc, local_wc, local_wc_acc, local_gaps, isWc=True, isAcc=True)

def plotLrVloss(wc, globalLr, lrToTextLabel, globalValidLoss, tToTextLabel, local_dg):
  plt.clf()
  plt.close()
  # 3/hermes/-/blob/main/code/plotUtilsCmd.py#L59
  fig = plt.figure()
  host = fig.add_subplot()
  par1 = host.twinx()
  host.set_ylabel("learning rate", color = "m")
  handles = []
  colors=plotUtils.getGradColors('m', 4) # 4: to use 2 different colors out of 4
  # linestyle=linestyle_tuple[u%nlinestyles][1]
  next( linestyleCyle ) # avoid first style
  next( linestyleCyle ) # avoid first style
  for i,t in enumerate( ['args.d_lr', 'args.g_lr'] ):
    handles.extend( host.plot(globalLr[t], label=lrToTextLabel[t], color=colors[i], linestyle=next( linestyleCyle )[1] ) )
  par1.set_ylabel("val loss", color = "teal")
  colors=plotUtils.getGradColors('teal', 4)
  for i,t in enumerate( ['d', 'g'] ):
    handles.extend( par1.plot(globalValidLoss[t], label=tToTextLabel[t], color=colors[i], linestyle=next( linestyleCyle )[1] ) )
  # colors=plotUtils.getColorsCmap('tab20', h.NS+h.NU)
  print( "local_dg.keys()", local_dg.keys() )
  print( "len( wc )", len( wc ) )
  # linestyle=next( linestyleCyle )[1]
  # local_dg.keys() dict_keys([0, 1, 2])
  # for u in range(num_users):
  if len(wc) > 0:
    local_gaps = {}
    local_wc = {}
    local_wc_acc = {}
    for u in range(num_users):
      # for t in ['d_valid_loss', 'g_valid_loss']:
      dnp = np.array( local_dg[u]['d_valid_loss'] )
      gnp = np.array( local_dg[u]['g_valid_loss'] )
      local_gaps[u] = np.absolute( dnp - gnp )
      local_wc[u] = []
      local_wc_acc[u] = []
    for ep in range( len(wc) ):
      wc_sum = sum( wc[ep].values() )
      wc_cum = wc[ep][0]/wc_sum
      local_wc_acc[0].append( wc_cum )
      local_wc[0].append( wc_cum )
      # wc_cum += wc[ep][1]/wc_sum
      # local_wc[1].append( wc_cum )
      # wc_cum += wc[ep][2]/wc_sum
      # local_wc[2].append( wc_cum )
      for u in range(1, num_users):
        local_wc[u].append( wc[ep][u]/wc_sum )
        wc_cum += wc[ep][u]/wc_sum
        local_wc_acc[u].append( wc_cum )
    colors=plotUtils.getColorsCmap('tab20', num_users)
    ulinestyle=next( linestyleCyle )[1]
    for u in range(num_users):
      # ulinestyle=next( linestyleCyle )[1]
      ucolor=colors[u]
      handles.extend( par1.plot( local_gaps[u], label=str(u) + ", local_gaps", color=ucolor, linestyle=ulinestyle ) )
      handles.extend( par1.plot( local_wc_acc[u], label=str(u) + ", wc" , color=ucolor ) )
  print( "len(handles)", len(handles) )
  print( "type(handles[0])", type(handles[0]) )
  plt.legend(handles=handles)
  plt.show()
  return local_wc, local_wc_acc, local_gaps

def plotWWAvgVloss(wc, local_wc, local_wc_acc, local_gaps, isWc=False, isAcc=True):
  if len(wc) > 0:
    colors=plotUtils.getColorsCmap('tab20', num_users)
    # ulinestyle=next( linestyleCyle )[1]
    # ulinestyle=next( linestyleCyle )[1]
    plt.clf()
    plt.close()
    fig = plt.figure()
    host = fig.add_subplot()
    par1 = host.twinx()
    par1.set_ylabel("clients val loss gaps", color = "teal")
    host.set_ylabel("weights of the w. avg", color = "m")
    handles = []

    # host.set_ylim(-10, 110)
    host.yaxis.set_major_formatter(PercentFormatter(100))
    # par1.set_ylim(bottom=0)

    orangeColors=plotUtils.getGradColors('yellow', num_users+1)
    tealColors=plotUtils.getGradColors('teal', num_users+1)
    mColors=plotUtils.getGradColors('m', num_users+1)
    for u in range(num_users):
      npArr = np.array( local_wc_acc[u] )
      local_wc_acc[u] = npArr*100
      npArr = np.array( local_wc[u] )
      local_wc[u] = npArr*100
    for u in range(num_users):
      # ucolor=tealColors[u]
      print( "local_gaps[u][0], u", local_gaps[u][0], u )
      if isWc: handles.extend( host.plot( local_wc[u], label=str(u) + ", wc" , color=mColors[u] ) )
      if isAcc: handles.extend( host.plot( local_wc_acc[u], label=str(u) + ", wc_acc" , color=orangeColors[u], linestyle=next( linestyleCyle )[1] ) )
      handles.extend( par1.plot( local_gaps[u], label=str(u) + ", local_gaps", color=colors[u] ) ) # linestyle=next( linestyleCyle )[1]

    par1.set_ylim(bottom=-1)
    host.set_ylim(-10, 110)
    print("par1 y-limits:", par1.get_ylim())
    print("host y-limits:", host.get_ylim())
    plt.legend(handles=handles)
    plt.show()

oName = ""
epochs_global, num_users, num_iters_local, log_step, n_critic = 0,0,0,0,0
now = datetime.datetime.now() ; formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--logdatetime', nargs='+', type=str)
  parser.add_argument('--ep_range', nargs='+', type=int, default=[0, 1]) # stop: not included ; https://www.w3schools.com/python/ref_func_range.asp
  args = parser.parse_args()
  # logs = args.logs
  logsDict = {}
  # global epochs_global, num_users, num_iters_local
  fnDict = getFilenames(args.logdatetime[0]) ; print( "fnDict", fnDict )
  pD = SqliteDict( fnDict['sqlite'] )
  print( "pD.keys()", pD.keys() )
  restored_args = getArgsSqliteDict(pD)
  print( "epochs_global, num_users, num_iters_local, log_step, n_critic", ": ", epochs_global, num_users, num_iters_local, log_step, n_critic )
  xYDict = getXYDict(pD)
  print( "len(pD)", len(pD) )
  print( "pD.keys()", pD.keys() )
  start = args.ep_range[0]
  plotGlobalVLoss(pD, fnDict, restored_args)
