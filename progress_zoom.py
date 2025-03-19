
import progress_parser
import peakfinding
import datetime
import argparse
import statistics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os

def getXYDict(pD):
  xYDict = {}

def plotGlobalVLoss(pD, isStr, args):
  globalValidLoss = {}
  tToTextLabel = {'d':'Dis Val Loss', 'g':'Gen Val Loss'}
  for t in ['d', 'g']:
    globalValidLoss[t] = []
  for ep in range(epochs_global):
    if isStr: ep = str(ep)
    for t in ['d', 'g']:
      globalValidLoss[t].append( statistics.mean( [pD[ep]['getGlobalValidLoss'][u][t] for u in range(num_users)] ) )

  series = []
  fig, ax = plt.subplots()
  ax.callbacks.connect('xlim_changed', peakfinding.on_xlims_change)
  ax.callbacks.connect('ylim_changed', peakfinding.on_ylims_change)
  peakfinding.args = args
  g_train_loss, d_train_loss = peakfinding.plotdgloss(args, isReturn=True)
  peakfinding.plotOriginalData(g_train_loss, d_train_loss, ax)
  series.append( (g_train_loss, "Gen Tr Loss") )
  series.append( (d_train_loss, "Dis Tr Loss") )
  for t in ['d', 'g']:
    ax.plot(globalValidLoss[t], label=tToTextLabel[t])
    series.append( (globalValidLoss[t], tToTextLabel[t]) )
  plt.legend()
  plt.show()
  plotZooms(series, isShow=False)

def plotZooms(series, isShow=False):
  xylimsOnChange = peakfinding.xylimsOnChange
  lx = xylimsOnChange['x'][0][1] - xylimsOnChange['x'][0][0]
  ly = xylimsOnChange['y'][0][1] - xylimsOnChange['y'][0][0]
  oa = lx*ly*0.9
  print(oa)
  imgsPaths = []
  if len( xylimsOnChange['x'] ) > 1:
    nz = len( xylimsOnChange['x'] ) # nz = number of zooms
    for i in range( 1, nz ):
      lx = xylimsOnChange['x'][i][1] - xylimsOnChange['x'][i][0]
      ly = xylimsOnChange['y'][i][1] - xylimsOnChange['y'][i][0]
      if lx*ly <= oa:
        print( "xylimsOnChange", i, xylimsOnChange['x'][i], [ int(ii) for ii in xylimsOnChange['x'][i] ] )
        plt.clf()
        plt.close()
        fig, main_ax = plt.subplots(); main_ax.set_box_aspect(0.5)
        inset_ax = main_ax.inset_axes(
          [0.05, 0.65, 0.3, 0.3],  # [x, y, width, height] w.r.t. axes
           xlim=xylimsOnChange['x'][i], ylim=xylimsOnChange['y'][i], # sets viewport & tells relation to main axes
        ) # yticklabels=[]
        for ax in main_ax, inset_ax:
          for s in series:
            ax.plot(s[0], label=s[1])
        main_ax.legend()
        nr = len(series[0][0])
        main_ax.set_xlabel("Global rounds (" + str( nr ) + ')')
        main_ax.set_ylabel("Loss")
        main_ax.indicate_inset_zoom(inset_ax, edgecolor="blue")
        if isShow:
          plt.show()
        else:
          xlims = [ str(int(ii)) for ii in xylimsOnChange['x'][i] ]
          imgsPath = "fedgan5/img/" +oName +'-nr.' + str( nr ) +'-xlims.' +xlims[0] +'-'+ xlims[1]+ ".png"
          plt.savefig( imgsPath )
          imgsPaths.append( imgsPath )
        pass
  return imgsPaths
  pass

oName = ""
epochs_global, num_users, num_iters_local = 0,0,0
now = datetime.datetime.now() ; formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--logs', nargs='+', type=str)
  parser.add_argument('--range', nargs='+', type=int, default=[0, 1]) # stop: not included ; https://www.w3schools.com/python/ref_func_range.asp
  parser.add_argument('--gtl_arr_fp', type=str, default="fedgan5/Gen-loss-FedAvg.txt")
  parser.add_argument('--dtl_arr_fp', type=str, default="fedgan5/Dis-loss-FedAvg.txt")
  args = parser.parse_args()
  logs = args.logs
  logsDict = {}
  # global epochs_global, num_users, num_iters_local
  for log in logs:
    oName = os.path.basename(log)
    pD = progress_parser.getPD(fn = log)
    epochs_global, num_users, num_iters_local = \
      progress_parser.epochs_global, \
      progress_parser.num_users, \
      progress_parser.num_iters_local
    xYDict = getXYDict(pD)
    print( "len(pD)", len(pD) )
    print( "pD.keys()", pD.keys() )
    start = args.range[0]
    isStr = False
    if start not in pD:
      start = str(start)
      isStr = True
      print( "start not in pD", start not in pD )
      print( "start", start )
    print( "len(pD[start]), start", len(pD[start]), start )
    print( "pD[start].keys()", pD[start].keys() )
    plotGlobalVLoss(pD, isStr, args)
