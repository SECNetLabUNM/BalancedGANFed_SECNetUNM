
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import argparse
import os
import datetime

# from trainer_test import getArgs
def getArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument('--gtl_arr_fp', type=str, default="fedgan5/Gen-loss-FedAvg.txt")
  parser.add_argument('--dtl_arr_fp', type=str, default="fedgan5/Dis-loss-FedAvg.txt")
  args = parser.parse_args()
  return args

def plotdgloss(args, isReturn=False
    ): # gtl_arr_fp = g_train_loss_array file path
  g_train_loss_array = np.loadtxt(args.gtl_arr_fp)
  g_train_loss = g_train_loss_array.tolist()
  d_train_loss_array = np.loadtxt(args.dtl_arr_fp)
  d_train_loss = d_train_loss_array.tolist()

  if isReturn:
    return g_train_loss, d_train_loss
  else:
    plt.plot(g_train_loss, label="Generator")
    plt.plot(d_train_loss, label="Discriminator")
    plt.legend()
    nr = len(d_train_loss) # nr: number of global rounds
    # plt.xticks(np.arange(0, nr, 20) )
    plt.xlabel("Global rounds (" + str( nr ) + ')')
    plt.ylabel("Loss")
    plt.savefig("Gen-Dis-Loss-for-FedAvg.png")

# https://stackoverflow.com/questions/31490436/matplotlib-finding-out-xlim-and-ylim-after-zoom
def on_xlims_change(event_ax):
  # print("updated xlims: ", event_ax.get_xlim())
  xylimsOnChange['x'].append( event_ax.get_xlim() )

def on_ylims_change(event_ax):
  # print("updated ylims: ", event_ax.get_ylim())
  xylimsOnChange['y'].append( event_ax.get_ylim() )

xylimsOnChange = {}
xylimsOnChange['x'] = []
xylimsOnChange['y'] = []

def plotOriginalData(g_train_loss, d_train_loss, ax):
  ax.plot(g_train_loss, label="Generator")
  ax.plot(d_train_loss, label="Discriminator")
  ax.legend()
  nr = len(d_train_loss)
  ax.set_xlabel("Global rounds (" + str( nr ) + ')')
  ax.set_ylabel("Loss")

import outset
def plotZoomsOutsetGrid(g_train_loss, d_train_loss, isShow=False):
  plt.clf()
  plt.close()
  lx = xylimsOnChange['x'][0][1] - xylimsOnChange['x'][0][0]
  ly = xylimsOnChange['y'][0][1] - xylimsOnChange['y'][0][0]
  oa = lx*ly*0.9 # oa" original area
  if len( xylimsOnChange['x'] ) > 1:
    nz = len( xylimsOnChange['x'] ) # nz = number of zooms
    grids = []
    for i in range( 1, nz ):
      lx = xylimsOnChange['x'][i][1] - xylimsOnChange['x'][i][0]
      ly = xylimsOnChange['y'][i][1] - xylimsOnChange['y'][i][0]
      if lx*ly <= oa:
        x0,x1 = xylimsOnChange['x'][i]
        y0,y1 = xylimsOnChange['y'][i]
        grids.append( (x0, y0, x1, y1) ) # as (x0, y0, x1, y1) ; https://towardsdatascience.com/a-comprehensive-guide-to-inset-axes-in-matplotlib-87400e00a4e5
    grid = outset.OutsetGrid( grids )
    grid.broadcast(plt.plot,
      g_train_loss,
      c="mediumblue",  zorder=-1,
      )
    grid.broadcast(plt.plot,
      d_train_loss,
      zorder=-1, label="Discriminator"
      )
    # outset.inset_outsets(grid, insets="NW")
    grid.marqueeplot()
    # plt.show()
    if isShow:
      plt.show()
    else:
      imgsPath = "fedgan5/img/" +oName +'-' + formatted_date +'-' + "OutsetGrid.png"
      plt.savefig( imgsPath )
      return imgsPath
    pass

def plotZooms(g_train_loss, d_train_loss, isShow=False):
  # plt.clf()
  lx = xylimsOnChange['x'][0][1] - xylimsOnChange['x'][0][0]
  ly = xylimsOnChange['y'][0][1] - xylimsOnChange['y'][0][0]
  oa = lx*ly*0.9 # oa" original area
  # fig, main_ax = plt.subplots(); main_ax.set_box_aspect(0.5) 
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
           yticklabels=[]
        )
        for ax in main_ax, inset_ax:
          ax.plot(g_train_loss, label="Generator")
          ax.plot(d_train_loss, label="Discriminator")
        main_ax.legend()
        nr = len(d_train_loss)
        main_ax.set_xlabel("Global rounds (" + str( nr ) + ')')
        main_ax.set_ylabel("Loss")
        main_ax.indicate_inset_zoom(inset_ax, edgecolor="blue")
        if isShow:
          plt.show()
        else:
          xlims = [ str(int(ii)) for ii in xylimsOnChange['x'][i] ]
          imgsPath = "fedgan5/img/" +oName +'-' +xlims[0] +'-'+ xlims[1]+ ".png"
          plt.savefig( imgsPath )
          imgsPaths.append( imgsPath )
  return imgsPaths

oName = ""
now = datetime.datetime.now() ; formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")
if __name__ == '__main__':
  args = getArgs()
  g_train_loss, d_train_loss = plotdgloss(args, isReturn=True)
  fig, ax = plt.subplots()
  ax.callbacks.connect('xlim_changed', on_xlims_change)
  ax.callbacks.connect('ylim_changed', on_ylims_change)
  plotOriginalData(g_train_loss, d_train_loss, ax)
  plt.show()
  # print(xylimsOnChange)
  oName = os.path.basename(args.gtl_arr_fp)
  print( plotZooms(g_train_loss, d_train_loss) )
  print( plotZoomsOutsetGrid(g_train_loss, d_train_loss) )

