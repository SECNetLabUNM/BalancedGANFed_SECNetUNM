
import os
import pprint
import pickle
import numpy as np
import ast

serialNameDict = {}
serialSmilesDict = {}
noSmilesList = []

fnDict = {}
fnDict["serialNameDict.txt.py"] = serialNameDict
fnDict["serialSmilesDict.txt.py"] = serialSmilesDict
fnDict["noSmilesList.txt.py"] = noSmilesList

def smilesToPkl():
  smilesList = []
  for i,kv in enumerate( serialSmilesDict.items() ):
    k,v = kv
    smilesList.append(v)

  filename = "diabetes-drugbank.pkl"
  with open(filename, 'wb') as f:
    pickle.dump( np.array(smilesList) , f)

def readNameSmilesDicts():
  for i,kv in enumerate( fnDict.items() ):
    k,v = kv
    with open(k) as f:
      data = f.read()
    v = ast.literal_eval(data)
    fnDict[k] = v
    print( k, len(v) )

def writeNameSmilesDicts():
  for i,kv in enumerate( fnDict.items() ):
    k,v = kv
    with open(k, mode ='w') as data:
      data.write( pprint.pformat(v, sort_dicts=False) + '\n' )

def populateNameSmilesDicts():
  path = "diabetes"
  dir_list = os.listdir(path)
  # print(dir_list)

  fn = "diabetes-drugbank.txt"
  with open(fn, mode ='r') as data:
    global serialNameDict, serialSmilesDict, noSmilesList
    lines = data.readlines()
    print( fn, len(lines) )
    for l in lines:
      parts = l.split('>')
      k = parts[0].replace('"', '').split('/')[-1]
      serialNameDict[k] = parts[1].rstrip()

  for fn in dir_list:
    with open(path+os.sep+fn, mode ='r') as data:
      lines = data.readlines()
      # print( fn, len(lines) )
      parts = fn.split('.')
      if len(lines) > 0:
        serialSmilesDict[parts[0]] = lines[0].rstrip()
        # print(fn, parts[0], serialSmilesDict[parts[0]] )
      else:
        # print(fn, serialNameDict[parts[0]] )
        noSmilesList.append( parts[0] )
        pass

def main():
  populateNameSmilesDicts()
  writeNameSmilesDicts()
  readNameSmilesDicts()
  smilesToPkl()

if __name__ == '__main__':
  main()
