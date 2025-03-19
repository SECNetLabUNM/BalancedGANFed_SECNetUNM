import pickle
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from datetime import datetime
import time
from molecular_dataset import MolecularDataset
from  data_smiles.drugbank import genPkl

import os
htmlStyle="""
<style>._txt_smlw {width:40px;float:left; margin:1px; padding:0px;border:solid 1px black;overflow:hidden;}</style>
<style>._txt_200 {width:200px;float:left; margin:1px; padding:0px;border:solid 1px black;overflow:hidden;}</style>
<style>._txt_400orig {color:#ff3399;font-family: "", Arial Black;height:300px;width:400px;float:left; margin:1px; padding:0px;border:solid 1px black;overflow:hidden;}</style>
"""

import difflib
import pprint

# unm2h/S24/lobophysionet/huggingface/course/ch2-6.Putting-it-all-together.py
def printObjectAttrs(obj):
  print( "", type( obj ) )
  for i,a in enumerate( vars(obj) ):
    print(i,a)

def printNdiff(a,b):
  if len(a) < 10 : print('{} => {}'.format(a,b))
  elif len(b) < 10 : print('{} => {}'.format(a,b))
  else : print('{}.. \n\t=> {}..'.format(a[:9],b[:9]))
  for i,s in enumerate(difflib.ndiff(a, b)):
    if s[0]==' ': continue
    elif s[0]=='-':
        print(u'Delete "{}" from position {}'.format(s[-1],i))
    elif s[0]=='+':
        print(u'Add "{}" to position {}'.format(s[-1],i))
  print()

def testMolecularDatasets(): # ds: datasets
    filenames = [
      'data-smiles/qm8_smiles.pkl',
      'data-smiles/qm9_smiles.pkl',
      'data-smiles/esol_smiles.pkl',
    ]
    for filename in filenames:
      with open(filename, 'rb') as f:
        pr = pickle.load(f) # pr: pickle load result
        print("pickle.load", filename, type(pr))
        for i,line in enumerate(pr):
          if i < 3:
            print(type(line), line)
      print( "pickle.load", filename, len(pr) )

    data = MolecularDataset()
    print( "MolecularDataset , __dict__", data.__dict__ )
    printObjectAttrs(data)

    filename = 'data_smiles/esol.dataset'
    data.load(filename) # args.mol_data_dir
    print( "MolecularDataset , __dict__", "after load", data.__dict__.keys() )
    printObjectAttrs(data)

MolecularDatasetkeys = list()
def getDicts():
    global MolecularDatasetkeys
    data = MolecularDataset()
    filename = 'data_smiles/qm8-diabetes-drugbank.pkl.dataset'
    data.load(filename) # args.mol_data_dir
    printObjectAttrs(data)
    MolecularDatasetkeys = data.__dict__.keys()

    filenameMDsDict = {} # MDs: MolecularDataset
    filenameMDsObjDict = {} # MDs: MolecularDataset

    for filename in filenames:
      # data.write("<td valign=top>" + filename + "</td>")
      mdata = MolecularDataset()
      ds_filename = filename + '.dataset'
      filenameMDsDict[filename] = {}
      mdata.load(ds_filename)
      for k in MolecularDatasetkeys:
        filenameMDsDict[filename][k] = mdata.__dict__[k]
      filenameMDsObjDict[filename] = mdata

    genPkl.readNameSmilesDicts()
    fieldsList = ["GetNumAtoms", "MolToSmiles", ]
    fieldsNameDict = {}
    fieldsNameDict["GetNumAtoms" ] = "max_length<br>GetNumAtoms"
    fieldsNameDict["MolToSmiles" ] = "max(len(Chem.MolToSmiles(mol))"
    fieldsDict = {}
    for filename in filenames:
      mdata = filenameMDsObjDict[filename]
      # max_length = max(mol.GetNumAtoms() for mol in mdata.data)
      fieldsDict[filename] = {}
      fieldsDict[filename]["GetNumAtoms"] =  max(mol.GetNumAtoms() for mol in mdata.data)
      fieldsDict[filename]["MolToSmiles"] =  max(len(Chem.MolToSmiles(mol)) for mol in mdata.data)

    return fieldsList, fieldsDict, fieldsNameDict, filenameMDsDict, filenameMDsObjDict

filenames = [
  # 'data-smiles/qm8_smiles.pkl',
  # 'data-smiles/qm9_smiles.pkl',
  # 'data-smiles/esol_smiles.pkl',
  # 'data_smiles/drugbank/diabetes-drugbank.pkl',
  'data_smiles/qm8-diabetes-drugbank.pkl',
]

MDsKeysToSkip = [
  "smiles",
  "data_S",
  "data_A",
  "data_X",
  "data_D",
  "data_F",
  "data_Le",
  "data_Lv",
  "all_idx",
  "train_idx",
  "validation_idx",
  "test_idx",
  ] # MDs: MolecularDataset


def dsCompareToHtml(fieldsList, fieldsDict, fieldsNameDict, filenameMDsDict, filenameMDsObjDict): # ds: datasets
    with open("data_smiles"+os.sep+'MolecularDataset.html', mode ='w') as data:
      data.write(htmlStyle)
      data.write("<table BORDER=\"1\">")
      data.write("<tr>")
      data.write("<td>" + "</td>")

      for filename in filenames:
        data.write("<td valign=top>" + filename + "</td>")
      data.write("</tr>")

      for k in MolecularDatasetkeys:
        data.write("<tr>")
        data.write("<td>" +k+ "</td>")
        for filename in filenames:
          # print( ds_filename, "atom_num_types", data.atom_num_types )
          if k in MDsKeysToSkip:
            data.write("<td>" + "len:" + str( len( filenameMDsDict[filename][k] ) ) + "</td>")
          elif k == "data":
            data.write("<td>" + "type:" + str( type( filenameMDsDict[filename][k] ) )+ "len:" + str( len( filenameMDsDict[filename][k] ) ) + "</td>")
          else:
            data.write("<td>" + str( filenameMDsDict[filename][k] ) + "</td>")
          # print( ds_filename, "",  )
        data.write("</tr>")

      for field in fieldsList:
        data.write("<tr>")
        data.write("<td>" + fieldsNameDict[field] + "</td>")
        for filename in filenames:
          data.write("<td>" + str( fieldsDict[filename][field] ) + "</td>")
        data.write("</tr>")
      data.write("</table>")

def testDrugbankMD(filenameMDsObjDict): # MD: MolecularDataset
      filename = filenames[-1]
      mdata = filenameMDsObjDict[filename]
      max_length_s = max(len(Chem.MolToSmiles(mol)) for mol in mdata.data)
      smile_max_length_s = max([Chem.MolToSmiles(mol) for mol in mdata.data], key = len)
      rSerialSmilesDictCanon = { Chem.CanonSmiles(value) : key for key, value in genPkl.fnDict["serialSmilesDict.txt.py"].items()} # r: reversed
      rSerialSmilesDict = { value : key for key, value in genPkl.fnDict["serialSmilesDict.txt.py"].items()} # r: reversed
      # print( "[genPkl.serialSmilesDict.items()][0]", [genPkl.serialSmilesDict.items()][0] )
      # print( "[reversed_dict.keys()][0]", [reversed_dict.keys()][0] )
      # print( "len(genPkl.serialSmilesDict)", len(genPkl.serialSmilesDict) )
      print( "len(genPkl.fnDict[\"serialSmilesDict.txt.py\"])", len(genPkl.fnDict["serialSmilesDict.txt.py"]) )
      print( "len(reversed_dict)", len(rSerialSmilesDict) )
      print( "[*reversed_dict][0]", [*rSerialSmilesDict][0] )

      print( "Chem.CanonSmiles(smile_max_length_s)", len( Chem.CanonSmiles(smile_max_length_s) ) )
      print( "smile_max_length_s", len( smile_max_length_s ) )
      a = smile_max_length_s
      b = Chem.CanonSmiles(smile_max_length_s)
      # print( "difflib.ndiff: CanonSmiles", len( difflib.ndiff(a, b) ) )
      # print( "difflib.ndiff: CanonSmiles", sum( 1 for x in difflib.ndiff(a, b) ) )
      printNdiff(a,b)
      serial_max_length_s = rSerialSmilesDictCanon[ Chem.CanonSmiles(smile_max_length_s) ]
      printObjectAttrs( mdata.data[0] )
      print( filename
        , "\n\ttype( mdata.data[0] )", type( mdata.data[0] )
        , "\n\tmdata.data[0].__dict__", pprint.pformat(mdata.data[0].__dict__, sort_dicts=False)
        , "\n\tmax_length_s", max_length_s
        , "\n\tlen( smile_max_length_s )", len( smile_max_length_s )
        , "\n\tserial", serial_max_length_s
        , "\n\tname", genPkl.fnDict["serialNameDict.txt.py"][ serial_max_length_s ] )

if __name__ == '__main__':
    # testMolecularDatasets()
    fieldsList, fieldsDict, fieldsNameDict, filenameMDsDict, filenameMDsObjDict = getDicts()
    dsCompareToHtml(fieldsList, fieldsDict, fieldsNameDict, filenameMDsDict, filenameMDsObjDict)
    testDrugbankMD(filenameMDsObjDict)

