import molecular_dataset_test
from rdkit import Chem

# 3\Documents\unm2h\git\graphganfeddrugbank\molecularGAN\GraphGANFed\molecular_dataset_merger.py
def getAtomicNumRepresentative(filenames = ['data-smiles/qm8_smiles.pkl','data_smiles/drugbank/diabetes-drugbank.pkl']) :
  fieldsList, fieldsDict, fieldsNameDict, filenameMDsDict, filenameMDsObjDict = molecular_dataset_test.getDicts()
  smilesList = []
  mols = []
  atomicNumRepresentative = {} # atom.GetAtomicNum()
  filenames = ['data_smiles/qm8-diabetes-drugbank.pkl']
  for filename in filenames:
    mdata = filenameMDsObjDict[filename]
    for mol in mdata.data:
      if mol.GetNumAtoms() < 33:
        mols.append(mol)
        mSmil = Chem.CanonSmiles( Chem.MolToSmiles(mol) )
        smilesList.append( mSmil )
        for atom in mol.GetAtoms():
          an = atom.GetAtomicNum()
          # print( "an = atom.GetAtomicNum()", type( an ) )
          if an not in atomicNumRepresentative:
            atomicNumRepresentative[an] = mSmil

  # print( "len(smilesList)", len(smilesList) )
  # print( "atomicNumRepresentative", atomicNumRepresentative )
  # print( "atomicNumRepresentative", len(atomicNumRepresentative) )
  return atomicNumRepresentative

if __name__ == '__main__':
  import pprint
  atomicNumRepresentative = getAtomicNumRepresentative()
  print( "atomicNumRepresentative", len(atomicNumRepresentative) )
  print( pprint.pformat(atomicNumRepresentative, sort_dicts=False) )

