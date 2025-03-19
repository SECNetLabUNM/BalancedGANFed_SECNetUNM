import pickle
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from datetime import datetime
import time
from molecular_dataset import MolecularDataset

if __name__ == '__main__':

  filenames = [
    'data-smiles/qm8_smiles.pkl',
    'data-smiles/qm9_smiles.pkl',
    'data-smiles/esol_smiles.pkl',
    'data_smiles/drugbank/diabetes-drugbank.pkl',
    'data_smiles/qm8-diabetes-drugbank.pkl',
  ]
  for filename in filenames:
    data = MolecularDataset()
    data.generate( filename, validation=0.1, test=0.1 ) # data_smiles\\esol_smiles.pkl
    data.save( filename + '.dataset' )
