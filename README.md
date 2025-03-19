
```
conda install -c conda-forge rdkit
conda install -c conda-forge tqdm
conda install scikit-learn
conda install -c conda-forge rdkit
conda install -c conda-forge tqdm
conda install scikit-learn 
conda install -c conda-forge gitpython
pip install sqlitedict

# generate dataset from pkl (list of smiles)
python3 molecular_dataset_linux.py

# test training using small number of global epochs and num_iters_local
python3 trainer_test.py --cmd train   --epochs_global  5  --isFL True  --isWAvg True  --num_iters_local 20  2>/dev/null

```
