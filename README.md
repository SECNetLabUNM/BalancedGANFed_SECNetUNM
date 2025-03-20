
```
# apple silicon macos
eval "$(/Users/watney/miniconda3/bin/conda shell.zsh hook)"

# conda
conda create -n torch11 python=3.11
conda activate torch11

# 
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
mkdir -p fedgan5/logs/
python3 trainer_test.py --cmd train   --epochs_global  5  --isFL True  --isWAvg True  --num_iters_local 20  2>/dev/null

# monitor training status
pip install outset
ls -laht | head
# create the folder for zoomed in figures to be saved
mkdir fedgan5/img/
python3 progress_zoom_sqlitedict.py --logdatetime  25-03-19_05-21-05

# use the same init weights
mkdir -p fedgan5/models/init/2025-03-19_05-21-05
scp watney@10.88.215.19:/Users/watney/git/BalancedGANFed_SECNetUNM/fedgan5/models/2025-03-19_05-21-05/init-*  fedgan5/models/init/2025-03-19_05-21-05/
python3 trainer_test.py --cmd train   --epochs_global  500  --isFL True  --man_resume_filepath fedgan5/models/init/2025-03-19_05-21-05/  --isWAvg True  --isFixedRatio  1 5 1 2>/dev/null

```
