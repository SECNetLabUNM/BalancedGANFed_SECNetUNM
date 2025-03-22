
# environment setup
```
# apple silicon macos
eval "$(/Users/watney/miniconda3/bin/conda shell.zsh hook)"
```
## references:
* https://developer.apple.com/metal/pytorch/


```
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
```

# basic training commands
```
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

# generate non-iid datasets
```
# the following will generate a non-iid dataset with the default alpha value:
% python3 check_iid.py
2025-03-20 16:34:56 Created 2456 features and adjacency matrices  out of 2456 molecules!
3017196it [00:00, 5681309.45it/s]
pathList ['data_smiles/noniid/split-15095-2444-4434/3/0.pkl.dataset', 'data_smiles/noniid/split-15095-2444-4434/3/1.pkl.dataset', 'data_smiles/noniid/split-15095-2444-4434/3/2.pkl.dataset']

# use text for the pathList above to generate another non-iid dataset:
% python3 dirichlet.py --splitText 15095-2444-4434
alphaKey 5
['2025-03-20_16-38-52']
['iid', 'nonIid-0.5', 'nonIid-5']

# previous command only generate a split with alpha = 5
# without actually generate the data structure to be used in training
# so use the following command to generate the data structure (file with .dataset extension)
# by using the console out from previous command as formatted_date argument to this next command

% python3 dirichlet_gen_dataset.py --formatted_date 2025-03-20_16-38-52  --existingdatasetid  15095-2444-4434
2025-03-20 16:46:48 Creating features and adjacency matrices..
38032281it [00:03, 21176038.64it/s]
2025-03-20 16:46:52 Created 8762 features and adjacency matrices  out of 8762 molecules!
38390703it [00:03, 12040437.05it/s]
pathList ['data_smiles/noniid/split-7092-6131-8750/3/0.pkl.dataset', 'data_smiles/noniid/split-7092-6131-8750/3/1.pkl.dataset', 'data_smiles/noniid/split-7092-6131-8750/3/2.pkl.dataset']

# finally, use this command to generate a pdf shown the corresponding iid and non iid datasets
% python3 dirichlet_gen_pdf.py  --formatted_date 2025-03-20_14-58-15 --existingdatasetid 9905-2465-9603

```
* in the section above, commands begin with a % sign and those lines without % in front are console outputs

# moving existing training setup to another computer and train there
```
2025-03-20 15:57:36 Created 9128 features and adjacency matrices  out of 9128 molecules!
41664756it [00:05, 8311705.77it/s]
pathList ['data_smiles/noniid/split-6477-6380-9116/3/0.pkl.dataset', 'data_smiles/noniid/split-6477-6380-9116/3/1.pkl.dataset', 'data_smiles/noniid/split-6477-6380-9116/3/2.pkl.dataset']
[20/Mar/2025 16:28:48] "GET /fedgan5/img/StackedSubplots2xCol.2025-03-20_14-58-15.pdf HTTP/1.1" 200 -
 1751  python3 dirichlet_gen_dataset.py --formatted_date 2025-03-20_14-58-15
 1761  python3 dirichlet_gen_dataset.py --formatted_date 2025-03-20_14-58-15 --existingdatasetid 9905-2465-9603
# according to these shell command history and console output
# 9905-2465-9603 is the beta = 0.5 dataset
# 6477-6380-9116 is the beta = 5 dataset generated subsequently
# and the following is an example of beginning training at another computer:
#
$ mkdir -p data_smiles/noniid/
$ scp -rp secnet@192.168.1.21:/home/secnet/git/BalancedGANFed_SECNetUNM/data_smiles/noniid/split-6477-6380-9116/ data_smiles/noniid/
$ mkdir -p fedgan5/models/init/
$ scp -rp secnet@192.168.1.21:/home/secnet/git/BalancedGANFed_SECNetUNM/fedgan5/models/init/2025-03-19_05-21-05 fedgan5/models/init/
$ python3 trainer_test.py --cmd train   --epochs_global 500   --isWAvg True --man_resume_filepath fedgan5/models/init/2025-03-19_05-21-05/    --isNonIid True   --nonIidDatasets data_smiles/noniid/split-6477-6380-9116/3/0.pkl.dataset  data_smiles/noniid/split-6477-6380-9116/3/1.pkl.dataset   data_smiles/noniid/split-6477-6380-9116/3/2.pkl.dataset    2>/dev/null

```

# monitor training results using ansible
## setup
```
sudo apt install ansible

ssh-keygen -t rsa
# copy public keys to remote computers
ssh-copy-id watney@10.88.215.?
ssh-copy-id seclab@192.168.1.?
ssh-copy-id secnet@192.168.1.?
```

## ansible commands
```
# check the sqlite files containing training logs at each compute
ansible-playbook find_sqlite.yml -i hosts-current.ini  --tags progress_dict | grep progr

# do git pull on all computers
ansible-playbook find_sqlite.yml -i hosts-current.ini  --tags git_pull

# git pull on specific computer
ansible-playbook find_sqlite.yml -i hosts-current.ini  --tags git_pull  --limit server1080

# to fix errors, not needed under normal circumstances
ansible-playbook find_sqlite.yml -i hosts-current.ini  --tags fix_git_remote

```

