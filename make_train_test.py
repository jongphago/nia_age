import os
import glob as glob
import numpy as np
from shutil import copyfile
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


file_list = []
all_file = []
for i in Path('./age_db').glob('*.jpg'):
    age = str(i)
    all_file.append(age)
    age = age.split('/')[1] 
    age = age.split('_')[2] 
    file_list.append(age)
   
id_num = np.array(file_list)  

gss = GroupShuffleSplit(n_splits=2, train_size = .7, random_state=42)

X = np.ones(shape=(len(id_num), 2))
y = np.ones(shape=(len(id_num), 1))


train_idx, test_idx = gss.split(X,y,id_num)

train_idx_test = train_idx[0]
test_idx_test = train_idx[1]

for j in train_idx_test:
    train_name = all_file[j] 
    train_name_path = train_name.split('/')[1]    
    target = './age_db_train/' + train_name_path    
    copyfile(train_name, target)
     
for k in test_idx_test:
    test_name = all_file[k]
    test_name_path = test_name.split('/')[1]
    target_t = './age_db_test/' + test_name_path
    copyfile(test_name, target_t)
    
