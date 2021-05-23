#%%
import json
import logging
import os
import pickle
import random
from collections import defaultdict
from multiprocessing import Pool
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from tqdm import tqdm

plt.style.use(['science', 'grid'])
plt.rc('figure', dpi=200)

seed = 42
random.seed(seed)
np.random.seed(seed)

logging.basicConfig(level=logging.INFO)

#%%
train_df = pd.read_csv('../features/flow_feature_train.csv')
test_df = pd.read_csv('../features/flow_feature_test.csv')

# %%
def read_label(phase=1):
    if phase == 1:
        label_folder = '../dataset/phase1_label'
    else:
        label_folder = '../dataset/phase2_label'
    black = open(os.path.join(label_folder, 'blackIPlist.txt')).readlines()
    white = open(os.path.join(label_folder, 'whiteIPlist.txt')).readlines()
    return set(i.strip() for i in black), set(i.strip() for i in white)


def label_host(ips: list, phase=1):
    if phase == 1:
        label_folder = '../dataset/phase1_label'
    else:
        label_folder = '../dataset/phase2_label'
    black = open(os.path.join(label_folder, 'blackIPlist.txt')).readlines()
    white = open(os.path.join(label_folder, 'whiteIPlist.txt')).readlines()
    black = set(i.strip() for i in black)
    return [float(i in black) for i in ips]

#%%

def preprocess_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame, phase=1, val_ratio=0.1):
    black, white = read_label(phase)
    columns = train_df.columns[3:]

    # scale features and remove unimportant features
    train = train_df.copy()
    test = test_df.copy()
    
    is_black = train_df['sip'].apply(lambda i: i in black)
    is_white = train_df['sip'].apply(lambda i: i in white)
    is_labeled = is_black | is_white

    train_data = train[is_labeled]

    labeled_train_data = train_df[is_labeled]
    label = is_black.to_numpy()[is_labeled.to_numpy()].astype(np.float)

    if val_ratio > 0:
        train_data, val_data, train_label, val_label = train_test_split(train_data, label, test_size=val_ratio, shuffle=True, random_state=seed, stratify=label)
    else:
        train_label = label
        val_data = val_label = None
    return train_data, train_label, val_data, val_label, test


train_data, train_label, val_data, val_label, test_data = preprocess_dataset(train_df, test_df, val_ratio=0)

# %%
black_train_flows = train_data.loc[train_label.astype(bool)]
black_train_ips = black_train_flows[['sip', 'dip']].groupby('sip').agg(set)['dip']

#%%
dip_black_score = train_data.loc[train_label.astype(bool)][['sip', 'dip']].groupby('dip').agg('count')['sip'].to_dict()
dip_white_score = train_data.loc[~train_label.astype(bool)][['sip', 'dip']].groupby('dip').agg('count')['sip'].to_dict()

# %%
"""
th      score
0.9     73.85
0.8     73.75
0.5     72.95
"""

test_matched_ips = []
for sip, dips in test_data[['sip', 'dip']].groupby('sip').agg(set)['dip'].iteritems():
    dip_scores = []
    for dip in dips:
        bs = dip_black_score.get(dip, 1e-5)
        ws = dip_white_score.get(dip, 1e-5)
        dip_scores.append(bs / (bs + ws))
    if max(dip_scores) > 0.9:
        test_matched_ips.append(sip)

len(test_matched_ips)
lgb_ips = open('../result_lgb.txt', 'r').readline().split(' ')

with open('../result.txt', 'w') as f:
    result = set(test_matched_ips) | set(lgb_ips)
    print(len(result))
    f.write(' '.join(result))