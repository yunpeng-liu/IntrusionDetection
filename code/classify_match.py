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

