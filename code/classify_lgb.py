#%%
import json
import logging
import os
import pickle
import random
from collections import defaultdict
from multiprocessing import Pool
from pprint import pprint

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from tqdm import tqdm
import warnings

plt.style.use(['science', 'grid'])
plt.rc('figure', dpi=200)

seed = 42
random.seed(seed)
np.random.seed(seed)

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

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
    scale_cols = []
    for col in columns:
        if not train[col].value_counts().size < 10:
            scale_cols.append(col)

    scaler = StandardScaler()
    scaler.fit(train.loc[:, scale_cols])
    train.loc[:, scale_cols] = scaler.transform(train.loc[:, scale_cols])
    test.loc[:, scale_cols] = scaler.transform(test.loc[:, scale_cols])
    
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


train_data, train_label, val_data, val_label, test_data = preprocess_dataset(train_df, test_df, phase=2, val_ratio=0.1)

#%%
#! parameter selection
def param_selection(train_data, train_label):
    data_train = lgb.Dataset(train_data.iloc[:, 3:], train_label)
    model = lgb.LGBMClassifier(
        objective='binary',
        random_state=42,
        learning_rate=0.1,
        # n_estimators=1000,
        max_depth=10,
        num_leaves=50,
        bagging_fraction=0.6,
        feature_fraction=0.9,
        metric='auc',
    )

    params = {
        'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
        'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0]
    }
    gsearch = GridSearchCV(model, params, cv=5, scoring='roc_auc', verbose=3, n_jobs=1)
    gsearch.fit(train_data.iloc[:, 3:], train_label, verbose=200)
    print(gsearch.best_params_, gsearch.best_score_)
    return gsearch

result = param_selection(train_data, train_label)


# %%

def train_model(train_data: pd.DataFrame, train_label: np.ndarray, test_data: pd.DataFrame, test_label: np.ndarray):
    kfold = StratifiedKFold(n_splits=5)

    models = []
    test_prob = 0
    for fold_id, (train_idx, val_idx) in enumerate(
        kfold.split(train_data, train_label)
    ):
        x_train = train_data.iloc[train_idx, 3:]
        y_train = train_label[train_idx]
        x_val = train_data.iloc[val_idx, 3:]
        y_val = train_label[val_idx]

        logging.info(f'Fold {fold_id} training...' + '=' * 50)

        model = lgb.LGBMClassifier(
            num_leaves=64,
            max_depth=10,
            learning_rate=0.1,
            n_estimators=1000,
            subsample=0.8,
            feature_fraction=0.6,
            reg_alpha=0.01,
            reg_lambda=0.01,
        )
        model.fit(
            x_train, y_train,
            eval_names=['train',' valid'],
            eval_set=[(x_train, y_train), (x_val, y_val)],
            verbose=50,
            eval_metric='auc',
            early_stopping_rounds=100
        )
        models.append(model)

        if test_data is not None:
            pred_test = model.predict_proba(test_data.iloc[:, 3:], num_iteration=model.best_iteration_)[:, 1]
            test_prob += pred_test / 5
    
    return models, test_prob

def get_val_score(val_label, val_pred):
    host_pred = pd.DataFrame(val_pred, index=val_data['sip']).groupby('sip').mean()
    host_label = label_host(host_pred.index)
    fpr, tpr, threshold = roc_curve(host_label, host_pred)
    auc = roc_auc_score(host_label, host_pred)
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr)
    plt.title(f'{auc=:.3f}')

    plt.subplot(1, 2, 2)
    plt.plot(threshold, tpr - fpr)
    plt.show()

    HOST_THRES = threshold[np.argmax(tpr - fpr)]
    print(HOST_THRES, np.max(tpr - fpr))

models, val_pred = train_model(train_data, train_label, val_data, val_label)
if val_data is not None:
    get_val_score(val_label, val_pred)

#%%
#! Flow threshold
fpr, tpr, threshold = roc_curve(val_label, val_pred)
auc = roc_auc_score(val_label, val_pred)
plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr)
plt.title(f'{auc=:.3f}')

plt.subplot(1, 2, 2)
plt.plot(threshold, tpr - 1.5 * fpr)
plt.show()

THRES = threshold[np.argmax(tpr - 1.5 * fpr)]
print(THRES)

#%%
#! Host threshold
host_pred = pd.DataFrame(val_pred, index=val_data['sip']).groupby('sip').agg(lambda x: (x > 0.528).sum() / x.shape[0])
host_label = label_host(host_pred.index)
fpr, tpr, threshold = roc_curve(host_label, host_pred)
auc = roc_auc_score(host_label, host_pred)
plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr)
plt.title(f'{auc=:.3f}')

plt.subplot(1, 2, 2)
plt.plot(threshold, tpr - fpr)
plt.show()

HOST_THRES = threshold[np.argmax(tpr - fpr)]
print(HOST_THRES)

#%%
def predict_test(models, test_data, threshold):
    pred_score = 0
    for model in models:
        pred_test = model.predict_proba(test_data.iloc[:, 3:], num_iteration=model.best_iteration_)[:, 1]
        pred_score += pred_test / 5
    return (pred_score > threshold).astype(float)

test_pred = predict_test(models, test_data, THRES)

pred = pd.DataFrame(test_pred, index=test_data['sip'])

#%%
#!  ???????????????????????????????????????????????????
"""
th      score
0.5     60
0.4     65.2
0.2     64.6

????????????????????????????????????
"""
th = 0.4
test_host_pred = pred.groupby('sip').mean()

test_black = test_host_pred.index[test_host_pred[0] > th]
print(test_black.shape)
# with open('../result_lgb.txt', 'w') as f:
#     f.write(' '.join(test_black))


#%%
#!  ?????????????????????????????????????????????
"""
th      score
0.5     60
0.4     65.2
0.3     64.9
0.2     64.6


full data
th      score
0.4     61
0.2     65
"""
th = 0.2
test_host_pred = pred.groupby('sip').agg(lambda x: (x > 0.5).sum() / x.shape[0])

test_black = test_host_pred.index[test_host_pred[0] > th]
print(test_black.shape)
with open('../result_lgb.txt', 'w') as f:
    f.write(' '.join(test_black))
