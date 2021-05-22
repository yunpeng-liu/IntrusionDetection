#%%
import numpy as np
import pandas as pd
import os, json
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats
from pprint import pprint
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import pickle

plt.style.use(['science', 'grid'])
plt.rc('figure', dpi=200)

TRAIN_RAW_PATH = '../processed_data/train_data'
TEST_RAW_PATH = '../processed_data/test_data'

# %%
def get_tcp_flow_paths(root='../processed_data/train_data'):
    assert os.path.exists(root)
    results = []
    for root, dirs, paths in os.walk(root):
        for path in paths:
            path = os.path.join(root, path)
            if path.endswith('.json') and not path.endswith('info.json') and not path.endswith('.tls.json'):
                results.append(path)
    return results


def get_flow_metadata(path: str):
    assert path.endswith('.json')
    src, dst, sport = path.split(os.sep)[-3:]
    sport = sport[:sport.find('.')]
    return src, dst, sport


def _get_seq_stats(seq: np.ndarray, name: str):
    seq = np.array(seq)
    if len(seq) == 0:
        seq = np.array([0])
    return {
        f'{name}_max': seq.max(),
        f'{name}_min': seq.min(),
        f'{name}_std': seq.std(),
        f'{name}_mean': seq.mean(),
        f'{name}_median': np.median(seq),
        f'{name}_skew': stats.skew(seq),
        f'{name}_kurt': stats.kurtosis(seq)
    }


def _get_len_dist(seq: np.ndarray, name: str):
    if len(seq) == 0:
        seq = [0]
    seq = np.array(seq)
    dist = np.concatenate([
        np.histogram(seq, bins=25, range=(0, 200))[0],
        np.histogram(seq, bins=10, range=(200, 1300))[0],
        np.histogram(seq, bins=25, range=(1300, 1500))[0]
    ])
    dist = dist / dist.sum()
    return dict(list([f'{name}_{i}', j] for i, j in enumerate(dist)))


def extract_flow_info(path: str):
    result = dict()
    eps = 1e-5

    with open(path, 'r') as f:
        data = json.load(f)

        metadata = get_flow_metadata(path)
        result.update(dict(zip(['sip', 'dip', 'sport'], metadata)))

        length = np.array(data['lenth'])
        fwd_pkt_idx = (length >= 0)
        bwd_pkt_idx = (length < 0)
        fwd_pkt_len = length[fwd_pkt_idx]
        bwd_pkt_len = -length[bwd_pkt_idx]
        result.update(_get_seq_stats(abs(length), 'pkt_len'))
        result.update(_get_seq_stats(fwd_pkt_len, 'pkt_len_fwd'))
        result.update(_get_seq_stats(bwd_pkt_len, 'pkt_len_bwd'))
        result['pkt_len_tot'] = sum(abs(length))
        result['pkt_len_tot_fwd'] = sum(fwd_pkt_len)
        result['pkt_len_tot_bwd'] = sum(bwd_pkt_len)
        result['pkt_len_tot_ratio'] = (result['pkt_len_tot_fwd'] + eps) / (result['pkt_len_tot_bwd'] + eps)

        result['pkt_num'] = len(length)
        result['pkt_num_fwd'] = len(fwd_pkt_len)
        result['pkt_num_bwd'] = len(bwd_pkt_len)
        result['pkt_num_ratio'] = (result['pkt_num_fwd'] + eps) / (result['pkt_num_bwd'] + eps)

        result.update(_get_len_dist(fwd_pkt_len, name='dist_pkt_len_fwd'))
        result.update(_get_len_dist(bwd_pkt_len, name='dist_pkt_len_bwd'))

        time = np.array(data['time'])
        inteval = time[1:] - time[:-1]
        fwd_time = time[fwd_pkt_idx]
        fwd_inteval = fwd_time[1:] - fwd_time[:-1]
        bwd_time = time[bwd_pkt_idx]
        bwd_inteval = bwd_time[1:] - bwd_time[:-1]
        result.update(_get_seq_stats(inteval, 'pkt_int'))
        result.update(_get_seq_stats(fwd_inteval, 'pkt_int_fwd'))
        result.update(_get_seq_stats(bwd_inteval, 'pkt_int_bwd'))
        result['tot_time'] = time[-1] - time[0] if len(inteval) > 0 else 0
        result['fwd_time'] = fwd_time[-1] - fwd_time[0] if len(fwd_time) > 0 else 0
        result['bwd_time'] = bwd_time[-1] - bwd_time[0] if len(bwd_time) > 0 else 0

    tls_info_path = path.replace('.json', '.tls.json')
    assert os.path.exists(tls_info_path)
    with open(tls_info_path, 'r') as f:
        records = json.load(f)['records']
        # update default TLS feature in case that some feature do not exist
        result.update({
            'is_tls': float(len(records) > 0),
            'tls_has_server_hello': False,
            'tls_num_ciphers': 1,
            'tls_ciphers': [],
            'tls_cipher_chosen': 0,
            'tls_num_client_ext': 0,
            'tls_num_server_ext': 0,
            'tls_client_ext': [],
            'tls_server_ext': [],
            'tls_client_version': 0,
            'tls_server_version': 0,
        })
        for record in records:
            msg = record.get('msg', [{}])[0]
            if 'ClientHello' in msg.get('class', ''):
                result['tls_num_ciphers'] = len(msg.get('ciphers', []))
                result['tls_ciphers'] = msg.get('ciphers', [])
                result['tls_client_version'] = msg.get('version', 0)
                result['tls_client_ext'] = list(set(i.get('class', '') for i in msg.get('ext', [])))
                result['tls_num_client_ext'] = len(result['tls_client_ext'])
            elif 'ServerHello' in msg.get('class', ''):
                result['tls_has_server_hello'] = 1
                result['tls_cipher_chosen'] = msg.get('cipher', 0)
                result['tls_server_ext'] = list(set(i.get('class', '') for i in msg.get('ext', [])))
                result['tls_num_server_ext'] = len(result['tls_server_ext'])
                result['tls_server_version'] = msg.get('version', 0)

    return result


def get_flow_dataframe(root='../processed_data/train_data'):
    paths = get_tcp_flow_paths(root)
    results = []
    pool = Pool(processes=50)
    for i in tqdm(pool.imap_unordered(extract_flow_info, paths), total=len(paths)):
        results.append(i)
    pool.close()

    keys = results[0].keys()
    data = defaultdict(list)
    for r in results:
        for k in keys:
            data[k].append(r[k])
    
    # need_one_hot = [
    #     'tls_ciphers',
    #     # 'tls_cipher_chosen',
    #     'tls_client_ext',
    #     'tls_server_ext',
    #     'tls_client_version',
    #     'tls_server_version',
    # ]

    # label_encoders = defaultdict(MultiLabelBinarizer)
    # for key_to_encode in need_one_hot:
    #     data_before = data[key_to_encode]
    #     if isinstance(data_before[0], list):
    #         data_before = [[i] for i in data_before]
    #     encoded = label_encoders[key_to_encode].fit_transform(data[key_to_encode])
    #     del data[key_to_encode]
    #     class_num = len(encoded[0])
    #     new_keys = [f'{key_to_encode}_{i}' for i in class_num]
    #     data.update(dict(zip(new_keys, zip(*encoded))))
    
    # temp = data['tls_cipher_chosen']
    return pd.DataFrame(data)


#%%
train_df = get_flow_dataframe(TRAIN_RAW_PATH)
test_df = get_flow_dataframe(TEST_RAW_PATH)

#%%
need_one_hot = [
    'tls_ciphers',
    'tls_cipher_chosen',
    'tls_client_ext',
    'tls_server_ext',
    'tls_client_version',
    'tls_server_version',
]

encoded_features = []

label_encoders = defaultdict(MultiLabelBinarizer)
for key_to_encode in need_one_hot:
    temp = []
    for data in [train_df, test_df]:
        data_before = data[key_to_encode]
        if not isinstance(data_before[0], list):
            data_before = data_before.apply(lambda i: [i])
        temp.append(data_before)
    temp = pd.concat(temp, axis=0)
    label_encoders[key_to_encode].fit(temp)
    for data in [train_df, test_df]:
        data_before = data[key_to_encode]
        if not isinstance(data_before[0], list):
            data_before = data_before.apply(lambda i: [i])
        encoded = label_encoders[key_to_encode].transform(data_before)
        class_num = len(encoded[0])
        new_keys = [f'{key_to_encode}_{i}' for i in range(class_num)]
        new_feature = pd.DataFrame(encoded, columns=new_keys)
        encoded_features.append(new_feature)
    
#%%
train_df.drop(need_one_hot, axis=1, inplace=True)
test_df.drop(need_one_hot, axis=1, inplace=True)
train_df = pd.concat([train_df] + encoded_features[0::2], axis=1)
test_df = pd.concat([test_df] + encoded_features[1::2], axis=1)

# %%
train_df.to_csv('../features/flow_feature_train.csv')
test_df.to_csv('../features/flow_feature_test.csv')