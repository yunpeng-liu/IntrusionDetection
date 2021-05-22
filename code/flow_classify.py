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

