import os, sys, random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Application.Model.Sql.SqlSelection import SqlSelection
from Application.Model.Config.Config import Config
import time
import collections
import nltk
from nltk.metrics import scores
import seaborn as sns
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk import word_tokenize
import itertools
from datetime import date
import numpy as np
import warnings
from inspect import signature
import datetime
from sklearn.metrics import mean_squared_error
import time
from sklearn.model_selection import StratifiedKFold

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

time_start = time.time()
today = date.today()
todayImage = today.strftime("%d_%m_%Y")
todayFile = time.strftime("%Y_%m_%d_%H_%M_%S")
configuration = Config()
sqlI = SqlSelection()

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.naive_bayes import MultinomialNB

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from pathlib import Path

CUDA_LAUNCH_BLOCKING = 1
# TensorFlow
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelWithLMHead
from transformers import BertTokenizer, BertModel, TFBertModel

from bert_score import score

output_dir = configuration.get_property('PYTHON_DIR') + "/gpt_models/bert-base-multilingual-cased/"
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = AutoModelWithLMHead.from_pretrained(output_dir)

results = sqlI.select_all_rows_from_table(['sql columns'],
                                          'table',
                                          'database', None, 'primaryKey ASC')

stop = stopwords.words('romanian')
if os.path.isabs(configuration.get_property('CONFIG_PATH') + os.sep + 'stop_words.txt'):
    with open(configuration.get_property('CONFIG_PATH') + os.sep + 'stop_words.txt', 'r') as opened_file:
        lines = opened_file.read().split("\n")
        for line in lines:
            if configuration.clean_text(line) not in stop:
                stop.append(configuration.clean_text(line))
        opened_file.close()

for id_single, text, new_text in results:
    try:
        P, R, F1 = score([text], [new_text], model_type=output_dir, num_layers=9, lang="ro", verbose=True)
        final = F1.mean()
        print('===================')
        print(f'BERT score: {final}')
        value = final.item()
        print('===================')
        sqlI.update_field_for_all_rows_from_table(['sql columns'], value,
                                                  'table',
                                                  'primaryKey="' + str(id_single) + '" AND primaryKey > 0',
                                                  'database')
    except Exception:
        print(text)
