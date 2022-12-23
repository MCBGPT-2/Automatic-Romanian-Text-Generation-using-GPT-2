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
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import numpy as np
import warnings
from inspect import signature
import datetime
import time
from google.colab import files

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

output_dir = configuration.get_property('PYTHON_DIR') + "/gpt_models/mcb_model/"
CUDA_LAUNCH_BLOCKING = 1
# TensorFlow
from transformers import AutoTokenizer, TFAutoModelForCausalLM
from sacrebleu import sentence_bleu

results = sqlI.select_all_rows_from_table(['columns'], 'table', 'database', None, 'primaryKey DESC')
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = TFAutoModelForCausalLM.from_pretrained(output_dir)

updated_texts = dict()
for result in results:
    updated_texts.update({result[0]: configuration.clean_text(result[1])})

for id_single, text in updated_texts.items():
    max_length = 4000
    min_length = 600
    print('===================')
    inputs = tokenizer.encode(text, return_tensors='tf', max_length=512, truncation=True)
    text_predicted = model.generate(inputs, max_length=int(max_length), min_length=int(min_length),
                                    no_repeat_ngram_size=2,
                                    temperature=0.8, num_beams=5, num_return_sequences=5, )
    new_text = tokenizer.decode(text_predicted[0], skip_special_tokens=True).replace('<|endoftext|>', '')
    new_text = new_text.replace("'", ' ')
    sqlI.update_field_for_all_rows_from_table(['column'], "'" + new_text + "'",
                                              'table',
                                              'primaryKey="' + str(id_single) + '" AND primaryKey > 0',
                                              'database')
