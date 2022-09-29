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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

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

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from inspect import signature
import datetime
import time
from pathlib import Path

time_start = time.time()
today = date.today()
todayImage = today.strftime("%d_%m_%Y")
todayFile = time.strftime("%Y_%m_%d_%H_%M_%S")
configuration = Config()
sqlI = SqlSelection()


class BPE_token(object):
    def __init__(self):
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.normalizer = Sequence([
            NFKC()
        ])
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()

    def bpe_train(self, paths):
        trainer = BpeTrainer(vocab_size=60000, show_progress=True, inital_alphabet=ByteLevel.alphabet(),
                             special_tokens=[
                                 "<s>",
                                 "<pad>",
                                 "</s>",
                                 "<unk>",
                                 "<mask>"
                             ])
        self.tokenizer.train(paths, trainer)

    def save_tokenizer(self, location, prefix=None):
        if not os.path.exists(location):
            os.makedirs(location)
        self.tokenizer.model.save(location, prefix)


sqlI.select_outfile_all_rows_from_table_to_txt(
    ['columns'],
    'training table',
    'primaryKey > 0', 'database', None, None,
    'text_generator_description_' + str(todayFile))

paths = [str(x) for x in
         Path(configuration.get_property('PYTHON_DIR')).glob("text_generator_description_" + str(todayFile) + ".txt")]
tokenizer = BPE_token()
# train the tokenizer model
tokenizer.bpe_train(paths)
# saving the tokenized data in our specified folder
save_path = 'tokenized_data'
tokenizer.save_tokenizer(save_path)

from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained(save_path)
tokenizer.add_special_tokens({
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>"
})
# creating the configurations from which the model can be made
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id
)
# creating the model
model = TFGPT2LMHeadModel(config)

single_string = ''
for filename in paths:
    with open(filename, "r", encoding='utf-8') as f:
        x = f.read()
        single_string += x + tokenizer.eos_token

string_tokenized = tokenizer.encode(single_string)

examples = []
block_size = 520
BATCH_SIZE = 12
BUFFER_SIZE = 1000

for i in range(0, len(string_tokenized) - block_size + 1, block_size):
    examples.append(string_tokenized[i:i + block_size])
inputs, labels = [], []
for ex in examples:
    inputs.append(ex[:-1])
    labels.append(ex[1:])

dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# defining our optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
# definining our loss function
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# defining our metric which we want to observe
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
# compiling the model
model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])

num_epoch = 15

history = model.fit(dataset, epochs=num_epoch)
from transformers import WEIGHTS_NAME, CONFIG_NAME

output_dir = configuration.get_property('PYTHON_DIR') + "/gpt_models/mcb_model"

# creating directory if it is not present
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

model_to_save = model.module if hasattr(model, 'module') else model
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)
# save model and model configs
model.save_pretrained(output_dir)
model_to_save.config.to_json_file(output_config_file)
# save tokenizer
tokenizer.save_pretrained(output_dir)
time_end = time.time()
print('Generating model! Time elapsed: {} seconds'.format(time_end - time_start))
