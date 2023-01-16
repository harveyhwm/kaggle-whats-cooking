import numpy as np
import pandas as pd
import os
import re
import time
import shutil
import json
import random
import copy
import zipfile
import pickle

import datetime
from datetime import datetime as dt

import tensorflow as tf
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model,save_model,load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import TextVectorization

from itertools import product

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter,defaultdict

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import catboost
from catboost import CatBoostClassifier as cbc,CatBoostRegressor as cbr

AUTOTUNE = tf.data.AUTOTUNE


from helpers import *

sns.set_theme()

VAL_FRAC = 0.1
if 'my_auc' not in globals():
    my_auc = AUC()  # store the AUC method, being sure to only do so once
MODEL_PATH = 'models/'
MODEL_PATH_TEST = 'models_test/'
HISTORY_PATH = 'model_data/model_history.csv'
HISTORY_PATH_TEST = 'model_data/model_history_test.csv'
SEED = 89  # this must remain the same for dataset generation consistency, as pandas uses the numpy seed to dictate its own randomness
VOCAB_SIZE = 10000


# initialize a global store for model histories, and re-fetch its contents whenever required 
def model_history(path=HISTORY_PATH):
    return pd.read_csv(path)

# self-ingestion to get out our python code when regular export fails
def get_raw_python_from_notebook(notebook,python=None):
    if python is None: python=notebook
    with open(notebook+'.ipynb','r') as f:
        rawpy = json.load(f)
    rawpy = [[] if c['source'] == [] else c['source'] for c in rawpy['cells'] if c['cell_type']=='code']
    for r in rawpy:
        r.extend(['\n','\n'])
    raw = [l for r in rawpy for l in r]
    with open(python+'.py', 'w') as f:
        f.write(''.join(raw))
get_raw_python_from_notebook('kaggle-whats-cooking')

def text_clean(c, stopwords = None, flatten_case = True):
    c = re.sub('\s+|\n',' ',c)
    c = re.sub(' \.|"','',c)
    if flatten_case is True:
        c = c.lower()
    if stopwords is not None:
        c = ' '.join([s for s in c.split() if s not in stopwords and len(s) < 50])
    return c.strip()

def make_confusion_matrix(data, *args, percent='precision', preds='pred', real='real', counts=None, groups=None, raw=False):
    if counts is None:
        data['accuracy'] = np.where(data['pred']==data['real'],1,0)
        counts = 'accuracy'
    columns = None if 'columns' not in args else columns['args']
    idx = preds if percent=='recall' else real
    col = real if percent=='recall' else preds
    data = data.groupby([preds,real]).count().reset_index().pivot(index=idx, columns=col, values=counts).fillna(0).reset_index(drop=True)
    if groups is not None: data.columns = groups
    for c in data.columns:
        try:
            data[c] = np.round(100*data[c]/np.sum(data[c]),5)
        except:
            pass
    pretty_data = pretty_pandas(data,configs=make_palette(5,98,palette=['#F7F7FE','#FE0','red'],number='pct',columns=columns))
    return data if raw is True else pretty_data

# legacy method for tokenizing using tf.keras.preprocessing.text.Tokenizer

def text_prepare_legacy(data, stopwords = None, num_words = 10000, oov_token = '<OOV>', max_len = 'max', padding = 'post',
                        val_frac=0.1, chunk_size = 1, encode = None):
    data.columns = ['text','id']+list(data.columns)[2:]
    data['len'] = data['text'].apply(lambda x: len(x.split()))

    max_len = np.max(data['len']) if max_len == 'max' else np.minimum(np.max(data['len']),max_len)

    train = data.drop_duplicates(subset=['text']).reset_index(drop=True)

    label_list = []
    encoder,decoder = {},{}
    for l in list(data.columns)[1:-1]: #-1]:
        if encode == 'ordinal':
            label_list.append(l)
            train[l+'_original'] = train[l]
            #return train[l]
            le = LabelEncoder().fit(train[l])
            for t in list(set(train[l])):
                encoder[t] = le.transform([t])[0]
                decoder[le.transform([t])[0]] = t
            train[l] = le.transform(train[l])
            #return train[l]
        elif encode == 'onehot':
            label_options = [re.sub(' ','_',i) for i in set(train[l])]
            for s in label_options:
                train[l+'_'+s] = [1 if j==s else 0 for j in train[l]]
                label_list.append(l+'_'+s)

    # return list(data.columns)[2:],label_list

    ids = list(train[['id']].drop_duplicates()['id'])
    ids_val = ids[:int(len(ids)*val_frac)]

    val = train[train['id'].isin(ids_val)].reset_index(drop = True)
    train = train[~train['id'].isin(ids_val)].reset_index(drop = True)

    tokenizer = Tokenizer(num_words = num_words,oov_token = oov_token)
    tokenizer.fit_on_texts(train['text'])

    train_sequences = tokenizer.texts_to_sequences(train['text'])
    val_sequences = tokenizer.texts_to_sequences(val['text'])

    train_padded = pad_sequences(train_sequences, maxlen = max_len, padding = padding)
    val_padded = pad_sequences(val_sequences, maxlen = max_len, padding = padding)

    train_labels,val_labels = {},{}
    for l in label_list:
        train_labels[l] = np.array(train[l])
        val_labels[l] = np.array(val[l])

    word_index = tokenizer.word_index

    # here we return a set of assets relevant to the data and its tokenization, for use in modeling later
    return {'train_padded': train_padded,
            'train_labels': train_labels,
            'val_padded': val_padded,
            'val_labels': val_labels,
            'encoder': encoder,
            'decoder': decoder,
            'max_len': max_len,
            'word_index': word_index,
            'vocab': num_words,
            'train': train,
            'val': val}

# new version - using the TextVectorizer layer method

def text_prepare(data, stopwords = None, num_words = 10000, oov_token = '<OOV>', max_len = 'max', batch_size=None,
                 label = 'last', padding = 'post', val_frac=0.1, chunk_size = 1, encode = None):
    
    cols = list(data.columns)
    if label is 'last':
        label = cols[-1]
    else:
        cols.append(cols.pop(cols.index(label)))
        data = data[cols]
    data['len'] = data[cols[0]].apply(lambda x: len(x.split()))
    
    max_len = np.max(data['len']) if max_len == 'max' else np.minimum(np.max(data['len']),max_len)

    train = data.drop_duplicates(subset=[cols[0]]).reset_index(drop=True)    
    
    unique_labels = np.unique(np.array(data[label]))
    label_encoding, label_decoding = {},{}
    for i,c in enumerate(unique_labels):
        label_encoding[c] = i
        label_decoding[i] = c
    if encode in ('ordinal', 'onehot'):
        labels = np.asarray(train[label].apply(lambda x: label_encoding[x]))
    if encode == 'onehot':
        labels = to_categorical(labels)
        
    split = int(val_frac*len(train))
    
    val_x,val_y = np.asarray(train[cols[0]][:split]),labels[:split]
    train_x,train_y = np.asarray(train[cols[0]][split:]),labels[split:]
    
    vectorize_layer = TextVectorization(
        # standardize = custom_standardization,
        max_tokens = num_words,
        output_mode = 'int',
        output_sequence_length = int(max_len)
    )
    
    text_ds = tf.data.Dataset.from_tensor_slices(np.asarray(train_x)).batch(64)
    vectorize_layer.adapt(text_ds)

    tokenizing_model = tf.keras.models.Sequential()
    tokenizing_model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    tokenizing_model.add(vectorize_layer)
    
    train_x = tokenizing_model.predict(train_x,verbose=0)
    val_x = tokenizing_model.predict(val_x,verbose=0)
    
    if encode == 'onehot':
        train = [tf.data.Dataset.from_tensor_slices((train_x,train_y[:,i])) for i in range(len(unique_labels))]
        val = [tf.data.Dataset.from_tensor_slices((val_x,val_y[:,i])) for i in range(len(unique_labels))]
    else:
        train = [tf.data.Dataset.from_tensor_slices((train_x,train_y))]
        val = [tf.data.Dataset.from_tensor_slices((val_x,val_y))]
    
    if batch_size is not None: # batching is useful for speed when we predict (indpendent of batches for training)
        train = [t.batch(batch_size) for t in train]
        val = [v.batch(batch_size) for v in val]
        
    vocab = vectorize_layer.get_vocabulary()
    word_encoding, word_decoding = {}, {}
    for i, w in enumerate(vocab):
        word_encoding[w] = i
        word_decoding[i] = w
        
    # we now return a set of assets relevant to the data and its tokenization, for use in modeling later
    return {'vectorizer': vectorize_layer,
            'train_data': train,
            'train_labels': train_y,
            'val_data': val,
            'val_labels': val_y,
            'word_encoding': word_encoding,
            'word_decoding': word_decoding,
            'label_encoding': label_encoding,
            'label_decoding': label_decoding,
            'max_len': max_len,
            'vocab_size': num_words,
            'input_shape': train_x[0].shape,
            'unique_labels': unique_labels}

# this function creates and maintains a centralized DataFrame of model history, and returns the relevant values for analysis and plotting etc

def make_history(history, model_name, lr, optimizer, batch_size, data_source, loss_name, timestamp, label, epoch, epoch_subset = 1, write = True):
    h = pd.DataFrame.from_dict(history).reset_index()
    h.columns = [re.sub('_[0-9]{1}.*','',c) for c in h.columns]
    h['timestamp'] = timestamp
    h['label'] = label
    if len(h)>1:
        h['epoch'] = h['index']+1
    else:
        h['epoch'] = epoch
    h['data'] = data_source
    h['lr_start'] = lr[0]
    h['lr_end'] = lr[1]
    h['optimizer'] = optimizer
    h['loss_name'] = loss_name
    h['batch_size'] = batch_size
    h['epoch_subset'] = epoch_subset
    h['name'] = model_name
    del h['index']
    return h

    try:
        h = pd.read_csv('model_data/'+name+'.csv').append(h).reset_index(drop=True)
    except:
        pass
    if write is True:
        h.to_csv('model_data/'+name+'.csv',index=False)
    cols = ['data','name','label','loss_name','optimizer','lr_start','lr_end','timestamp','loss',
          'accuracy','val_loss','val_accuracy','lr','epoch','batch_size','epoch_subset']
    if 'val_auc' in h.columns:
        cols+=['auc','val_auc']
    h = h[cols]
    return h

def plot_history_v1(hist,name):
    hist = model_history()
    hist = hist[hist['name'].str.contains(name)].reset_index(drop=True)

    (d1,d2) = (1,2)
    fig,axes = plt.subplots(d1, d2, figsize=(20,6))
    #fig.tight_layout()
    for p in set(hist['label']):
        data = hist[hist['label']==p]
        for i in range(d2): s = sns.lineplot(ax=axes[i].twinx(), x = data['epoch'], y = data['lr'], color='#CCE', linestyle='--')
        s = sns.lineplot(ax=axes[0], x = data['epoch'], y = data['accuracy'])
        s = sns.lineplot(ax=axes[0], x = data['epoch'], y = data['val_accuracy'])
        s = sns.lineplot(ax=axes[1], x = data['epoch'], y = data['loss'])
        s = sns.lineplot(ax=axes[1], x = data['epoch'], y = data['val_loss'])

def plot_history(hist, filters=None, group='name', metrics=['accuracy','loss'], width=2, height=1, tight=False, size=(20,6), val=None, show_lr=True):
    if filters is not None:
        hist = hist[hist['name'].str.contains(filters)].reset_index(drop=True)
    fig,axes = plt.subplots(height, width, figsize=size)
    if tight is True: fig.tight_layout()    
    for l,p in enumerate(set(hist[group])):
        data = hist[hist[group]==p]
        for i,j in enumerate(product(range(height),range(width))):
            if i==len(metrics): break
            idx = i%d2 if val is None else i
            if height > 1:
                s = sns.lineplot(ax=axes[0,j[1]], x = data['epoch'], y = data[metrics[idx]])
                s = sns.lineplot(ax=axes[1,j[1]], x = data['epoch'], y = data['val_'+metrics[idx]])
                if show_lr is True and l==0:
                    s = sns.lineplot(ax=axes[0,j[1]].twinx(), x = data['epoch'], y = data['lr'], color='#CCE', linestyle='--')
                    s = sns.lineplot(ax=axes[1,j[1]].twinx(), x = data['epoch'], y = data['lr'], color='#CCE', linestyle='--') # i%height,i%width
            else:
                s = sns.lineplot(ax=axes[j[1]], x = data['epoch'], y = data[metrics[idx]])
                if val=='overlap': s = sns.lineplot(ax=axes[j[1]], x = data['epoch'], y = data['val_'+metrics[idx]])
                if show_lr is True and l==0: s = sns.lineplot(ax=axes[j[1]].twinx(), x = data['epoch'], y = data['lr'], color='#CCE', linestyle='--')

# model wrapper for catboost classification and regression
def cbc_wrapper(data, target='y', test_frac=0.1, loss_function='MultiClass', verbose=1, eval_metric='Accuracy',
                rounds=1, epochs=200, depth=5, learning_rate=0.01, use_best_model=True):

    types_df = pd.DataFrame(data.dtypes).reset_index().reset_index()
    types_char = types_df[types_df[0]=='object']
    types_float = types_df[types_df[0]=='float64']
    types_int = types_df[types_df[0]=='int64']
    ordcols = list(types_char['index'])+list(types_float['index'])+list(types_int['index'])
    ordindices = list(types_char['level_0'])
    data = data[ordcols]
    data[list(types_char['index'])] = data[list(types_char['index'])].astype(str)

    my_features = data.filter(regex='^(?!'+target+')')
    feature_names = list(my_features.columns.values)

    models, model_paths = [], []
    for i in range(rounds):
        data_shuffled = data.loc[:,:].copy().sample(frac=1)
        x_val = data_shuffled.iloc[:int(len(data_shuffled)*test_frac),:]
        y_val = x_val.pop(target)
        x_train = data_shuffled.iloc[int(len(data_shuffled)*test_frac):,:]
        y_train = x_train.pop(target)
        cat_features=list(range(0,len(ordindices)-0))

        model = cbc(loss_function=loss_function,verbose=verbose,eval_metric=eval_metric,metric_period=1,
                          iterations=epochs,depth=depth,learning_rate=learning_rate,od_wait=20)
        model.fit(x_train,y_train,eval_set=(x_val,y_val),cat_features=cat_features,metric_period=1)
        y_pred = model.predict(x_val,verbose=verbose)
        
        res = model.evals_result_
        for m,k in enumerate(res.keys()):
            for j,l in enumerate(res[k].keys()):
                if m==0 and j==0:
                    history = pd.DataFrame(res[k][l],columns=[(k+'_'+l).lower()]).reset_index()
                else:
                    history = pd.concat([history,pd.DataFrame(res[k][l],columns=[(k+'_'+l).lower()]).reset_index(drop=True)],axis=1)
        history.columns = ['epoch']+list(history.columns)[1:]
        history['depth'] = depth
        history['learning_rate'] = learning_rate
        history['max_epochs'] = epochs
        history['rounds'] = rounds
        history['round'] = i+1
        if type == 'regressor':
            r2 = r2_score(np.array(y_val),y_pred)
            history['r2'] = np.round(r2,6)
        history_all = history if i==0 else history_all.append(history) 

        models.append(model)
    
    if rounds==1: models = models[-1]
    params = {'x_train': x_train,
              'y_train': y_train,
              'x_val': x_val,
              'y_val': y_val,
              'cat_features': cat_features,
              'y_pred': y_pred}
        
    return models,history_all,params

# model wrapper for fully connected model architectures 

def embedding_model(input_length, embedding_dim=16, vocab_size=VOCAB_SIZE, categories=1, lr=1e-3, dense_layers=[256], metrics=['accuracy']):

    if categories >= 3:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        output_activation='softmax'
    else:
        loss = tf.keras.losses.BinaryCrossentropy()
        output_activation='sigmoid'
    
    input_layer = tf.keras.Input(shape=input_length)
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=input_length)(input_layer)
    x = tf.keras.layers.Flatten()(x)
    for l in dense_layers:
        x = tf.keras.layers.Dense(l, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(categories, activation=output_activation)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # setup the training parameters
    model.compile(loss=loss,
                  optimizer=tf.keras.optimizers.Adam(lr=lr),
                  metrics=metrics)

    return model

# model wrapper for LSTM returrant NN architectures - these are works in progress

def lstm_model(data, loss = 'binary_crossentropy', optimizer = 'adam', rnn_depth = 64, embedding_dim = 16, layer_norm = False,
               categories = 1, activation = 'sigmoid', metrics=['accuracy']):
    input_layer = tf.keras.Input(shape = data['input_shape'])
    x = tf.keras.layers.Embedding(data['vocab_size'], embedding_dim, input_length = data['max_len'])(input_layer)
    # x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_depth,))(x)
    if layer_norm is True: x = tf.keras.layers.LayerNormalization()(x) # experimental!
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(categories, activation)(x)
    model = Model(inputs = input_layer, outputs = output_layer)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    return model

# stateful version of LSTM model wrapper - work in progress - bidirectional stateful to fix

class lstm_model_stateful(tf.keras.Model):
    def __init__(self, data, rnn_units, embedding_dim = 16):
        super().__init__(self)
        self.input_layer = tf.keras.Input(shape = data['input_shape'])
        self.embedding = tf.keras.layers.Embedding(data['vocab_size'], embedding_dim, input_length = data['max_len'])
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_units, return_sequences=False, return_state=True))
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        #if states is None:
        #    states = self.lstm.get_initial_state(x)
        #return self.lstm(x, initial_state=states, training=training)
        
        x, state1, state2, state3, state4 = self.lstm(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, [state1, state2, state3, state4]
        else:
            return x

def model_gen(data): # to be expanded for different model types
    return lstm_model(data, loss = 'sparse_categorical_crossentropy', optimizer = 'adam',
                      rnn_depth = 64, embedding_dim = 16, layer_norm = False, categories = 20,
                      activation = 'softmax', metrics=['accuracy'])

def learning_trajectory(epochs, lr=(1e-5, 1e-3), mode='linear', return_list=False):
    # modes can be linear, exponential, (plateau...?)
    if mode == 'linear':
        lr_list = [LearningRateScheduler(lambda epoch: lr[0] + ((lr[1]-lr[0])*epoch/epochs))]
        if return_list is True:
            return [lr[0] + ((lr[1]-lr[0])*epoch/epochs) for epoch in range(epochs)]
    elif (mode == 'exp') or (mode == 'exponential'):
        lr_list = [LearningRateScheduler(lambda epoch: lr[0] * 10**(np.log10(lr[1]/lr[0])*(epoch)/(epochs-1)))]
        if return_list is True:
            return [lr[0] * 10**(np.log10(lr[1]/lr[0])*(epoch)/(epochs-1)) for epoch in range(epochs)]
    return lr_list

def model_wrapper_legacy(data, label = 'y', model_name = 'mymodel', lr = (5e-5,1e-3), epochs = 1, split_epochs = False, start = 1, end = None, batch_size = 128,
                      epoch_subset = 1, epoch_subset_val = 1, complete_all_subsets = True, history = 'model_history', class_weight = None, rnn_depth = 64,
                      embedding_dim = 16, layer_norm = False, multiclass = False, save_freq = 1, save_format = 'savedmodel', stop = None):

    ## to do: implement early stopping when we split the epochs
    ## to do: make the early stopping dynamic in relation to the learning rate (intuition seen from other projects)
    ## implement regularization parameters
    ## to do: make this into a class-based architecture
    
    if split_epochs is True:
        runs = epochs
        epochs = 1
    else:
        runs = 1 if complete_all_subsets is False else int(1/epoch_subset)
    end = min(runs, start + end) if end is not None else runs

    if multiclass is True:
        loss_name = 'sparse_categorical_crossentropy' # tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = 'adam'
        activation = 'softmax'
        categories = len(set(data['train_labels'][label]))
        monitor_metric = 'val_accuracy'
        metrics = ['accuracy']
    else: 
        loss_name = 'binary_crossentropy' # tf.keras.losses.BinaryCrossentropy()
        optimizer = 'adam'
        activation = 'sigmoid'
        categories = 1
        monitor_metric = 'val_auc'
        metrics = ['accuracy',AUC]

    try:
        model_data = pd.read_csv('model_data/'+history+'.csv')
        model_data = model_data[model_data['name'] == model_name]       
        start = max(model_data['epoch'])+1
        print(start)
        if start > max(runs,epochs): return
        best = max(model_data[monitor_metric])
        models = [load_model('model_data/'+model_name+'_last')]
    except:
        model_data = None
        print('no model found - initializing')
        best = 0.0
        models = [lstm_model(data, loss_name, optimizer, rnn_depth, embedding_dim, layer_norm, categories, activation, metrics)]

    data_source = 'kaggle_whats_cooking'
    r = start
    timestamp = dt.strftime(dt.now(),'%Y-%m-%d %H:%M:%S')
    metric_history = [0]

    # checkpoint = ModelCheckpoint(
    #     'model_data/'+model_name+'_best',
    #     monitor = monitor_metric,
    #     verbose = 0,
    #     mode = 'max',
    #     save_best_only = True
    # )

    stop_early = EarlyStopping(monitor = 'val_loss', patience = stop, restore_best_weights = True)

    if class_weight is not None:
        class_sizes = pd.DataFrame.from_dict([Counter(data['train_labels'][label])]).transpose()
        class_weight = {}
        for l in set(data['train_labels'][label]):
            class_weight[l] = 1/class_sizes[0][l]

    while r <= min(end,runs):
        if (r > 1):
            if (r%save_freq == 1 or save_freq == 1):
                models = [load_model('model_data/'+model_name+'_last')]
                metric_history = [0]
            else:
                models.append(copy.copy(models[-1]))

        lr_scheduler = LearningRateScheduler(
            lambda epoch: lr[0] * 10**(np.log10(lr[1]/lr[0])*(max(r,epoch+1)-1)/max(1,epochs-1,runs-1))
        )

        np.random.seed(np.random.randint(1e6))
        mylen = data['train_padded'].shape[0]
        # subset = np.random.choice(mylen, int(mylen*epoch_subset), replace=False)
        subset = np.arange(int(mylen*epoch_subset*((r-1)%runs)),min(mylen,int(mylen*epoch_subset*(r%runs))))
        x = data['train_padded'][subset]
        y = data['train_labels'][label][subset]
        
        np.random.seed(480)
        mylen_val = data['val_padded'].shape[0]
        subset_val = np.random.choice(mylen_val, int(mylen_val*epoch_subset_val), replace=False)
        x_val = data['val_padded'][subset_val]
        y_val = data['val_labels'][label][subset_val]

        print(models[-1].summary)
        h = models[-1].fit(
            x,y,
            batch_size = batch_size,
            epochs = epochs,
            validation_data = (x_val,y_val), #(data['val_padded'],data['val_labels'][label]),
            callbacks = [lr_scheduler,stop_early],
            class_weight = class_weight
        )

        if np.isclose([h.history[monitor_metric][-1:]],[0.5000]):
            if r == 1:
                models[-1] = lstm_model(data, loss_name, optimizer, rnn_depth, embedding_dim, layer_norm, categories, activation, metrics)
            else:
                models[-1] = load_model('model_data/'+model_name+'_last')
        else:
            metric_history.append(h.history[monitor_metric][-1:][0])
            full_history = make_history(h, model_name, lr, optimizer, batch_size, data_source, loss_name, timestamp,
                                        label, epoch = r, epoch_subset = epoch_subset, name = history)
            
            full_history = np.array(full_history[full_history['name'] == model_name][monitor_metric])
            
            if split_epochs is False:
                save_model(models[-1],'model_data/'+model_name+'_last')
            elif r%save_freq == 0:
                best = max(full_history)
                save_model(models[-1],'model_data/'+model_name+'_last')
                print('last model saved')
                if max(metric_history) >= best:
                    shutil.copytree('model_data/'+model_name+'_last','model_data/'+model_name+'_best',dirs_exist_ok=True)
                    print('best model saved')
            r += 1
            
        # let's catch the early stopping rule here
        if (len(full_history) > stop) and (np.argmax(full_history)<(len(full_history)-stop)):
            return

# this function provides a wrapper to train models either in one go, or incrementally by epoch, tracking progress
# the notion was established to tackle the fact that many models would crash before they finished training as they are very heavy

def process_model(data, model, label = 'y', model_name = 'mymodel', model_selection = 'lstm', lr = (5e-5,1e-3), epochs = 1, split_epochs = False,
                start = 1, end = None, batch_size = 128, epoch_subset = 1, epoch_subset_val = 1, complete_all_subsets = True, model_path = MODEL_PATH,
                history_path = HISTORY_PATH, class_weight = None, optimizer = None, loss_name = None, rnn_depth = 64, embedding_dim = 16,
                layer_norm = False, multiclass = True, label_class = 0, save_format = 'savedmodel', stop_early = True, verbose = False):
    
    # split_epochs allows us save the model after *every* epoch, and resume from the reloaded
    # version if necessary. This failsafes against model crashes when they are heavy
    if split_epochs is True:
        runs = epochs
        epochs = 1
    else:
        runs = 1
    end = min(runs, start + end) if end is not None else runs   
    
    # some parameters to store and monitor results
    data_source = 'kaggle_whats_cooking'
    stop_early = 5 if stop_early is True else stop_early
    timestamp = dt.strftime(dt.now(),'%Y-%m-%d %H:%M:%S')
    monitor_metric = 'val_auc' if multiclass is False else 'val_accuracy'
    lr_mode = lr[-1] if len(lr)>2 else 'exp' # default learning rate mode to exponential if we didn't set one
    
    try: # check if we already have a model history file in general
        model_data_all = pd.read_csv(history_path)
    except:
        model_data_all = None
        
    try: # check if any existing model history file contains history of the current model 
        model_data = model_data_all[model_data_all['name'] == model_name]     
        start = max(model_data['epoch'])+1
        if verbose is True: print(start)
        if start > max(runs,epochs): return # this means we've already completed our epochs and didn't need any further runs
        metric_history = [max(model_data[monitor_metric])] # a record of the model performance for each epoch, if they render separate models
        model = load_model('model_data_test/'+model_name+'_last') # load most recent completed epoch of the model
    except: # if we have not got the model in our history, we start from scratch
        if verbose is True: print('no model found - initializing')
        metric_history = [0]
        # return a compiled model with requested parameters
            # model = model_gen(data) # model, multiclass, rnn_depth, embedding_dim, layer_norm, categories, activation, metrics)]
            # model = model

    r = start # which epoch we start from
    
    # set up model callbacks - we need a set of learning rates as a list to apply to our iterator
    learning_rate_list = learning_trajectory(max(runs,epochs), lr = lr, mode = lr_mode, return_list = True)
    early_stopping = EarlyStopping(monitor = monitor_metric, patience = stop_early, restore_best_weights = True)
    
    # set optional class weights to tackle imbalance if we wish 
    if class_weight is not None:
        class_sizes = pd.DataFrame.from_dict([Counter(data['train_labels'])]).transpose()
        class_weight = {}
        for l in set(data['train_labels']): # [label]
            class_weight[l] = min(class_sizes[0])/class_sizes[0][l]
            
    # batch and optimize our train and validation data
    train_data = data['train_data'][label_class]
    val_data = data['val_data'][label_class]
    
    try:
        train_data = train_data.unbatch()
        val_data = val_data.unbatch()
    except:
        pass
    
    train_data = train_data.cache().batch(batch_size).shuffle(1000).prefetch(buffer_size = AUTOTUNE)
    val_data = val_data.cache().batch(batch_size).shuffle(1000).prefetch(buffer_size = AUTOTUNE)
    
    while r <= min(end, runs):
        # the models are always stored in an array. This just has length 1 if we didnt set split_epochs to true. Else, it accumulates per epoch
        learning_rate = (learning_rate_list[r-1], learning_rate_list[r-1]) if split_epochs is True else lr
        learning_rates = learning_trajectory(max(runs,epochs), lr = learning_rate, mode = 'exp', return_list = False)
        callbacks = [learning_rates]
        if (split_epochs is False) and (stop_early is not False): callbacks.append(early_stopping)
        
        h = model.fit(
            train_data,
            batch_size = batch_size,
            epochs = epochs,
            validation_data = val_data,
            callbacks = callbacks,
            class_weight = class_weight
        ).history
        
        history = make_history(h, model_name, lr, optimizer, batch_size, data_source, loss_name,
                               timestamp, label, epoch = r, epoch_subset = epoch_subset)
        try:
            model_data_all = model_data_all.append(history)
        except:
            model_data_all = history
        model_data_all.to_csv(history_path,index=False)
        metric_history.append(h[monitor_metric][-1:][0]) # add most recent model performance metric to the history to compare
        
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        save_model(model,model_path+model_name+'_last'+('.h5' if save_format=='h5' else ''))
        
        if split_epochs is True:
            if verbose is True: print('last model saved')
            if np.argmax(np.flip(metric_history)) == 0: # if the most recent epoch was the best, the argmax of the flip will always be 0
                # store the last model as the best if appropriate
                shutil.copytree(model_path+model_name+'_last', model_path+model_name+'_best', dirs_exist_ok=True)
                if verbose is True: print('best model saved')
            elif (stop_early is not False) and (np.argmax(np.flip(metric_history)) >= stop_early): # early stopping for the split epoch case
                break
        r += 1

with open('data/train.json','r') as f: train = json.load(f)
with open('data/test.json','r') as f: test = json.load(f)

np.random.seed(SEED)
train_df = pd.DataFrame.from_dict(train).sample(frac=1).reset_index(drop=True)
test_df = pd.DataFrame.from_dict(test)

train_df.head()

cuisine_groups = np.unique(train_df['cuisine'])

groups_df = pd.DataFrame.from_dict([Counter(train_df['cuisine'])]).transpose().reset_index().sort_values(by='index').reset_index(drop=True)
groups_df['share'] = (groups_df[0]/sum(groups_df[0])).apply(lambda x: '{0:.2%}'.format(x))
groups_df

foods = [] # get an exhaustive list of foods contained in recipes
for i in train:
    foods = foods+i['ingredients']

food_counts = Counter(foods) # get the frequencies of each food to aid prioritization

food_counts = pd.DataFrame.from_dict([food_counts]).transpose().reset_index()
food_counts.columns = ['food','freq']
food_counts = food_counts.sort_values(by='freq',ascending=False).reset_index(drop=True)
food_counts['cum_sum'] = np.cumsum(food_counts['freq'])/sum(food_counts['freq'])
print('Total number of foods:',len(food_counts),'\n')
food_counts.head()

for i,j in enumerate(food_counts['food'][:len(food_counts)]):
    if i%200 == 0: print(i)
    train_df[j] = train_df['ingredients'].apply(lambda x: 1 if j in x else 0)
    test_df[j] = test_df['ingredients'].apply(lambda x: 1 if j in x else 0)

train_df.head()

pca = PCA().fit(train_df.iloc[:,3:])

plt.figure(figsize=(10,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

pca = PCA(n_components=250).fit(train_df.iloc[:,3:])

x_train = pca.transform(train_df.iloc[:,3:])
y_train_ordinal = LabelEncoder().fit(train_df['cuisine']).transform(train_df['cuisine'])
y_train_one_hot = OneHotEncoder().fit(np.array(train_df['cuisine']).reshape(-1, 1)).transform(np.array(train_df['cuisine']).reshape(-1, 1)).toarray()

data_ordinal = pd.concat([pd.DataFrame(x_train),pd.DataFrame(y_train_ordinal)],axis=1)

data_ordinal.columns = list(data_ordinal.columns)[:-1]+['y']
data_ordinal.head()

model1,model_history1,model_data1 = cbc_wrapper(data_ordinal,epochs=20000,verbose=1000,eval_metric='Accuracy')
model1.save_model('models/catboost_1.json',format='json')
model_history1.to_csv('models/catboost_history_1.csv',index=False)

model2,model_history2,model_data2 = cbc_wrapper(data_ordinal,epochs=20000,verbose=1000,eval_metric='MultiClass')
model2.save_model('models/catboost_1.json',format='json')
model_history2.to_csv('models/catboost_history_2.csv',index=False)

model_history_all = pd.concat([model_history2.iloc[:,:3],model_history1[['learn_accuracy','validation_accuracy']]],axis=1)
model_history_all.columns = ['epoch','loss','val_loss','accuracy','val_accuracy']

model_history_all

fig,axes = plt.subplots(1, 2, figsize=(20,6))
s = sns.lineplot(ax=axes[0], x = model_history_all['epoch'], y = model_history_all['accuracy'])
s = sns.lineplot(ax=axes[0], x = model_history_all['epoch'], y = model_history_all['val_accuracy'])
s = sns.lineplot(ax=axes[1], x = model_history_all['epoch'], y = model_history_all['loss'])
s = sns.lineplot(ax=axes[1], x = model_history_all['epoch'], y = model_history_all['val_loss'])

preds1 = pd.concat([pd.DataFrame(model_data1['y_pred']).reset_index(drop=True),pd.DataFrame(model_data1['y_val']).reset_index(drop=True)],axis=1)
preds2 = pd.concat([pd.DataFrame(model_data2['y_pred']).reset_index(drop=True),pd.DataFrame(model_data2['y_val']).reset_index(drop=True)],axis=1)

preds1['match'] = preds1.apply(lambda r: 1 if r['y']==r[0] else 0,axis=1)
preds2['match'] = preds2.apply(lambda r: 1 if r['y']==r[0] else 0,axis=1)

print('Accuracy after',len(model_history1),'epochs is:',sum(preds1['match'])/len(preds1))
print('Accuracy after',len(model_history2),'epochs is:',sum(preds2['match'])/len(preds2))

# ADDITIONAL DATA PREPARATION
# concatenate the words into a single string, and clean out any unwanted characters

train_df['recipe_concat'] = train_df['ingredients'].apply(lambda x: ' '.join(x))
train_df['recipe_clean'] = train_df['recipe_concat'].apply(lambda x: text_clean(x))
check_dupes = train_df[['cuisine', 'recipe_concat']].groupby('recipe_concat').count().reset_index()
check_dupes = check_dupes[check_dupes['cuisine']==1]
train_df = train_df[train_df['recipe_concat'].isin(check_dupes['recipe_concat'])].reset_index(drop=True)
train_df.head()

model_data_legacy = text_prepare_legacy(train_df[['recipe_clean','cuisine']], num_words = VOCAB_SIZE, max_len = 30, encode = 'ordinal')
model_data_onehot_legacy = text_prepare_legacy(train_df[['recipe_clean','cuisine']], num_words = VOCAB_SIZE, max_len = 30, encode = 'onehot')

model_data = text_prepare(train_df[['recipe_clean','cuisine']], num_words = VOCAB_SIZE, max_len = 30, encode = 'ordinal')
model_data_onehot = text_prepare(train_df[['recipe_clean','cuisine']], num_words = VOCAB_SIZE, max_len = 30, encode = 'onehot', batch_size = 32)

# model wrapper for fully connected model architectures 

def embedding_model(input_length, embedding_dim=16, vocab_size=VOCAB_SIZE, categories=1, lr=1e-3, dense_layers=[256], metrics=['accuracy']):

    if categories >= 3:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        output_activation='softmax'
    else:
        loss = tf.keras.losses.BinaryCrossentropy()
        output_activation='sigmoid'
        
    if type(dense_layers) is not list: # we pass a string to the function to help with pandas sorting, then coerce back to list later
        dense_layers = [int(i) for i in dense_layers.split(',')]
    
    input_layer = tf.keras.Input(shape=input_length)
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=input_length)(input_layer)
    x = tf.keras.layers.Flatten()(x)
    for l in dense_layers:
        x = tf.keras.layers.Dense(l, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(categories, activation=output_activation)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # setup the training parameters
    model.compile(loss=loss,
                  optimizer=tf.keras.optimizers.Adam(lr=lr),
                  metrics=metrics)

    return model

train_data = model_data['train_data'][0].batch(64)
val_data = model_data['val_data'][0].batch(64)

train_shape = iter(train_data).next()[0].shape # verify the length of our tokenized strings. Here, we've used 30
token_length = train_shape[1]

hyperparams = {
    'learning_rate': [1e-3,1e-4,1e-5],
    'embedding_dim': [16,64],
    'dense_layers': ['256','256,256']
}

def gen_hypers(params, continuous = None, limit = 150, frac=1, keep_sorted=True):
    hypers = pd.DataFrame(product(*[params[c] for c in params]),columns=params.keys())
    hypers = hypers.sample(frac=frac).reset_index(drop=True)
    hypers = hypers[:limit]
    if keep_sorted is True:
        hypers = hypers.sort_values(by=list(hypers.columns),ascending=[True for c in hypers.columns]).reset_index(drop=True)
    return hypers

hyperparams_df = gen_hypers(hyperparams)
hyperparams_df

# use an early stopping callback to save training time
early_stopping = EarlyStopping(monitor='val_accuracy', patience=4, mode='max', restore_best_weights=True)

for i in range(len(hyperparams_df)):
    if i==0:
        history = []
        models = []
        preds = []
    learning_rate = hyperparams_df.iloc[i,:]['learning_rate']
    embedding_dim = hyperparams_df.iloc[i,:]['embedding_dim']
    dense_layers = hyperparams_df.iloc[i,:]['dense_layers']
    models.append(embedding_model(input_length=token_length,categories=len(cuisine_groups),lr=learning_rate,
                                  embedding_dim=embedding_dim,dense_layers=dense_layers))
    print('Training Model',i+1,':',', '.join([str(a)+': '+str(b) for a,b in zip(list(hyperparams_df.columns),list(hyperparams_df.iloc[i,:]))]))
    
    epochs = 50 if learning_rate < 1e-4 else 25
    
    history.append(models[-1].fit(
        train_data,
        epochs = epochs,
        validation_data = val_data,
        verbose = 0,
        callbacks = [early_stopping]
    ).history)
    save_model(models[-1],'models/model_dense_'+str(i+1)+'.h5')
    preds.append(np.argmax(models[-1].predict(val_data),axis=1))
history = pd.concat([pd.DataFrame(h) for h in history],axis=1).reset_index()
history.columns = ['epoch']+list(history.columns)[1:]
history.to_csv('model_data/history_dense.csv',index=False)

accuracy = pd.concat([history.iloc[:,i:i+1] for i in range(2,len(history.columns),4)],axis=1)
val_accuracy = pd.concat([history.iloc[:,i:i+1] for i in range(4,len(history.columns),4)],axis=1)

print('Accuracy is:',np.nanmax(np.max(val_accuracy)),'from model',np.argmax(np.max(val_accuracy))+1)

# make a dataframe ot the preds from our perferred model, vs the real values, so we can make a confusion matrix later
preds_real = pd.DataFrame(np.array([preds[6],model_data['val_labels']]).T,columns=['pred','real'])

labels = hyperparams_df.apply(lambda x: str(x['embedding_dim'])+', ('+str(x['dense_layers']+')'),axis=1)

fig,axes = plt.subplots(2, 3, figsize=(20,10))
for i in range(len(accuracy.columns)):
    sns.lineplot(ax=axes[0,i//4], x = history['epoch'], y = accuracy.iloc[:,i], legend='brief', label=labels[i])
    sns.lineplot(ax=axes[1,i//4], x = history['epoch'], y = val_accuracy.iloc[:,i], legend='brief', label=labels[i])

train_data_onehot = model_data_onehot['train_data']
val_data_onehot = model_data_onehot['val_data']

best_model_index = np.argmax(np.max(val_accuracy))
best_hps = list(hyperparams_df.iloc[best_model_index,:])

best_hps

for i in range(len(cuisine_groups)):
    if i==0:
        model_onehot = []
        history_onehot = []
        preds_onehot = []
    model_onehot.append(
        embedding_model(input_length=token_length, categories=1, lr=best_hps[0], embedding_dim=best_hps[1], dense_layers=best_hps[2], metrics=['accuracy'])
    )
    print('Training Model',i+1,':',cuisine_groups[i])
    history_onehot.append(model_onehot[-1].fit(
        train_data_onehot[i],
        epochs = 50,
        validation_data = val_data_onehot[i],
        verbose = 0,
        callbacks = [early_stopping]
    ).history)
    save_model(model_onehot[-1],'models/model_dense_onehot_'+cuisine_groups[i]+'.h5')
    preds_onehot.append(model_onehot[-1].predict(val_data_onehot[0], verbose=0))
history_onehot = pd.concat([pd.DataFrame(h) for h in history_onehot],axis=1).reset_index()
history_onehot.columns = ['epoch']+list(history_onehot.columns)[1:]
history_onehot.to_csv('model_data/history_dense_onehot.csv',index=False)

model_onehot[-1].predict(val_data_onehot[0])

preds_onehot = np.argmax(np.asarray(preds_onehot).squeeze(),axis=0)
real_onehot = np.argmax(model_data_onehot['val_labels'],1)
preds_real_onehot = pd.DataFrame(np.array([preds_onehot,real_onehot]).T,columns=['pred','real'])

accuracy_onehot = pd.concat([history_onehot.iloc[:,i:i+1] for i in range(2,len(history_onehot.columns),4)],axis=1)
val_accuracy_onehot = pd.concat([history_onehot.iloc[:,i:i+1] for i in range(4,len(history_onehot.columns),4)],axis=1)

print('Accuracy is:',np.mean(real_onehot==preds_onehot))

fig,axes = plt.subplots(1, 2, figsize=(20,6))
print('')
for i in range(len(accuracy_onehot.columns)):
    sns.lineplot(ax=axes[0], x = history_onehot['epoch'], y = accuracy_onehot.iloc[:,i], legend='brief',
                 label=cuisine_groups[i]).set_title('TRAINING ACCURACY FOR EACH CUISINE GROUP')
    sns.lineplot(ax=axes[1], x = history_onehot['epoch'], y = val_accuracy_onehot.iloc[:,i], legend='brief',
                 label=cuisine_groups[i]).set_title('VALIDATION ACCURACY FOR EACH CUISINE GROUP')

make_confusion_matrix(preds_real, groups=cuisine_groups)

make_confusion_matrix(preds_real_onehot, groups=cuisine_groups)

confusion = make_confusion_matrix(preds_real, groups=cuisine_groups, raw=True)
confusion_onehot = make_confusion_matrix(preds_real_onehot, groups=cuisine_groups, raw=True)
diag, diag_onehot = [],[]
for i in range(len(cuisine_groups)):
    diag.append(confusion.iloc[i,i])
    diag_onehot.append(confusion_onehot.iloc[i,i])
diag_diff = confusion.T.iloc[:,:1]
diag_diff[0] = np.round(np.array(diag_onehot)-np.array(diag),3)
diag_diff.columns = ['diff']
diag_diff = diag_diff.T

diag_diff

print('Mean Confusion Gain:',np.round(np.sum(np.array(diag_onehot)-np.array(diag)),4)/len(cuisine_groups))

model_lstm_1 = lstm_model(model_data, loss='sparse_categorical_crossentropy', optimizer = 'adam', rnn_depth = 128, embedding_dim = 64,
                           layer_norm = True, categories = 20, activation = 'softmax', metrics=['accuracy'])

process_model(model_data, model_lstm_1, label = 'cuisine', model_name = 'model_lstm', lr = (1e-3, 1e-3), epochs = 20, batch_size = 512, epoch_subset = 1)

plot_history(model_history(),'model_lstm$',val='overlap')

model_lstm_2 = lstm_model(model_data, loss = 'sparse_categorical_crossentropy', optimizer = 'adam', rnn_depth = 512, embedding_dim = 64,
                           layer_norm = False, categories = 20, activation = 'softmax', metrics=['accuracy'])

process_model(model_data, model_lstm_2, label = 'cuisine', model_name = 'model_lstm_2', lr = (5e-5,1e-3),
              epochs = 100, batch_size = 256, epoch_subset = 1, split_epochs=True)

plot_history(model_history(),'model_lstm_2$',val='overlap')

model_lstm_3 = lstm_model(model_data, loss = 'sparse_categorical_crossentropy', optimizer = 'adam', rnn_depth = 512, embedding_dim = 64,
                           layer_norm = True, categories = 20, activation = 'softmax', metrics=['accuracy'])

process_model(model_data, model_lstm_3, label = 'cuisine', model_name = 'model_lstm_3', lr = (5e-5,1e-3),
              epochs = 100, batch_size = 256, epoch_subset = 1, split_epochs=True)

plot_history(model_history(),'model_lstm_3$',val='overlap')

model_lstm_4 = lstm_model(model_data, loss = 'sparse_categorical_crossentropy', optimizer = 'adam', rnn_depth = 512, embedding_dim = 64,
                           layer_norm = True, categories = 20, activation = 'softmax', metrics=['accuracy'])

process_model(model_data, label = 'cuisine', model_name = 'model_lstm_4', lr = (5e-5,1e-3), epochs = 20,
              batch_size = 256, epoch_subset = 1, split_epochs=True)

plot_history(model_history(),'model_lstm_4$',val='overlap')

model_lstm_5 = lstm_model(model_data, loss='sparse_categorical_crossentropy', optimizer='adam', rnn_depth=512, embedding_dim=64,
                          layer_norm = True, categories = 20, activation = 'softmax', metrics=['accuracy'])

process_model(model_data, model_lstm_5, label = 'cuisine', model_name = 'model_lstm_5', lr = (5e-5,1e-3),
              epochs = 20, batch_size = 256, epoch_subset = 1, class_weight = True, split_epochs=True)

plot_history(model_history(),'model_lstm_5$',val='overlap')

model_lstm_6 = lstm_model(model_data, loss = 'sparse_categorical_crossentropy', optimizer = 'adam', rnn_depth = 512, embedding_dim = 64,
                          layer_norm = True, categories = 20, activation = 'softmax', metrics=['accuracy'])

process_model(model_data, model_lstm_6, label = 'cuisine', model_name = 'model_lstm_6', lr = (5e-5,1e-3),
              epochs = 20, batch_size = 16, epoch_subset = 1, class_weight = True, split_epochs=True)

plot_history(model_history(),'model_lstm_6$',val='overlap')

cuisine_groups = list(model_data_onehot['label_encoding'].keys())

model_onehot = lstm_model(model_data_onehot, loss = 'binary_crossentropy', optimizer = 'adam', rnn_depth = 128, embedding_dim = 32,
                          layer_norm = True, categories = 1, activation = 'sigmoid', metrics=[my_auc,'accuracy'])
for i,c in enumerate(cuisine_groups):
    process_model(model_data_onehot, model_onehot, label_class = i, label = 'cuisine_'+c, model_name = 'model_lstm_onehot_'+c,
                  lr = (5e-5,1e-3), epochs = 50, batch_size = 64, epoch_subset = 1, split_epochs = True) #, model_path = MODEL_PATH_TEST, history_path = HISTORY_PATH_TEST)

hist = model_history()
hist = hist[hist['name'].str.contains('model_lstm_onehot_('+('|'.join(cuisine_groups))+')$')].reset_index(drop=True)

hist[['label','epoch','val_loss','val_auc','val_accuracy']].groupby(['label','epoch']).sum().reset_index()

filters = 'model_lstm_onehot_('+('|'.join(cuisine_groups))+')$'
plot_history(model_history(), filters, metrics=['auc','accuracy','loss'], width=3, height=2, tight=False, size=(30,12), val='separate')

# fix these scales!
# show the bootstrapped line only

preds_onehot = []
for i,p in enumerate(cuisine_groups):
    m = load_model(MODEL_PATH+'model_lstm_onehot_'+p+'_best.h5')
    preds_onehot.append(m.predict(model_data_onehot['val_data'][i], verbose=0)) #.map(lambda x,y: x[np.newaxis])

preds_onehot_df = pd.DataFrame([[s[0] for s in p] for p in preds_onehot]).transpose()
preds_onehot_df.columns = cuisine_groups
np.round(preds_onehot_df,5)

onehot_pred = np.asarray([np.argmax(preds_onehot_df.iloc[i,:]) for i in range(len(preds_onehot_df))])[np.newaxis]
onehot_real = np.argmax(model_data_onehot['val_labels'],axis=1)[np.newaxis]
onehot_pred_real = pd.DataFrame(np.concatenate([onehot_pred,onehot_real],axis=0).T,columns=['pred','real'])
onehot_pred_real['acc'] = np.where(onehot_pred_real['pred']-onehot_pred_real['real']==0,1,0)
print('Accuracy is:',sum(onehot_pred_real['acc'])/len(onehot_pred_real))

my_cuisines = ['greek']

for c in my_cuisines:
    model_tune_1 = lstm_model(model_data_onehot, loss = 'binary_crossentropy', optimizer = 'adam', rnn_depth = 128, embedding_dim = 32,
                              layer_norm = True, categories = 1, activation = 'sigmoid', metrics=[my_auc,'accuracy'])
    process_model(model_data_onehot, model_tune_1, label_class = 6, label = 'cuisine_'+c, model_name = 'model_lstm_onehot_'+c+'_v2', lr = (1e-6,5e-4),
                  epochs = 100, batch_size = 64, epoch_subset = 1, split_epochs=True)

    model_tune_2 = lstm_model(model_data_onehot, loss = 'binary_crossentropy', optimizer = 'adam', rnn_depth = 512, embedding_dim = 64,
                              layer_norm = True, categories = 1, activation = 'sigmoid', metrics=[my_auc,'accuracy'])
    process_model(model_data_onehot, model_tune_2, label_class = 6, label = 'cuisine_'+c, model_name = 'model_lstm_onehot_'+c+'_v3', lr = (1e-5,1e-5),
                  epochs = 20, batch_size = 128, epoch_subset = 1, split_epochs=True)

    model_tune_3 = lstm_model(model_data_onehot, loss = 'binary_crossentropy', optimizer = 'adam', rnn_depth = 64, embedding_dim = 6,
                              layer_norm = True, categories = 1, activation = 'sigmoid', metrics=[my_auc,'accuracy'])
    process_model(model_data_onehot, model_tune_3, label_class = 6, label = 'cuisine_'+c, model_name = 'model_lstm_onehot_'+c+'_v3', lr = (1e-5,1e-5),
                  epochs = 20, batch_size = 16, epoch_subset = 1, split_epochs=True)

filters = 'model_lstm_onehot_('+('|'.join(cuisine_groups))+')_v.$'
plot_history(model_history(), filters, group='name', metrics=['val_auc','val_accuracy','val_loss'], width=3, height=1, tight=False, size=(30,6))

np.random.seed(79)
train_df_shuffled = train_df.iloc[:,:3].sample(frac=1).reset_index(drop=True)
if type(train_df_shuffled['ingredients'][0])==str:
    train_df_shuffled['ingredients'] = train_df_shuffled['ingredients'].apply(lambda x: eval(x))
train_df_shuffled.head()

copies = 20
removal_rate = 0.4

train_df_shuffled['recipe_concat'] = train_df_shuffled['ingredients'].apply(lambda x: ' '.join([re.sub(' ','_',i) for i in x]))
train_df_shuffled['recipe_clean'] = train_df_shuffled['recipe_concat'].apply(lambda x: text_clean(x))

id2 = np.arange(len(train_df_shuffled))
random.shuffle(id2)
id2 = pd.DataFrame(id2,columns=['id2'])
train_df_shuffled = pd.concat([train_df_shuffled,id2],axis=1)

for i in range(copies):
    train_df_shuffled_expanded = train_df_shuffled.copy() if i==0 else train_df_shuffled_expanded.append(train_df_shuffled.copy())
train_df_shuffled_expanded = train_df_shuffled_expanded.sort_values(by='id2').reset_index(drop=True)

train_df_shuffled_expanded

def remove_underscores(col,removal_rate):
    c_new = ''
    for c in col:
        if c == '_':
            c_new += (' ' if np.random.random()<removal_rate else '')
        else:
            c_new += c
    return c_new

train_df_shuffled_expanded['recipe_clean'] = train_df_shuffled_expanded['recipe_clean'].apply(lambda x: remove_underscores(x,removal_rate))

# locked in!
try:
    train_df_shuffled_expanded = pd.read_csv('data/train_df_shuffled_expanded.csv')
except:
    train_df_shuffled_expanded.to_csv('data/train_df_shuffled_expanded.csv')

train_df_shuffled_expanded

model_data_expanded = text_prepare(train_df_shuffled_expanded[['recipe_clean','id2','cuisine']], num_words = VOCAB, max_len = 40, encode = 'ordinal', batch_size = 512)
model_data_onehot_expanded = text_prepare(train_df_shuffled_expanded[['recipe_clean','id2','cuisine']], num_words = VOCAB, max_len = 40, encode = 'onehot', batch_size = 512)

# onehot
model_onehot_expanded = lstm_model(model_data_onehot_expanded, loss = 'binary_crossentropy', optimizer = 'adam', rnn_depth = 64, embedding_dim = 32,
                                   layer_norm = True, categories = 1, activation = 'sigmoid', metrics=[my_auc,'accuracy'])
for i,c in enumerate(cuisine_groups):
    process_model(model_data_onehot_expanded, model_onehot_expanded, label_class = i, label = 'cuisine_'+c, model_name = 'model_lstm_onehot_'+c+'_expanded',
                  lr = (5e-6, 5e-5), epochs = 20, batch_size = 512, epoch_subset = 1, split_epochs = True)

preds_onehot_expanded = []
for i,p in enumerate(cuisine_groups):
    m = load_model(MODEL_PATH+'model_lstm_onehot_'+p+'_expanded_best')
    preds_onehot_expanded.append(m.predict(model_data_onehot_expanded['val_data'][i], verbose=0))

preds_onehot_expanded_df = pd.DataFrame([[s[0] for s in p] for p in preds_onehot_expanded]).transpose()
preds_onehot_expanded_df.columns = cuisine_groups

onehot_pred_expanded = np.asarray([np.argmax(preds_onehot_expanded_df.iloc[i,:]) for i in range(len(preds_onehot_expanded_df))])[np.newaxis]
onehot_real_expanded = np.argmax(model_data_onehot_expanded['val_labels'],axis=1)[np.newaxis]
onehot_pred_real_expanded = pd.DataFrame(np.concatenate([onehot_pred_expanded,onehot_real_expanded],axis=0).T,columns=['pred','real'])
onehot_pred_real_expanded['accuracy'] = np.where(onehot_pred_real_expanded['pred']-onehot_pred_real_expanded['real']==0,1,0)
print('Accuracy is:',sum(onehot_pred_real_expanded['accuracy'])/len(onehot_pred_real_expanded),'\n')

filters = 'model_lstm_onehot.*expanded'
plot_history(model_history(), filters, metrics=['auc','accuracy','loss'], width=3, height=2, tight=False, size=(30,12), val='separate')

make_confusion_matrix(onehot_pred_real_expanded, groups=cuisine_groups)

# multiclass - complete :)
model_wrapper_new(
    model_data_expanded, label = 'cuisine', model_name = 'model_lstm_expanded', lr = (5e-6, 1e-4), epochs = 20, split_epochs = True,
    batch_size = 512, epoch_subset = 1, rnn_depth = 64, embedding_dim = 32, layer_norm = True, multiclass = True
)

model_lstm_expanded_best = load_model('models/model_lstm_expanded_best')
model_lstm_expanded_best_preds = model_lstm_expanded_best.predict(model_data_expanded['val_data'][0], verbose=0)
best_preds_max = np.array([np.argmax(x) for x in model_lstm_expanded_best_preds])
pred_real_expanded = pd.concat([pd.DataFrame(np.array(best_preds_max)),pd.DataFrame(model_data_expanded['val_labels'])],axis=1)
pred_real_expanded.columns = ['pred','real']
pred_real_expanded['accuracy'] = pred_real_expanded.apply(lambda r: 1 if r['pred']==r['real'] else 0,axis=1)
print('Accuracy is: ',sum(pred_real_expanded['accuracy'])/len(pred_real_expanded),'\n')

filters = '^model_lstm_expanded$'
plot_history(model_history(), filters, metrics=['accuracy','loss'], width=2, height=1, tight=False, size=(30,8), val='overlap')

make_confusion_matrix(preds_df_expanded, groups=cuisine_groups)



embedding_dim = [8,16,32,64]
rnn_dim = [64,256,1024]
dropout = [0.0,0.2,0.4]
recurrent_dropout = [0.0,0.1,0.2,0.3,0.4,0.5]
kernel_regularizer = [0.0,0.1,0.2,0.3,0.4,0.5]
recurrent_regularizer = [None,'l1','l2','l1_l2']
bias_regularizer = [None,'l1','l2','l1_l2']
activity_regularizer = [None,'l1','l2','l1_l2']

model_wrapper_new(
    model_data_expanded, label = 'cuisine', model_name = 'model_lstm_expanded', lr = (5e-6,1e-4), epochs = 7, split_epochs = True,
    batch_size = 512, epoch_subset = 1, rnn_depth = 64, embedding_dim = 32, layer_norm = True, multiclass = True
)









