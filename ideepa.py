import sys
import os
import numpy as np
import pdb
#sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages/Keras-2.0.4-py2.7.egg')
from keras.models import Sequential
import keras.layers.core as core
import keras.layers.convolutional as conv
from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers import normalization, Lambda, GlobalMaxPooling2D
from keras.layers import LSTM, Bidirectional, Reshape
from keras.layers.embeddings import Embedding
#from keras.layers.convolutional import Conv2D, MaxPooling2D,Conv1D, MaxPooling1D
from keras.layers import merge, Input
from keras.layers.normalization import BatchNormalization
from keras.regularizers import WeightRegularizer, l1, l2
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.np_utils import to_categorical
#from customlayers import Recalc, ReRank, ExtractDim, SoftReRank, ActivityRegularizerOneDim, RecalcExpand, Softmax4D
from keras.constraints import maxnorm
from attention import Attention,myFlatten

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras import objectives
from keras import backend as K
#from keras.utils import np_utils, plot_model
from sklearn.metrics import roc_curve, auc, roc_auc_score
_EPSILON = K.epsilon()
import random
import gzip
import pickle
import timeit

def read_fasta_file(fasta_file):
    seq_dict = {}    
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()
        #distinguish header from sequence
        if line[0]=='>': #or line.startswith('>')
            #it is the header
            name = line[1:] #discarding the initial >
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper()
    fp.close()
    
    return seq_dict

def padding_sequence(seq, max_len = 501, repkey = 'N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq
def read_rna_dict(rna_dict = 'rna_dict'):
    odr_dict = {}
    with open(rna_dict, 'r') as fp:
        for line in fp:
            values = line.rstrip().split(',')
            for ind, val in enumerate(values):
                val = val.strip()
                odr_dict[val] = ind
    
    return odr_dict

def split_training_validation(classes, validation_size = 0.2, shuffle = False):
    """split sampels based on balnace classes"""
    num_samples=len(classes)
    classes=np.array(classes)
    classes_unique=np.unique(classes)
    num_classes=len(classes_unique)
    indices=np.arange(num_samples)
    #indices_folds=np.zeros([num_samples],dtype=int)
    training_indice = []
    training_label = []
    validation_indice = []
    validation_label = []
    for cl in classes_unique:
        indices_cl=indices[classes==cl]
        num_samples_cl=len(indices_cl)

        # split this class into k parts
        if shuffle:
            random.shuffle(indices_cl) # in-place shuffle
        
        # module and residual
        num_samples_each_split=int(num_samples_cl*validation_size)
        res=num_samples_cl - num_samples_each_split
        
        training_indice = training_indice + [val for val in indices_cl[num_samples_each_split:]]
        training_label = training_label + [cl] * res
        
        validation_indice = validation_indice + [val for val in indices_cl[:num_samples_each_split]]
        validation_label = validation_label + [cl]*num_samples_each_split

    training_index = np.arange(len(training_label))
    random.shuffle(training_index)
    training_indice = np.array(training_indice)[training_index]
    training_label = np.array(training_label)[training_index]
    
    validation_index = np.arange(len(validation_label))
    random.shuffle(validation_index)
    validation_indice = np.array(validation_indice)[validation_index]
    validation_label = np.array(validation_label)[validation_index]    
    
            
    return training_indice, training_label, validation_indice, validation_label 


def read_seq_graphprot(seq_file, label = 1):
    seq_list = []
    labels = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper()
                seq = seq.replace('T', 'U')
                seq_list.append(seq)
                labels.append(label)
    
    return seq_list, labels

def get_RNA_seq_concolutional_array(seq, motif_len = 4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    #for seq in seqs:
    #for key, seq in seqs.iteritems():
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)
        
    #pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        #if val == 'N' or i < motif_len or i > len(seq) - motif_len:
        #    new_array[i] = np.array([0.25]*4)
        #else:
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        #data[key] = new_array
    return new_array

def get_RNA_concolutional_array(seq, motif_len = 4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    #for seq in seqs:
    #for key, seq in seqs.iteritems():
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)
        
    #pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        #data[key] = new_array
    return new_array

def load_graphprot_data(protein, train = True, path = '../GraphProt_CLIP_sequences/'):
    data = dict()
    tmp = []
    listfiles = os.listdir(path)
    
    key = '.train.'
    if not train:
        key = '.ls.'
    mix_label = []
    mix_seq = []
    mix_structure = []    
    for tmpfile in listfiles:
        if protein not in tmpfile:
            continue
        if key in tmpfile:
            if 'positive' in tmpfile:
                label = 1
            else:
                label = 0
            seqs, labels = read_seq_graphprot(os.path.join(path, tmpfile), label = label)
            #pdb.set_trace()
            mix_label = mix_label + labels
            mix_seq = mix_seq + seqs
    
    data["seq"] = mix_seq
    data["Y"] = np.array(mix_label)
    
    return data

def loaddata_graphprot(protein, train = True, ushuffle = True):
    #pdb.set_trace()
    data = load_graphprot_data(protein, train = train)
    label = data["Y"]
    rna_array = []
    #trids = get_6_trids()
    #nn_dict = read_rna_dict()
    for rna_seq in data["seq"]:
        #rna_seq = rna_seq_dict[rna]
        rna_seq = rna_seq.replace('T', 'U')
        
        seq_array = get_RNA_seq_concolutional_array(seq)
        #tri_feature = get_6_nucleotide_composition(trids, rna_seq_pad, nn_dict)
        rna_array.append(seq_array)
    
    return np.array(rna_array), label

def get_bag_data(data):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    longlen = 0
    for seq in seqs:
        bag_seq = padding_sequence(seq, max_len = 501)
        tri_fea = get_RNA_concolutional_array(bag_seq)
        bags.append(tri_fea)

    return bags, labels # bags,

def get_rnarecommend(rnas, rna_seq_dict, shuffle = True):
    data = {}
    label = []
    rna_seqs = []
    if not shuffle:
        all_rnas = set(rna_seq_dict.keys()) - set(rnas)
        all_rnas = list(all_rnas)
        random.shuffle(all_rnas)
    ind = 0    
    for rna in rnas:
        rna_seq = rna_seq_dict[rna]
        rna_seq = rna_seq.replace('T', 'U')
        label.append(1)
        rna_seqs.append(rna_seq)
        label.append(0)
        if shuffle:
            shuff_seq = doublet_shuffle(rna_seq)
        else:
            sele_rna = all_rnas[ind]
            shuff_seq = rna_seq_dict[sele_rna]
            ind = ind + 1
            
        rna_seqs.append(shuff_seq)
    data["seq"] = rna_seqs
    data["Y"] = np.array(label)
    
    return data

def get_bag_data_1_channel(seqs, labels, max_len = 501):
    bags = []
    #seqs = data["seq"]
    #labels = data["Y"]
    for seq in seqs:
        bag_seq = padding_sequence(seq, max_len = max_len)
        #flat_array = []
        bag_subt = []
        #for bag_seq in bag_seqs:
        tri_fea = get_RNA_seq_concolutional_array(bag_seq)
        #bag_subt.append(tri_fea.T)
        bags.append(np.array(tri_fea))    
    return bags, labels


def read_seq_again(seq_file):
    seq_list = []
    labels = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line.rstrip().split()
                label = int(name[-1])
                labels.append(label)
            else:
                seq = line[:-1].upper()
                seq_list.append(seq)

    return seq_list, labels  
        
def get_all_embedding(protein):
    
    data = load_graphprot_data(protein)
    #pdb.set_trace()
    train_bags, label = get_bag_data(data)
    #pdb.set_trace()
    test_data = load_graphprot_data(protein, train = False)
    test_bags, true_y = get_bag_data(test_data) 
    
    return train_bags, label, test_bags, true_y

def set_cnn_model_attention(input_dim = 4, input_length = 507):
	attention_reg_x = 0.25
	attention_reg_xr = 1
	attentionhidden_x = 16
	attentionhidden_xr = 8
	nbfilter = 16
	input = Input(shape=(input_length, input_dim))
	x = conv.Convolution1D(nbfilter, 10 ,border_mode="valid")(input) 
	x = Dropout(0.5)(x)
	x = Activation('relu')(x)
	x = conv.MaxPooling1D(pool_length=3)(x)
	x_reshape=core.Reshape((x._keras_shape[2],x._keras_shape[1]))(x)

	x = Dropout(0.5)(x)
	x_reshape=Dropout(0.5)(x_reshape)

	decoder_x = Attention(hidden=attentionhidden_x,activation='linear') # success  
	decoded_x=decoder_x(x)
	output_x = myFlatten(x._keras_shape[2])(decoded_x)

	decoder_xr = Attention(hidden=attentionhidden_xr,activation='linear')
	decoded_xr=decoder_xr(x_reshape)
	output_xr = myFlatten(x_reshape._keras_shape[2])(decoded_xr)

	output=merge([output_x,output_xr, Flatten()(x)],mode='concat')
        #output = BatchNormalization()(output)
	output=Dropout(0.5)(output)
        print output.shape
	output=Dense(nbfilter*10,activation="relu")(output)
	output=Dropout(0.5)(output)
	out=Dense(2,activation='softmax')(output)
        #output = BatchNormalization()(output)
	model=Model(input,out)
	model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

	return model

def run_network_att(model, total_hid, train_bags, test_bags, y_bags):
    #model.add(Dense(2)) # binary classification
    #model.add(Activation('softmax')) # #instance * 2
    #model.add(GlobalMaxPooling2D()) # max pooling multi instance 

    #model.summary()
    print(len(train_bags), len(test_bags), len(y_bags), train_bags[0].shape, len(train_bags[0]))

    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    #model.compile(loss=custom_objective, optimizer='rmsprop')
    print 'model training'
    nb_epochs= 30
    y_bags = to_categorical(y_bags, 2)
    #pdb.set_trace()
    model.fit(np.array(train_bags), np.array(y_bags), batch_size = 100, nb_epoch=nb_epochs, verbose = 0)
    print 'predicting' 
    #pdb.set_trace()        
    predictions = model.predict(np.array(test_bags))[:, 1]

    return predictions

def get_all_mildata(protein, dataset = 'graphprot'):
    data = load_graphprot_data(protein)

    train_bags, label = get_bag_data(data)

    test_data = load_graphprot_data(protein, train = False)
    test_bags, true_y = get_bag_data(test_data) 
    
    return train_bags, label, test_bags, true_y

def run_graphprot_ideepa():

    data_dir = '../GraphProt_CLIP_sequences/'

    fw = open('result_micnn_att', 'w')
    print(len(os.listdir(data_dir)))
    print(os.listdir(data_dir))
    finished_protein = set()
    for protein in os.listdir(data_dir):
        
        protein = protein.split('.')[0]

        if protein in finished_protein:
            continue
        finished_protein.add(protein)
        print protein
        fw.write(protein + '\t')
        train_bags, train_labels, test_bags, test_labels = get_all_mildata(protein)

        net =  set_cnn_model_attention()
        
        #seq_auc, seq_predict = calculate_auc(seq_net)
        hid = 16
        predict = run_network_att(net, hid, train_bags, test_bags, train_labels)
        
        auc = roc_auc_score(test_labels, predict)
        print 'AUC:', auc
        fw.write(str(auc) + '\n')
        mylabel = "\t".join(map(str, test_labels))
        myprob = "\t".join(map(str, predict))  
        fw.write(mylabel + '\n')
        fw.write(myprob + '\n')
    
    fw.close()


run_graphprot_ideepa()

