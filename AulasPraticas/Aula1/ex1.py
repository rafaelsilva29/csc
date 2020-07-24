import numpy as np 
import pandas as pd

file = 'flights_dataset.csv'

# Load data
def load_data():
    data = pd.read_csv(file)
    return data

# Data contain all data readed
data = load_data()

# Print dataset info
print(data.info())
print('---- All data -------')

# Drop unwanted columns
data.drop(['hour','minute','tailnum'], 1, inplace=True)

# Print dataset info
print(data.info())
print('---- Dropped data -------')

# Infer objects type
data.infer_objects()

# Check and replace missing with -99 (masking)
print(data.isnull().sum()) 
print('---- Count num feature -------')
data.fillna(-99, inplace=True)

# Frequency distribution of categories within a feature 'dest'
print(data['dest'].unique())
print('Unique count(dest): %d' %data['dest'].value_counts().count()) 
print('---- Valores unicos de dest -------')
print(data['dest'].value_counts())
print('---- Count num feature dest -------')


'''
Function to encode all non-(int/float) features in a dataframe.
For each column, if its dtype is neither int or float, get the list of unique values,
store the relation between the label and the integer that encodes it and apply it.
Return a labelled dataframe and a dictionary label to be able to restore the original value. 
'''
def label_encoding(data):
    data_carrier = data['carrier'].unique()
    data_origin = data['origin'].unique()
    data_dest = data['dest'].unique()
    
    # Init variables
    label = 0
    dictionary_carrier = []
    dictionary_origin = []
    dictionary_dest = []
    label_dictionary = []

    for carrier in data_carrier:
        temp = (carrier,label)
        dictionary_carrier.append(temp)
        new_label = str(label)
        data['carrier'].replace(
            to_replace=[carrier],
            value=new_label,
            inplace=True
        )
        label += 1
    
    label = 0
    data['carrier'] = data['carrier'].astype(int) 

    for origin in data_origin:
        temp = (origin,label)
        dictionary_origin.append(temp)
        new_label = str(label)
        data['origin'].replace(
            to_replace=[origin],
            value=new_label,
            inplace=True
        )
        label += 1
    
    label = 0
    data['origin'] = data['origin'].astype(int) 

    for dest in data_dest:
        temp = (dest,label)
        dictionary_dest.append(temp)
        new_label = str(label)
        data['dest'].replace(
            to_replace=[dest],
            value=new_label,
            inplace=True
        )
        label += 1
    
    data['dest'] = data['dest'].astype(int) 
    
    label_dictionary.append(['carrier',dictionary_carrier])
    label_dictionary.append(['origin',dictionary_origin])
    label_dictionary.append(['dest',dictionary_dest])
 
    return data, label_dictionary

'''
Function to encode all non-(int/float) features in a dataframe. (v2.0)
'''
def label_encoding_v2(data):
    dic = {}
    for col in data.columns:
        if data[col].dtype == np.object:
            dic[col] = {}
    for col,diccol in dic.items():
        i = 0
        while i< data[col].value_counts().count():
            diccol[data[col].unique()[i]] = i
            i += 1
    data_labelled = data.replace(to_replace=dic, value=None)
    return data_labelled, dic

'''
Function to decode what was previously encoded - get the original value! 
'''
def label_decoding(data_labelled, label_dictionary):
    data_decoded = data_labelled
    
    for item in label_dictionary:
        for tupl in item[1]:
            data_decoded[item[0]].replace(
                to_replace=[tupl[1]],
                value=tupl[0],
                inplace=True
            )

    data_decoded['carrier'] = data_decoded['carrier'].astype(str)
    data_decoded['origin'] = data_decoded['origin'].astype(str)
    data_decoded['dest'] = data_decoded['dest'].astype(str)

    return data_decoded


data_labelled, label_dictionary = label_encoding(data)
print(data_labelled['dest'].unique())
print('Unique count after Label Encoding: %d' %data_labelled['dest'].value_counts().count())
print('--- Valores codificados -----')

data_labelled_decoded = label_decoding(data_labelled, label_dictionary) 
print(data_labelled_decoded['dest'].unique())
print('Unique count after dec.: %d' %data_labelled_decoded['dest'].value_counts().count()) 
print('--- Valores descodificados -----')


'''
Use a pandas' function to apply one-hot encoding to the origin column 
'''
print(data.columns.values)
print('--- Valores originais ---')
data_pandas_ohe = pd.get_dummies(data,prefix=['carrier','origin','dest'])
print(data_pandas_ohe.columns.values)
print('--- Valores depois de one-hot encoding ---')
