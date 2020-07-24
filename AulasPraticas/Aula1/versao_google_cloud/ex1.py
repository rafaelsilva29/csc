import numpy as np 
import pandas as pd

# Load data
def load_data(file_url):
    data = pd.read_csv(file_url)
    return data

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

'''
Use a pandas' function to apply one-hot encoding to the origin column 
'''
def hot_ecoding(data):
  data_pandas_ohe = pd.get_dummies(data,prefix=['carrier','origin','dest'])
  return data_pandas_ohe
