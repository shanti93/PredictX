from __future__ import unicode_literals
import pandas as pd
import os
import sys
from sklearn.preprocessing import LabelEncoder
import numpy as np



# Create the necessary folders

if not os.path.isdir('input/cleaned_data'):
    os.makedirs('input/cleaned_data')

if not os.path.isdir('./trained_models'):
    os.makedirs('./trained_models')

if not os.path.isdir('./forcasts'):
    os.makedirs('./forcasts')


def getValuesOfPage(first_column):
    first_column = first_column.split('_')
    return ' '.join(first_column[:-3]), first_column[-3], first_column[-2], first_column[-1]


def reArrange(dataMatrix):
    for i in range(dataMatrix.shape[0]):
        value_fill = None
        for j in range(dataMatrix.shape[1] - 3, dataMatrix.shape[1]):
            if np.isnan(dataMatrix[i, j]) and value_fill is not None:
                dataMatrix[i, j] = value_fill
            else:
                value_fill = dataMatrix[i, j]
    return dataMatrix


inputDataFrame = pd.read_csv('input/train.csv', encoding='utf-8')

date_cols = [i for i in inputDataFrame.columns if i != 'Page']

inputDataFrame['name'], inputDataFrame['project'], inputDataFrame['access'], inputDataFrame['agent'] = zip(*inputDataFrame['Page'].apply(getValuesOfPage))



le = LabelEncoder()
#print le.fit(inputDataFrame['Page'])
#print len(list(le.classes_))
#sys.exit()
inputDataFrame['project'] = le.fit_transform(inputDataFrame['project'])
inputDataFrame['access'] = le.fit_transform(inputDataFrame['access'])
inputDataFrame['agent'] = le.fit_transform(inputDataFrame['agent'])
inputDataFrame['page_id'] = le.fit_transform(inputDataFrame['Page'])

#print inputDataFrame['name']
#print inputDataFrame['project']
#print inputDataFrame['access']
#print inputDataFrame['agent']





inputDataFrame[['page_id', 'Page']].to_csv('input/cleaned_data/page_ids.csv', encoding='utf-8', index=False)

data = inputDataFrame[date_cols].values

#test_data = reArrange(inputDataFrame[date_cols].values)
#sys.exit()
#print "*****", inputDataFrame['project'].values
#print "11111", inputDataFrame['project'].shape
#print "*****", inputDataFrame['agent'].values
#print "11111", inputDataFrame['agent'].shape
#print "*****", inputDataFrame['page_id'].values
#print "11111", inputDataFrame['page_id'].shape



np.save('input/cleaned_data/data.npy', np.nan_to_num(data))
np.save('input/cleaned_data/is_nan.npy', np.isnan(data).astype(int))
np.save('input/cleaned_data/project.npy', inputDataFrame['project'].values)
np.save('input/cleaned_data/access.npy', inputDataFrame['access'].values)
np.save('input/cleaned_data/agent.npy', inputDataFrame['agent'].values)
np.save('input/cleaned_data/page_id.npy', inputDataFrame['page_id'].values)

test_data = reArrange(inputDataFrame[date_cols].values)
np.save('input/cleaned_data/test_data.npy', np.nan_to_num(test_data))
np.save('input/cleaned_data/test_is_nan.npy', np.isnan(test_data).astype(int))
