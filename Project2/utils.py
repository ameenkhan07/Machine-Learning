import pandas as pd
import numpy as np
import os

OUTPUT_DIR = 'outputs/'
INPUT_DIR = 'data/'

HOD_same = 'HumanObserved-Dataset/same_pairs.csv'
HOD_diff = 'HumanObserved-Dataset/diffn_pairs.csv'
HOD_features = 'HumanObserved-Dataset/HumanObserved-Features-Data.csv'

GSC_same = 'GSC-Dataset/same_pairs.csv'
GSC_diff = 'GSC-Dataset/same_pairs.csv'
GSC_diff = 'GSC-Dataset/diffn_pairs.csv'
GSC_features = 'GSC-Dataset/GSC-Features.csv'
# def _save(filename):
#     if not os.path.exists(OUTPUT_DIR):
#             os.makedirs(OUTPUT_DIR)

def _append_dataframes(same_data, diff_data):
    """Appends the dataframes one after another.
    """
    appended_data = pd.concat([same_data, diff_data])
    appended_data = appended_data.sample(frac=1)
    return(appended_data)
    # return(pd.concat([same_data, diff_data]))


def _concat_dataframes(same_data, diff_data, features_data):
    """
    """
    # Combine the diff and concat data
    appended_data = _append_dataframes(same_data, diff_data)
    if 'Unnamed: 0' in appended_data:
        appended_data.drop(['Unnamed: 0'])
    # Merge the datatables, same and diff
    merged_data = pd.merge(appended_data, features_data,
                           left_on='img_id_A', right_on='img_id', how='left')
    merged_data = pd.merge(merged_data, features_data, left_on='img_id_B',
                           right_on='img_id', how='left', suffixes=('_a', '_b'))
    merged_data = merged_data.drop(columns = ['img_id_a', 'img_id_b'])
    # print(merged_data.loc[:, 'f1_a':])
    raw_data = merged_data.loc[:, 'f1_a':].values
    raw_target = merged_data['target'].values
    return(raw_data, raw_target)


def _subtract_dataframes(same_data, diff_data, features_data):
    """Subtract img_id_a columns with img_id_b_columns
    """
    appended_data = _append_dataframes(same_data, diff_data)
    temp1 = pd.merge(appended_data, features_data,
                     left_on='img_id_A', right_on='img_id', how='left')
    temp2 = pd.merge(appended_data, features_data,
                     left_on='img_id_B', right_on='img_id', how='left')
    raw_data = abs(temp1.loc[:, 'f1':] - temp2.loc[:, 'f1':])
    # print(raw_data.columns.values)
    raw_target = appended_data['target'].values

    return(raw_data, raw_target)


def get_data_features(category='hod', operation='concat'):
    """Returns concat/subtracted feature values
    Merge the same and diff datasets, and process the data columns
    """
    category_dict = {
        'hod': [HOD_same, HOD_diff, HOD_features],
        'gsc': [GSC_same, GSC_diff, GSC_features],
    }
    same, diff, features = category_dict[category]
    same_data = pd.read_csv(os.path.join(INPUT_DIR, same), sep=',', nrows=1000)
    diff_data = pd.read_csv(os.path.join(
        INPUT_DIR, diff),  sep=',', nrows=1000)
    features_data = pd.read_csv(os.path.join(INPUT_DIR, features), sep=',')

    if operation == 'concat':
        raw_data, raw_target = _concat_dataframes(same_data, diff_data, features_data)

    if operation == 'subtract':
        raw_data, raw_target = _subtract_dataframes(same_data, diff_data, features_data)

    return(raw_data, raw_target)


if __name__ == '__main__':
    data1_d, data1_f = get_data_features('hod', operation='concat')
    print(data1_d.shape, data1_f.shape)
    data2_d, data2_f = get_data_features('hod', operation='subtract')
    print(data2_d.shape, data2_f.shape)
    data3_d, data3_f = get_data_features('gsc', operation='concat')
    print(data3_d.shape, data3_f.shape)
    data4_d, data4_f = get_data_features('gsc', operation='subtract')
    print(data4_d.shape, data4_f.shape)
