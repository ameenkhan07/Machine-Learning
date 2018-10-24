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
GSC_features = 'GSC-Dataset/GSC-Features.csv'
# def _save(filename):
#     if not os.path.exists(OUTPUT_DIR):
#             os.makedirs(OUTPUT_DIR)


def _concat_dataframes(same_data, diff_data, features_data):
    """
    """
    # Combine the diff and concat data
    merged_data = pd.concat([same_data, diff_data])
    # Merge the datatables, same and diff
    merged_data = pd.merge(merged_data, features_data,
                           left_on='img_id_A', right_on='img_id', how='left')
    merged_data = pd.merge(merged_data, features_data, left_on='img_id_B',
                           right_on='img_id', how='left', suffixes=('_a', '_b'))
    # Slight preprocessing
    # merged_data = merged_data.drop(
    #     columns=['Unnamed: 0_a', 'Unnamed: 0_b'])
    return(merged_data)


def _subtract_dataframes(same_data, diff_data, features_data):
    """Subtract img_id_a columns with img_id_b_columns
    """
    merged_data = _concat_dataframes(same_data, diff_data, features_data)
    merged_data['f1'] = abs(merged_data['f1_a']-merged_data['f1_b'])
    merged_data['f2'] = abs(merged_data['f2_a']-merged_data['f2_b'])
    merged_data['f3'] = abs(merged_data['f3_a']-merged_data['f3_b'])
    merged_data['f4'] = abs(merged_data['f4_a']-merged_data['f4_b'])
    merged_data['f5'] = abs(merged_data['f5_a']-merged_data['f5_b'])
    merged_data['f6'] = abs(merged_data['f6_a']-merged_data['f6_b'])
    merged_data['f7'] = abs(merged_data['f7_a']-merged_data['f7_b'])
    merged_data['f8'] = abs(merged_data['f8_a']-merged_data['f8_b'])
    merged_data['f9'] = abs(merged_data['f9_a']-merged_data['f9_b'])
    merged_data = merged_data.drop(
        columns=['img_id_a', 'img_id_b'])
    merged_data = merged_data.drop(columns=['f1_a', 'f2_a', 'f3_a',
                                            'f4_a', 'f5_a', 'f6_a', 'f7_a', 'f8_a', 'f9_a'])
    merged_data = merged_data.drop(columns=['f1_b', 'f2_b', 'f3_b',
                                            'f4_b', 'f5_b', 'f6_b', 'f7_b', 'f8_b', 'f9_b'])
    # print(merged_data.shape, merged_data.columns.values, merged_data.loc[1, :])
    return(merged_data)


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
        merged_data = _concat_dataframes(same_data, diff_data, features_data)

    if operation == 'subtract':
        merged_data = _subtract_dataframes(same_data, diff_data, features_data)
    
    merged_data = merged_data.drop(
        columns=['img_id_a', 'img_id_b'])
    merged_data = merged_data.sample(frac=1)
    print(
        f"Data Shape : {merged_data.shape}, Data Columns : {merged_data.columns.values}")
    return(merged_data)


if __name__ == '__main__':
    get_data_features('hod', operation='concat')
    get_data_features('hod', operation='subtract')
    get_data_features('gsc', operation='concat')
    # get_data_features('gsc', operation='subtract')
