import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split


one_hot_params = ['city', 'bd', 'gender', 'registered_via', 'language', 
                  'source_system_tab', 'source_screen_name', 'source_type',
                  'song_year', 'song_country']

custom_one_hot_params = ['genre_ids']

normalize_params = ['song_length', 'membership_days', 'number_of_time_played',
                    'user_activity', 'lyricists_count', 'composer_count',
                    'artist_count', 'number_of_genres']

extra_params_data = ['artist_composer_lyricist', 'is_featured', 'artist_composer']
extra_params_res = ['target']


def make_one_hot(df):
    one_hot_len = {}
    one_hots = []
    for param in one_hot_params:
        df_with_param = df[param]
        one_hot_target = pd.get_dummies(df_with_param)
        one_hot_len[param] = one_hot_target.shape[1]
        df_columns = list(one_hot_target.columns)
        new_columns = {el: str(el) + '_' + param for el in df_columns}
        one_hot_target = one_hot_target.rename(columns=new_columns)
        one_hots.append(one_hot_target.values.tolist())
    return one_hot_len, one_hots


def make_custom_one_hot(df):
    custom_one_hot_len = {}
    custom_one_hots = []
    df_size = df.shape[0]
    for param in custom_one_hot_params:
        df_with_param = df[param]
        df_all_values = []
        for val in df_with_param.values:
            val = str(val)
            df_all_values.append(val.split('|'))
        values = set()
        for vals in df_all_values:
            for val in vals:
                values.add(val)
        values = {val: i for i, val in enumerate(values)}
        one_hot_target = np.zeros((df_size, len(values)))
        for i, vals in enumerate(df_all_values):
            for val in vals:
                one_hot_target[i, values[val]] = 1
        custom_one_hot_len[param] = one_hot_target.shape[1]
        custom_one_hots.append(one_hot_target.tolist())
    return custom_one_hot_len, custom_one_hots


def make_normalize(df):
    normalized = []
    df_size = df.shape[0]
    for param in normalize_params:
        df_with_param = df[param]
        df_array = [np.array(df_with_param.values)]
        norm_res = preprocessing.normalize(df_array).reshape(df_size, 1)
        normalized.append(norm_res.tolist())
    return normalized


def get_extra_params(df, extra_params):
    res_extra_params = []
    for param in extra_params:
        res = np.array(df[param]).reshape(-1, 1).tolist()
        res_extra_params.append(res)
    return res_extra_params

def get_padding(feat, max_len):
    if len(feat) == max_len:
        return feat
    return feat + [0] * (max_len - len(feat))

def get_input_tensor(df, test_size=0.2):
    df = df.dropna()
    one_hot_train_len, one_hot_train_subs = make_one_hot(df)
    custom_one_hot_len, custom_one_hots = make_custom_one_hot(df)
    normalized_train_subs = make_normalize(df)
    bool_train_subs = get_extra_params(df, extra_params_data)

    y = get_extra_params(df, extra_params_res)[0]

    one_hot_train_len.update(custom_one_hot_len)
    to_tensor_list = one_hot_train_subs + custom_one_hots + normalized_train_subs + bool_train_subs


    df_size = df.shape[0]
    num_of_features = len(to_tensor_list)

    tensor_list = np.zeros((df_size, num_of_features)).tolist()

    for i, feat in enumerate(to_tensor_list):
        for j in range(df_size):
            tensor_list[j][i] = feat[j]

    if test_size == 0:
        X_train = None
        X_test = tensor_list
        y_train = None
        y_test = y
    else:
        X_train, X_test, y_train, y_test = train_test_split(tensor_list, y, test_size=test_size)

    
    features_name = one_hot_params + custom_one_hot_params + normalize_params + extra_params_data
    

    return X_train, X_test, y_train, y_test, features_name