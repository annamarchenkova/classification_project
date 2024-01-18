import logging
import os
import pickle
import joblib
from datetime import datetime as dt
from datetime import timedelta
from typing import  List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
from catboost import  CatBoostClassifier, EFeaturesSelectionAlgorithm, EShapCalcType, Pool
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor

from project_dirs import PROJECT_DIR, DATA_DIR, OUTPUT_DIR, MODELS_DIR, FIGURES_DIR

##########################################################################################
# ---------------------------------  DS PREPROCESSING  ---------------------------------- #

def load_config(cnf_dir=PROJECT_DIR, cnf_name='config.yml'):
    """_summary_

    Args:
        cnf_dir (_type_, optional): _description_. Defaults to PROJECT_DIR.
        cnf_name (str, optional): _description_. Defaults to 'config.yml'.

    Returns:
        _type_: _description_
    """
    config_file = open(os.path.join(cnf_dir, cnf_name))
    return yaml.load(config_file, yaml.FullLoader)

def get_cols_too_similar(data, threshold=0.95):
    """
    Find features with too many similar values.
    :return: the pandas dataframe of sought features with the fraction of values which are similar, 
             as well as a list containing the most present value.
    
    :data: (pd.DataFrame) dataset
    :threshold: (float, default=0.95) fraction of similar values, must be a number in [0,1] interval
    """
    
    L = len(data)
    
    cols_counts = list()

    for col in data.columns:
        try:
            unique_values, unique_counts = np.unique(data[col].values, return_counts=True)
        except TypeError:
            unique_values, unique_counts = np.unique(data[col].astype(str).values, return_counts=True)

        idx_max = np.argmax(unique_counts)
        cols_counts.append((col, unique_values[idx_max], unique_counts[idx_max]))
    
    colname_and_values = map(lambda x: (x[0], x[2]), cols_counts)
    most_present_value = map(lambda x: x[1], cols_counts)

    df_similar_values = pd.DataFrame(colname_and_values)\
        .rename(columns={0: 'col_name', 1: 'frac'})\
        .sort_values('frac', ascending=False)

    df_similar_values['frac'] = df_similar_values['frac'].apply(lambda x: x / L)
    df_similar_values.query('frac >= @threshold', inplace=True)
    
    return df_similar_values, list(most_present_value)


def fill_nan_categorical_w_value(df, fill_with='missing'):
    # Fill NaNs in categorical columns with mode
    nan_cols_cat = df.isna().sum()[(df.isna().sum() > 0) & (df.dtypes == 'object')].index.values

    for column in nan_cols_cat:
        df[column] = df[column].fillna(fill_with)
        
    return df


def fill_nan_numerical_w_median(df: pd.DataFrame) -> pd.DataFrame:
    # Fill NaNs in numerical columns with median
    nan_cols_num = df.isna().sum()[
        (
            (df.isna().sum() > 0) & 
            (df.dtypes != 'object') & 
            (df.dtypes != 'datetime64[ns]')
            )
        ].index.values

    for column in nan_cols_num:
        col_median = df[column].median()
        df[column] = df[column].fillna(col_median)
        
    return df


def fill_nan_categorical_w_mode(df: pd.DataFrame) -> pd.DataFrame:
    # Fill NaNs in categorical columns with mode'
    nan_cols_cat = df.isna().sum()[(df.isna().sum() > 0) & (df.dtypes == 'object')].index.values
    
    for column in nan_cols_cat:
        col_mode = df[column].mode()[0]
        df[column] = df[column].fillna(col_mode)
    
    return df


def get_non_collinear_features_from_vif(data, vif_threshold=5, idx=0):
    """
    Find features whose variance inflation factor (VIF) exceeds the desired threshold and eliminate them.
    :return: list of feature names without the features whose VIF exceeds the threshold.
    
    :data: (pd.DataFrame) dataset
    :vif_threshold: (int, default=5) VIF threshold
    :idx: (int, default=0) DO NOT TOUCH
    """

    num_features = [i[0] for i in data.dtypes.items() if i[1] in ['float64', 'float32', 'int64', 'int32']]
    df = data[num_features].copy()
    
    if idx >= len(num_features):
        return df.columns.to_list()

    else:
        print('\rProcessing feature {}/{}'.format(idx+1, len(num_features)), end='')
        vif_ = variance_inflation_factor(df, idx)

        if vif_ > vif_threshold:
            df.drop(num_features[idx], axis=1, inplace=True)
            return get_non_collinear_features_from_vif(df, idx=idx, vif_threshold=vif_threshold)

        else:
            return get_non_collinear_features_from_vif(df, idx=idx+1, vif_threshold=vif_threshold)

def find_cols_w_2many_nan(
        data: pd.DataFrame,
        *,
        thr:float=0.95, 
        f_display:bool=False) -> Tuple[List[str], Optional[pd.DataFrame]]:
    
    na_cols = data.columns[data.isna().any()]

    df_nans = data[na_cols].copy() \
                .isna().sum() \
                .apply(lambda x: x / data.shape[0]) \
                .reset_index().rename(columns={0: 'f_nans', 'index': 'feature_name'}) \
                .sort_values(by='f_nans', ascending=False)
    
    cols_2_many_nans = df_nans.loc[df_nans.f_nans >= thr, 'feature_name'].to_list()

    if f_display:
        disp_df = df_nans.style.background_gradient(axis=0, gmap=df_nans['f_nans'], cmap='Oranges')
        return cols_2_many_nans, disp_df
    else:
        return cols_2_many_nans
    
    
def find_cols_w_single_value(data: pd.DataFrame) -> List[str]:
    return list(
        col for col, n_unique in data.nunique().items() if n_unique==1
    )


def drop_unnecessary_cols(df, cnf):

    df.drop(cnf['columns_to_drop'], axis=1, inplace=True, errors='ignore')

    nan_list = find_cols_w_2many_nan(df, thr=cnf['nan_value_threshold'])
    logging.info(f"Dropped cols with NaN % > {cnf['nan_value_threshold']*100}: {[i for i in nan_list]}")
    # print([i for i in df.columns if i in ['ACCORDATO_VL_CQS_ATTUALE', 'NUM Rate mancanti MUTUO']])
    df.drop(nan_list, axis=1, inplace=True)
    
    single_value_list = find_cols_w_single_value(df)
    logging.info(f"Dropped single value cols: {[i for i in single_value_list]}")
    # print([i for i in df.columns if i in ['ACCORDATO_VL_CQS_ATTUALE', 'NUM Rate mancanti MUTUO']])
    df.drop(single_value_list, axis=1, inplace=True)
    
    df_sim_vals, most_present_value = get_cols_too_similar(df, cnf['value_similarity_threshold'])
    cols_2similar = df_sim_vals.col_name
    logging.info(f"Dropped too similar cols: {[i for i in cols_2similar]}; thr: {cnf['value_similarity_threshold']}")
    # print([i for i in df.columns if i in ['ACCORDATO_VL_CQS_ATTUALE', 'NUM Rate mancanti MUTUO']])
    df.drop(cols_2similar, axis=1, inplace=True)

    return df


def preprocessing(
    df,
    cnf,
    drop_collinear_fs=True,
    datetime_cols=None,
    save_pickle=True,
    save_csv=True,
    ):
    '''preprocessing of the dataset'''

    if datetime_cols:
        df = date_cols_to_datetime(df, cols=datetime_cols)

    # Add cols with too similar values to columns_to_drop list
    df_sim_vals, most_present_val = get_cols_too_similar(df, cnf['value_similarity_threshold'])
    _ = [cnf['columns_to_drop'].append(i) for i in df_sim_vals['col_name']]

    df = drop_unnecessary_cols(df, cnf)

    # Fill Nan Values
    logging.info('Filling Nan values')
    df = fill_nan_categorical_w_mode(df)
    df = fill_nan_numerical_w_median(df)

    if drop_collinear_fs:
        df = drop_collinear_features(df, cnf)
        ds_name_prefix = '_collin_fs_dropped'
    else:
        ds_name_prefix = ''

    if len(cnf['unfrequent_cat_cols']['col_names']) > 0:
        df = group_unfrequent_vals_cat_columns(df, cnf)

    if save_pickle:
        full_path = os.path.join(DATA_DIR, f'preprocessed{ds_name_prefix}.pkl')
        df.to_pickle(full_path)
        logging.info(f"Saved preprocessed dataset to {full_path}")
    if save_csv:
        full_path = os.path.join(DATA_DIR, f'preprocessed{ds_name_prefix}.csv')
        df.to_csv(full_path)
        logging.info(f"Saved preprocessed dataset to {full_path}")
    
    return df

def group_unfrequent_vals_cat_columns(df, cnf):

    assert len(cnf['unfrequent_cat_cols']['col_names']) == len(cnf['unfrequent_cat_cols']['thresholds'])

    id_col = cnf['unfrequent_cat_cols']['id_col']
    columns = cnf['unfrequent_cat_cols']['col_names']
    thresholds = cnf['unfrequent_cat_cols']['thresholds']
    grouped_val_name = cnf['unfrequent_cat_cols']['grouped_val_name']

    for col, thresh in zip(columns, thresholds):
        df = group_unfrequent_vals_cat_col(
            df,
            id_col=id_col,
            cat_col=col,
            count_thresh=thresh,
            grouped_val_name=grouped_val_name
            )
    return df


def group_unfrequent_vals_cat_col(
    df,
    id_col,
    cat_col,
    count_thresh,
    grouped_val_name='altro'):
    '''
    Group non frequent values (with count < `count_thresh`) for specified 
    categorical column `cat_col` in the dataset `df`. `id_col`
    '''

    count_df = df[[cat_col, id_col]].drop_duplicates().copy()
    count_df = count_df.groupby(cat_col)[[id_col]]\
        .count()\
            .rename(columns={id_col:'count'})\
                .reset_index()\
                    .sort_values(by='count')

    vals_to_group = count_df[count_df['count'] < count_thresh][cat_col].values
    df.loc[df[cat_col].isin(vals_to_group), cat_col] = grouped_val_name
    return df


def date_cols_to_datetime(df, cols):
    '''Convert selected columns to datetime type'''
    df[cols] = df[cols].apply(lambda x: pd.to_datetime(x, infer_datetime_format=True))
    logging.info(f"Converted columns {cols} to datetime")
    return df


def drop_collinear_features(df, cnf):
    try:
        logging.info(f"Trying to load a list of collinear features {cnf['collinear_fs_file']}")
        print(f"Trying to load a list of collinear features {cnf['collinear_fs_file']}")
        collinear_fs = pd.read_csv(os.path.join(OUTPUT_DIR, cnf['collinear_fs_file']))['collinear features'].values

    except FileNotFoundError:
        logging.info(f"File {cnf['collinear_fs_file']} not found, recalculating collinear features]")
        print(f"File {cnf['collinear_fs_file']} not found, recalculating collinear features]")

        logging.info('Elimination of collinear features. This step can take a few minutes')
        non_collinear_fs = get_non_collinear_features_from_vif(df, vif_threshold=cnf['vif_threshold'])
        collinear_fs = set(c[0] for c in df.dtypes.items() if c[1]!='O') - set(non_collinear_fs)
    
        path = os.path.join(OUTPUT_DIR, f"collinear_fs_vif_{cnf['vif_threshold']}.csv")
        pd.DataFrame(collinear_fs, columns=['collinear features']).to_csv(path, index=False)
        logging.info(f"Saved a list of eliminated collinear features to {path}")
    
    df = df.drop(collinear_fs, axis=1, errors='ignore')
    return df
