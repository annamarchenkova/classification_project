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


def load_config(cnf_dir: str = PROJECT_DIR, cnf_name: str = 'config.yml') -> dict:
    """
    Load configuration file from specified directory and file name.

    Args:
        cnf_dir (str, optional): Directory path where configuration file is located. Defaults to PROJECT_DIR.
        cnf_name (str, optional): Name of configuration file. Defaults to 'config.yml'.

    Returns:
        dict: Dictionary containing configuration data.
    """
    config_file = open(os.path.join(cnf_dir, cnf_name))
    return yaml.load(config_file, yaml.FullLoader)

def get_cols_too_similar(data: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, List]:
    """
    Find features with too many similar values.

    Args:
        data: (pd.DataFrame) dataset
        threshold: (float, default=0.95) fraction of similar values, must be a number in [0,1] interval

    Returns:
        Tuple[pd.DataFrame, List]: A tuple containing the pandas dataframe of sought features with the fraction of values
        which are similar, as well as a list containing the most present value.

    This function takes a pandas dataframe and a threshold value as input and returns a tuple containing the pandas
    dataframe of sought features with the fraction of values which are similar, as well as a list containing the most
    present value. The function finds features with too many similar values by calculating the fraction of similar
    values for each feature and returning the features with a fraction greater than or equal to the threshold value.
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
    """
    Fill NaNs in categorical columns with a specified value.

    Args:
        df (pandas.DataFrame): The DataFrame to fill NaNs in.
        fill_with (str): The value to fill NaNs with. Default is 'missing'.

    Returns:
        pandas.DataFrame: The DataFrame with NaNs in categorical columns filled with the specified value.
    """
    nan_cols_cat = df.isna().sum()[(df.isna().sum() > 0) & (df.dtypes == 'object')].index.values
    for column in nan_cols_cat:
        df[column] = df[column].fillna(fill_with)
    return df

def fill_nan_numerical_w_median(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaNs in numerical columns with median.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with NaNs in numerical columns filled with median.
    """
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
    """
    Fill NaNs in categorical columns with mode.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with NaNs in categorical columns filled with mode.
    """
    nan_cols_cat = df.isna().sum()[(df.isna().sum() > 0) & (df.dtypes == 'object')].index.values
    for column in nan_cols_cat:
        col_mode = df[column].mode()[0]
        df[column] = df[column].fillna(col_mode)
    return df

def get_non_collinear_features_from_vif(data: pd.DataFrame, vif_threshold: int = 5, idx: int = 0) -> list:
    """
    Find features whose variance inflation factor (VIF) exceeds the desired threshold and eliminate them.

    :param data: (pd.DataFrame) dataset
    :param vif_threshold: (int, default=5) VIF threshold
    :param idx: (int, default=0) DO NOT TOUCH
    :return: list of feature names without the features whose VIF exceeds the threshold.
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

def find_cols_w_2many_nan(data: pd.DataFrame, *, thr:float=0.95, f_display:bool=False) -> Tuple[List[str], Optional[pd.DataFrame]]:
    """
    Find columns in a pandas DataFrame with too many NaN values.

    Args:
        data (pd.DataFrame): The input DataFrame.
        thr (float, optional): The threshold for the percentage of NaN values in a column. Defaults to 0.95.
        f_display (bool, optional): Whether to display a styled DataFrame with the percentage of NaN values for each column. Defaults to False.

    Returns:
        Tuple[List[str], Optional[pd.DataFrame]]: A tuple containing a list of column names with too many NaN values and an optional styled DataFrame with the percentage of NaN values for each column.
    """
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
    """
    Return a list of column names in the given DataFrame that have only a single unique value.

    Args:
        data (pd.DataFrame): The DataFrame to search for columns with a single unique value.

    Returns:
        List[str]: A list of column names that have only a single unique value.
    """
    return list(
        col for col, n_unique in data.nunique().items() if n_unique==1
    )

def drop_unnecessary_cols(df, cnf):
    """
    Drop unnecessary columns from a pandas DataFrame based on given configuration.

    Args:
        df (pandas.DataFrame): The DataFrame to drop columns from.
        cnf (dict): A dictionary containing configuration parameters.
            'columns_to_drop' (list): A list of column names to drop.
            'nan_value_threshold' (float): A threshold for dropping columns with too many NaN values.
            'value_similarity_threshold' (float): A threshold for dropping columns that are too similar.

    Returns:
        pandas.DataFrame: The modified DataFrame with unnecessary columns dropped.
    """
    df.drop(cnf['columns_to_drop'], axis=1, inplace=True, errors='ignore')
    nan_list = find_cols_w_2many_nan(df, thr=cnf['nan_value_threshold'])
    logging.info(f"Dropped cols with NaN % > {cnf['nan_value_threshold']*100}: {[i for i in nan_list]}")
    df.drop(nan_list, axis=1, inplace=True)
    single_value_list = find_cols_w_single_value(df)
    logging.info(f"Dropped single value cols: {[i for i in single_value_list]}")
    df.drop(single_value_list, axis=1, inplace=True)
    df_sim_vals, most_present_value = get_cols_too_similar(df, cnf['value_similarity_threshold'])
    cols_2similar = df_sim_vals.col_name
    logging.info(f"Dropped too similar cols: {[i for i in cols_2similar]}; thr: {cnf['value_similarity_threshold']}")
    df.drop(cols_2similar, axis=1, inplace=True)
    return df

def preprocessing(df, cnf, drop_collinear_fs=True, datetime_cols=None, save_pickle=True, save_csv=True):
    '''
    Preprocesses the dataset by performing the following steps:
    
    1. Converts specified columns to datetime format.
    2. Identifies columns with too similar values and adds them to the list of columns to drop.
    3. Drops unnecessary columns.
    4. Fills NaN values in categorical columns with mode and numerical columns with median.
    5. Drops collinear features if specified.
    6. Groups infrequent values in categorical columns.
    7. Saves preprocessed dataset as a pickle and/or csv file.
    
    Args:
        df (pandas.DataFrame): The dataset to be preprocessed.
        cnf (dict): A dictionary containing configuration parameters.
        drop_collinear_fs (bool, optional): Whether or not to drop collinear features. Defaults to True.
        datetime_cols (list, optional): A list of columns to convert to datetime format. Defaults to None.
        save_pickle (bool, optional): Whether or not to save preprocessed dataset as a pickle file. Defaults to True.
        save_csv (bool, optional): Whether or not to save preprocessed dataset as a csv file. Defaults to True.
        
    Returns:
        pandas.DataFrame: The preprocessed dataset.
    '''
    if datetime_cols:
        df = date_cols_to_datetime(df, cols=datetime_cols)
    
    # Add cols with too similar values to columns_to_drop list
    df_sim_vals, most_present_val = get_cols_too_similar(df, cnf['value_similarity_threshold'])
    _ = [cnf['columns_to_drop'].append(i) for i in df_sim_vals['col_name']]
    df = drop_unnecessary_cols(df, cnf)
    
    # Fill NaN Values
    logging.info('Filling NaN values')
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
    """
    Group infrequent categorical values in specified columns of a pandas DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to be processed.
        cnf (dict): A dictionary containing configuration parameters.
            'unfrequent_cat_cols' (dict): A dictionary containing the following keys:
                'id_col' (str): The name of the column containing the unique identifier.
                'col_names' (list): A list of column names to be processed.
                'thresholds' (list): A list of integer thresholds for each column in 'col_names'.
                'grouped_val_name' (str): The name of the new value to be assigned to infrequent values.

    Returns:
        pandas.DataFrame: The processed DataFrame with infrequent categorical values grouped.
    """
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

def group_unfrequent_vals_cat_col(df, id_col, cat_col, count_thresh, grouped_val_name='altro'):
    '''
    Group non frequent values (with count < `count_thresh`) for specified categorical column `cat_col` in the dataset `df`.
    
    Args:
        df (pandas.DataFrame): The dataset to be processed.
        id_col (str): The name of the column containing the unique identifier for each row.
        cat_col (str): The name of the categorical column to be processed.
        count_thresh (int): The threshold count value for grouping non-frequent values.
        grouped_val_name (str, optional): The name to be assigned to the grouped values. Defaults to 'altro'.
    
    Returns:
        pandas.DataFrame: The processed dataset with non-frequent values in `cat_col` grouped and assigned `grouped_val_name`.
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

def date_cols_to_datetime(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Convert selected columns to datetime type.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cols (List[str]): A list of column names to convert to datetime.

    Returns:
        pd.DataFrame: The input DataFrame with the selected columns converted to datetime.
    """
    df[cols] = df[cols].apply(lambda x: pd.to_datetime(x, infer_datetime_format=True))
    logging.info(f"Converted columns {cols} to datetime")
    return df

def drop_collinear_features(df: pd.DataFrame, cnf: dict) -> pd.DataFrame:
    """
    Drop collinear features from a pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cnf (dict): A dictionary containing configuration parameters.

    Returns:
        pd.DataFrame: The input DataFrame with collinear features dropped.

    Raises:
        FileNotFoundError: If the collinear features file specified in cnf is not found.

    """
    try:
        logging.info(f"Trying to load a list of collinear features {cnf['collinear_fs_file']}")
        print(f"Trying to load a list of collinear features {cnf['collinear_fs_file']}")
        collinear_fs = pd.read_csv(os.path.join(OUTPUT_DIR, cnf['collinear_fs_file']))['collinear features'].values
    except FileNotFoundError:
        logging.info(f"File {cnf['collinear_fs_file']} not found, recalculating collinear features")
        print(f"File {cnf['collinear_fs_file']} not found, recalculating collinear features")
        logging.info('Elimination of collinear features. This step can take a few minutes')
        non_collinear_fs = get_non_collinear_features_from_vif(df, vif_threshold=cnf['vif_threshold'])
        collinear_fs = set(c[0] for c in df.dtypes.items() if c[1]!='O') - set(non_collinear_fs)
        path = os.path.join(OUTPUT_DIR, f"collinear_fs_vif_{cnf['vif_threshold']}.csv")
        pd.DataFrame(collinear_fs, columns=['collinear features']).to_csv(path, index=False)
        logging.info(f"Saved a list of eliminated collinear features to {path}")
        df = df.drop(collinear_fs, axis=1, errors='ignore')
    return df

##########################################################################################
# --------------------------------  FEATURE SELECTION  --------------------------------- #


def feature_selection_shap(X_train, y_train, X_test, y_test, cnf, features, cat_features):
    """
    Select the most relevant features using the SHAP method.

    Args:
        X_train (array-like): The training input samples.
        y_train (array-like): The target values for the training input samples.
        X_test (array-like): The testing input samples.
        y_test (array-like): The target values for the testing input samples.
        cnf (dict): A dictionary containing the configuration parameters.
        features (list): A list of feature names.
        cat_features (list): A list of categorical feature names.

    Returns:
        tuple: A tuple containing the selected feature names and the trained model.
    """
    train_pool = Pool(
        X_train, y_train,
        feature_names=features,
        cat_features=cat_features
    )
    test_pool = Pool(
        X_test, y_test,
        feature_names=features,
        cat_features=cat_features
    )

    feat_idx_list = np.arange(train_pool.num_col())

    summary, model = select_features_with_shap(
        cnf['n_features_2select'],
        train_pool,
        test_pool,
        algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
        steps=cnf['steps'],
        iterations=cnf['iterations'],
        feat_for_select=feat_idx_list
    )

    sel_features = summary.get('selected_features_names')

    return sel_features, model

def select_features_with_shap(n, train_pool, test_pool, algorithm, iterations=250, feat_for_select=None, flg_final_model=True, steps=3, **kwargs):
    """
    Perform recursive feature selection with CatBoost using SHAP values.

    :param n: (int) number of features to select
    :param train_pool: (catboost.Pool) train data as catboost Pool
    :param test_pool: (catboost.Pool) validation data as catboost Pool
    :param algorithm: (catboost.EFeaturesSelectionAlgorithm) which algorithm to use for recursive feature selection
    :param iterations: (int, default=250) number of iterations to perform
    :param feat_for_select: (list, default=None) index of features to be considered in feature selection,
    :param flg_final_model: (bool, default=True) whether to fit the final model (i.e. with only selected feautures) or not
        --> note that if `True` the final model is returned as well
    :param steps: (int, default=3) number of steps
    :param kwargs: additional arguments for the `.select_features()` method

    :return: the summary of the search as a Dict, and possibly a model trained on the dataset with the selected
             features only.
    """
    if feat_for_select is None:
        feat_for_select = list()

    print('Algorithm:', algorithm)

    model = CatBoostClassifier(
        iterations=iterations,
        loss_function='Logloss',
        random_seed=42,
    )

    summary = model.select_features(
        train_pool,
        eval_set=test_pool,
        features_for_select=feat_for_select,
        num_features_to_select=n,
        steps=steps,
        algorithm=algorithm,
        shap_calc_type=EShapCalcType.Regular,
        train_final_model=flg_final_model,
        logging_level='Silent',
        plot=False,
        **kwargs
    )

    res = (summary, model) if flg_final_model else summary

    return res

def f_selection_catboost_shap(df, y, test_size=0.2, shuffle=True, n_features=30, steps=5, iterations=500, random_state=42):
    """
    Selects the most important features using the CatBoost algorithm and SHAP values.

    Args:
        df (pandas.DataFrame): The input dataframe.
        y (pandas.Series): The target variable.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        shuffle (bool, optional): Whether or not to shuffle the data before splitting. Defaults to True.
        n_features (int, optional): The number of features to select. Defaults to 30.
        steps (int, optional): The number of steps to take in the recursive feature selection process. Defaults to 5.
        iterations (int, optional): The number of iterations to run the CatBoost algorithm. Defaults to 500.
        random_state (int, optional): The random state to use for reproducibility. Defaults to 42.

    Returns:
        tuple: A tuple containing the summary of the selected features and the trained CatBoost model.
    """
    features = df.columns.to_list()
    cat_features = [f for f, dtype in df.dtypes.items() if dtype=='O']
    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size=test_size,
        stratify=y,
        shuffle=shuffle,
        random_state=random_state
    )
    train_pool = Pool(
        X_train, y_train,
        feature_names=features,
        cat_features=cat_features
    )
    test_pool = Pool(
        X_test, y_test,
        feature_names=features,
        cat_features=cat_features
    )
    feat_idx_list = np.arange(train_pool.num_col())
    summary, model = select_features_with_shap(
        n_features,
        train_pool, test_pool,
        algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
        steps=steps, iterations=iterations,
        feat_for_select=feat_idx_list
    )
    return summary, model

##########################################################################################
# ------------------------------------  PLOTTING  ------------------------------------- #
def save_shap_plot(X, shap_values, fig_path=FIGURES_DIR, prefix=''):
    """save_shap_plot

    Args:
        X (_type_): _description_
        shap_values (_type_): _description_
        fig_path (_type_, optional): _description_. Defaults to FIGURES_DIR.
    """

    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(os.path.join(fig_path, f'shap_plot_{prefix}.png'), pad_inches=0.2, bbox_inches='tight')


def optuna_visualization_plots(study, save=True, save_dir=FIGURES_DIR):
    """_summary_

    Args:
        study (_type_): _description_
        save (bool, optional): _description_. Defaults to True.
        save_dir (_type_, optional): _description_. Defaults to FIGURES_DIR.
    """
    fig = optuna.visualization.plot_param_importances(study)
    fig1 = optuna.visualization.plot_optimization_history(study)
    plt.show(fig)
    plt.show(fig1)

    if save:
        fig.savefig(os.path.join(save_dir, 'optuna_param_importances.png'))
        fig1.savefig(os.path.join(save_dir, 'optuna_optimization_history.png'))
    

def feature_importance_plot(model, figsize=(10,15), save=True, save_dir=FIGURES_DIR, prefix=''):
    """Plot feature importance and safe the plot.

    Args:
        model (_type_): _description_
        save (bool, optional): _description_. Defaults to True.
        save_dir (_type_, optional): _description_. Defaults to FIGURES_DIR.
    """
    if prefix == '':
        prefix = str(dt.today().date())
        
    f_importance_df = pd.DataFrame(
    [model.feature_names_, model.feature_importances_],
    ['features', 'f_importance']
    ).T.sort_values(by='f_importance', ascending=True)

    if save:
        plt.figure(figsize=figsize)
        plt.barh(f_importance_df.features, f_importance_df.f_importance)
        plt.savefig(
            os.path.join(save_dir, f'feature_importance_{prefix}.png'), 
            bbox_inches='tight'
            )
        

