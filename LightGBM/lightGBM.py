import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

path = os.getcwd()

#LOAD DATA
x_train = pd.read_csv(os.path.join(path,'x_train.csv'), sep=';')
y_train = pd.read_csv(os.path.join(path,'y_train.csv'), sep=';')
x_val = pd.read_csv(os.path.join(path,'x_val.csv'), sep=';')
y_val = pd.read_csv(os.path.join(path,'y_val.csv'), sep=';')



features = [
 'item_id',
 'sell_price_x',
 'sell_price_y',
 'store_id',
 'wm_yr_wk',
 'wday',
 'month',
 'year',
 'event_name_1',
 'event_type_1',
 'event_name_2',
 'event_type_2',
 'snap_CA']

# params = {
#     'num_leaves': 555,
#     # 'min_child_weight': 0.034,
#     # 'feature_fraction': 0.379,
#     # 'bagging_fraction': 0.418,
#     # 'min_data_in_leaf': 106,
#     'boosting_type': 'gbdt',
#     'metric': 'rmse',
#     'force_row_wise':True,
#     'objective': 'poisson',
#     'n_jobs': -1,
#     'seed': 236,
#     'learning_rate': 0.01,
#     'bagging_fraction': 0.75,
#     'bagging_freq': 10,
#     'colsample_bytree': 0.75}

params={
    'objective':'poisson',
    'metric':['rmse'],
    'force_row_wise':True,
    'learning_rate':0.075,
    'sub_row': 0.75,
    'bagging_freq': 1,
    'lambda_12':0.1,
    'verbosity':1,
    'num_iterations':1200,
    'num_leaves':2**6 -1,
    'min_data_in_leaf':2**6 -1
}


train_lgbm = lgb.Dataset(x_train[features], y_train)
val_lgbm = lgb.Dataset(x_val[features], y_val)


from sklearn import metrics
model = lgb.train(params,
                   train_lgbm,
                   num_boost_round = 100000,
                   early_stopping_rounds = 10000,
                   valid_sets = [train_lgbm, val_lgbm],
                   verbose_eval = 100)

val_pred = model.predict(x_val[features])
val_score = np.sqrt(metrics.mean_squared_error(val_pred, y_val))
print(f'Our val rmse score is {val_score}')