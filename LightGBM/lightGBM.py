import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

path = os.getcwd()

#LOAD DATA
pd.read_csv(os.path.join(path,'x_train.csv'), sep=';')
pd.read_csv(os.path.join(path,'y_train.csv'), sep=';')
pd.read_csv(os.path.join(path,'x_val.csv'), sep=';')
pd.read_csv(os.path.join(path,'y_val.csv'), sep=';')



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

params = {
    'num_leaves': 555,
    'min_child_weight': 0.034,
    'feature_fraction': 0.379,
    'bagging_fraction': 0.418,
    'min_data_in_leaf': 106,
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'objective': 'regression',
    'n_jobs': -1,
    'seed': 236,
    'learning_rate': 0.01,
    'bagging_fraction': 0.75,
    'bagging_freq': 10,
    'colsample_bytree': 0.75}


train_lgbm = lgb.Dataset(x_train[features], y_train)
val_lgbm = lgb.Dataset(x_val[features], y_val)


from sklearn import metrics
model = lgb.train(params,
                   train_lgbm,
                   num_boost_round = 100000,
                   early_stopping_rounds = 1000,
                   valid_sets = [train_lgbm, val_lgbm],
                   verbose_eval = 100)

val_pred = model.predict(x_val[features])
val_score = np.sqrt(metrics.mean_squared_error(val_pred, y_val))
print(f'Our val rmse score is {val_score}')