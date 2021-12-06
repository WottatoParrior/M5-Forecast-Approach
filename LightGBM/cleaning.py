import os
import pandas as pd


def train_test(folder, file_name):

    path = os.getcwd()
    df = pd.read_csv(os.path.join(path, folder, file_name))

    # create merge key to merge with calendar and sales
    for col in df.columns:
        if(col == 'id'):
            df['id'] = df.id.apply(lambda x: x.replace('_validation', ''))
            df = pd.melt(df,
                    id_vars='id', value_vars=df.columns[1:],
                    var_name='d', value_name='sales')
    return df


def merge(sell_prices, calendar , train , test):

    #create merge key for prices table
    sell_prices['id'] = sell_prices.item_id + '_' + sell_prices.store_id

    #merged training set
    train = train.merge(calendar, on='d', copy=False)
    train = pd.merge(train, sell_prices, on=['id', 'wm_yr_wk'])

    #merged test set
    test = test.merge(calendar, on='d', copy=False)
    test = pd.merge(test, sell_prices, on=['id', 'wm_yr_wk'])

    #drop
    train.drop(['d', 'weekday'], axis=1, inplace=True)
    test.drop(['d', 'weekday'], axis=1, inplace=True)

    train = encode_categorical(train, ["event_name_1", "event_type_1", "event_name_2", "event_type_2",
                           "item_id", "store_id"])
    test = encode_categorical(test, ["event_name_1", "event_type_1", "event_name_2", "event_type_2",
                           "item_id", "store_id"])

    return train, test


def encode_categorical(df, cols):
    for col in cols:
        # Leave NaN as it is.
        le = LabelEncoder()
        # not_null = df[col][df[col].notnull()]
        df[col] = df[col].fillna('nan')
        df[col] = pd.Series(le.fit_transform(df[col]), index=df.index)

    return df





if __name__ == '__main__':

    #read files
    folder = 'data'
    train = train_test(folder, 'sales_train_validation_afcs2021.csv')
    test = train_test(folder, 'sales_test_validation_afcs2021.csv')

    calendar = pd.read_csv(os.path.join(os.getcwd(), folder, 'calendar_afcs2021.csv'))
    sell_prices = pd.read_csv(os.path.join(os.getcwd(), folder, 'sell_prices_afcs2021.csv'))

    #join calendar and sell prices to train and test
    train, test = merge(sell_prices,calendar,train,test)

    # winning model added median of sell price have to check whether this is a usefull feature
    ########################################################################################################
    median_train = train.groupby(['id', 'month', 'item_id', 'event_type_1'])['sell_price'].median().reset_index()
    median_test = test.groupby(['id', 'month', 'item_id', 'event_type_1'])['sell_price'].median().reset_index()

    # Merge by type, store, department and month
    train = train.merge(median_train, how='outer', on=['id', 'month', 'item_id', 'event_type_1'])
    test = test.merge(median_test, how='outer', on=['id', 'month', 'item_id', 'event_type_1'])
    ########################################################################################################

    x_train = train.loc[:, train.columns != 'sales']
    y_train = train['sales']

    x_val = test.loc[:, test.columns != 'sales']
    y_val = test['sales']

    #write to csv
    x_train.to_csv('x_train.csv', sep=';')
    y_train.to_csv('y_train.csv', sep=';')
    x_val.to_csv('x_val.csv', sep=';')
    y_val.to_csv('y_val.csv', sep=';')








































