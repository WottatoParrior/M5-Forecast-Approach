import math
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Normalizer, MinMaxScaler, LabelEncoder

scaler = Normalizer()


calendarDTypes = {
    "event_name_1": "category",
    "event_name_2": "category",
    "event_type_1": "category",
    "event_type_2": "category",
    "weekday": "category",
    "wm_yr_wk": "int16",
    "wday": "int16",
    "month": "int16",
    "year": "int16",
    "snap_CA": "float32",
}

priceDTypes = {
    "store_id": "category",
    "item_id": "category",
    "wm_yr_wk": "int16",
    "sell_price": "float32",
}

calendar = pd.read_csv(r"C:\Users\alexd\PycharmProjects\m5-accuracy\input\calendar.csv", dtype=calendarDTypes)
prices = pd.read_csv(r"C:\Users\alexd\PycharmProjects\m5-accuracy\input\sell_prices.csv", dtype=priceDTypes)
sales  = pd.read_csv(r"C:\Users\alexd\PycharmProjects\m5-accuracy\input\sales_train_validation.csv", index_col=0)

cols = sales.columns.tolist()
cols = [cols[0]] + cols[-2:] + cols[1:-2]
sales = sales[cols]
cols.iloc[:, :1] + sales.iloc[:, -2:]
sales['store_id'] = sales.store_id.apply(lambda x: re.sub(r"[][']", "", x))




plt.figure(figsize=(8.5, 9))
plt.hist(item_prices, bins=np.arange(math.ceil(max(item_prices)) + 1))
plt.xticks(ticks=np.arange(0, math.ceil(max(item_prices)) + 1))

plt.xlabel("Price")
plt.ylabel("Count")
plt.title("Price Distribution of All Items")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


wday_sales = sales_cal.groupby('wday')['sales'].mean()
plt.figure(figsize=(10, 5))
plt.bar(sales_cal[:7]['weekday'], wday_sales)
plt.ylabel("Average unit sales")
plt.title("Average Unit Sales per Day of Week")

plt.show()




#sales per month
plt.figure(figsize=(17, 8))
monthly_sales = sales_cal.groupby(['year', 'month'])['sales'].sum().reset_index()
monthly_sales['date'] = monthly_sales.apply(lambda x: datetime.datetime(x['year'], x['month'], 1), axis=1)

plt.plot('date', 'sales', data=monthly_sales)
plt.xlabel("Date")
plt.ylabel("Unit sales")
plt.title("Number of Unit Sales per Month")

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
plt.setp(ax.xaxis.get_minorticklabels(), rotation=90)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)

plt.show()

#avg montly sales

month_sales = monthly_sales[monthly_sales['year'] != 2016].groupby('month')['sales'].mean()
plt.figure(figsize=(10, 5))
plt.bar(month_sales.index, month_sales)
plt.ylabel("Average unit sales")
plt.xlabel("Month of year")

plt.title("Average Unit Sales per Month of Year")

plt.show()

#avg sales per week
wday_sales = sales.groupby('weekday')['sales'].mean()
plt.figure(figsize=(10, 5))
plt.bar(sales[:7]['weekday'], wday_sales)
plt.ylabel("Average unit sales")
plt.title("Average Unit Sales per Day of Week")

plt.show()



# Sales
total_sales = np.sum(sales.iloc[:, 3:], axis=0)
plt.figure(figsize=(20, 25))
f, (ax1, ax2) = plt.subplots(2,1)
window_size_7 = 7
window_size_14 =14

ax1.plot(np.arange(len(total_sales)), total_sales.rolling(window_size_7).mean())
ax2.plot(np.arange(len(total_sales)), total_sales.rolling(window_size_14).mean())

f.suptitle("Moving average 7 and 14 days")
plt.xlabel("Day number")
plt.ylabel("Unit sales")
plt.savefig('sales_rolling_avg.pdf')
plt.show()






num_days = sales.shape[1] - 3
week_ids = calendar['wm_yr_wk'][:num_days].unique()
num_weeks = len(week_ids)
#

weekly_sales = sales.iloc[:, :3].copy()
for mult, wid in enumerate(week_ids):
    start_idx = 6 + mult * 7
    weekly_sales[wid] = np.sum(sales.iloc[:, start_idx: start_idx + 7], axis=1)
weekly_sales = weekly_sales.groupby(
    ['store_id', 'item_id', 'id']).first()
weekly_sales_scaled = pd.DataFrame(scaler.fit_transform(weekly_sales),
                                    columns=weekly_sales.columns,
                                    index=weekly_sales.index)
weekly_sales




weekly_prices = prices.merge(
    sales.iloc[:, :3], how='left', on=['store_id', 'item_id']).\
    groupby(['store_id', 'item_id', 'id','wm_yr_wk'])['sell_price'].first().unstack('wm_yr_wk').\
         iloc[:, :weekly_sales.shape[1]]
weekly_prices.replace(np.nan, 0, inplace=True)
weekly_prices_scaled = pd.DataFrame(scaler.fit_transform(weekly_prices),
                                    columns=weekly_prices.columns,
                                    index=weekly_prices.index)
weekly_prices



weekly_revenue = weekly_sales * weekly_prices
weekly_revenue




total_rev = np.sum(weekly_revenue, axis=0)
total_sales = np.sum(weekly_sales, axis=0)
avg_prices = np.mean(weekly_prices, axis=0)

window_size = 4
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8.5, 11))
axs[0].plot(np.arange(num_weeks), total_rev.rolling(window_size).mean())
axs[1].plot(np.arange(num_weeks), total_sales.rolling(window_size).mean())
axs[2].plot(np.arange(num_weeks), avg_prices.rolling(window_size).mean())

axs[0].set_title(f"{window_size} Week Moving-Average of Revenue Earned")
axs[1].set_title(f"{window_size} Week Moving-Average of Unit Sales")
axs[2].set_title(f"{window_size} Week Moving-Average of Average Price of All Items")

axs[0].set_ylabel("Revenue")
axs[1].set_ylabel("Number of unit sales")
axs[2].set_ylabel("Average price")

plt.xlabel("Week number")


plt.tight_layout()
plt.show()





daily_sales = np.sum(sales.iloc[:, 6:], axis=0)
sales_cal = calendar[:daily_sales.shape[0]].copy()
sales_cal['sales'] = daily_sales.values
sales_cal




year = 2011
years = sales_cal['year'].unique()
fig, axs = plt.subplots(6, 1, figsize=(20, 30))
axs = axs.flatten()
for i, year in enumerate(years):
    year_sales = sales_cal[sales_cal['year'] == year]
    minv, maxv = min(sales_cal['sales']), max(sales_cal['sales'])
    evnts = year_sales['event_name_1'][~year_sales['event_name_1'].isnull()]
    axs[i].plot(year_sales['sales'])
    axs[i].set_xticks(ticks=year_sales[year_sales['wday'] == 1].index)
    axs[i].set_title(year)
    axs[i].tick_params(labelrotation=90)
    axs[i].vlines(evnts.index, minv, maxv, linestyles='--', color='b', alpha = 0.5)
    for ie, (idx, evnt) in enumerate(evnts.items()):
        text_y = minv if ie % 2 else maxv
        text_align = 'bottom' if ie % 2 else 'top'
        axs[i].text(idx, text_y, evnt, rotation=90, va=text_align)
        axs[i].set_xlabel("Day number")
        axs[i].set_ylabel("Unit sales")

fig.suptitle("Number of Unit Sales per Day for Each Year")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('units_sales_per_day_per_year.pdf')
plt.show()

year = 2016
years = sales_cal['year'].unique()
plt.figure(figsize=(17, 8))
year_sales = sales_cal[sales_cal['year'] == year]
minv, maxv = min(year_sales['sales']), max(year_sales['sales'])
evnts = year_sales[~year_sales['event_name_1'].isnull()][['event_name_1',
                                                          'event_type_1']]
colors = {'Cultural': 'royalblue', 'National': 'purple',
          'Religious': 'darkgreen', 'Sporting': 'black'}
plt.plot(year_sales['sales'])
plt.xticks(ticks=year_sales[year_sales['wday'] == 1].index, rotation=90)
plt.title(year)
for etype, ecolor in colors.items():
    evnt = evnts[evnts['event_type_1'] == etype]
    plt.vlines(evnt.index, minv, maxv, linestyles='--',
                color=ecolor, alpha = 0.5, label=etype)
for ie, (idx, evnt, etype) in enumerate(evnts.itertuples()):
    text_y = minv if ie % 2 else maxv
    text_align = 'bottom' if ie % 2 else 'top'
    plt.text(idx + 1, text_y, evnt, rotation=90,
             va=text_align, color=colors[etype])
    plt.xlabel("Day number")
    plt.ylabel("Unit sales")

leg = plt.legend(prop={'size': 10}, bbox_to_anchor=(0.092, 0.158))
for lh in leg.legendHandles:
    lh.set_alpha(0.8)
plt.title("Number of Unit Sales per Day for 2016")
plt.savefig("Number of Unit Sales per Day for 2016.pdf")
plt.show()










event_types_1 = calendar.groupby('event_type_1')['d'].nunique()
event_types_2 = calendar.groupby('event_type_2')['d'].nunique()
event_types = event_types_1.add(event_types_2, fill_value=0)

plt.figure(figsize=(7, 5))
plt.bar(event_types.index, event_types)
plt.ylabel("Count")
plt.title("Number of Events in each Category")
plt.savefig('Number of Events in each Category.pdf')
plt.show()



wday_sales = sales_cal.groupby('wday')['sales'].mean()
plt.figure(figsize=(10, 5))
plt.bar(sales_cal[:7]['weekday'], wday_sales)
plt.ylabel("Average unit sales")
plt.title("Average Unit Sales per Day of Week")
plt.savefig('Average Unit Sales per Day of Week.pdf')
plt.show()


month_sales = monthly_sales[monthly_sales['year'] != 2016].groupby('month')['sales'].mean()
plt.figure(figsize=(10, 5))
plt.bar(month_sales.index, month_sales)
plt.ylabel("Average unit sales")
plt.xlabel("Month of year")

plt.title("Average Unit Sales per Month of Year")
plt.savefig("Average Unit Sales per Month of Year.pdf")

plt.show()



total_sales = np.sum(sales.iloc[:, 6:], axis=0)
plt.figure(figsize=(20, 10))
window_size = 30
plt.plot(np.arange(len(total_sales)), total_sales.rolling(window_size).mean())
plt.title(f"{window_size} Day Moving-Average of Unit Sales.")
plt.xlabel("Day number")
plt.ylabel("Unit sales")
plt.show()












g_tst = pd.concat([lgbm_tst['x'][['store_id', 'item_id']], lgbm_tst['y'],
                   pd.Series(lgbm_tst['yhat'].reshape(-1),
                             name='preds',
                             index=lgbm_tst['y'].index)],
                  axis=1).join(lgbm_data['id']).drop(columns=['store_id', 'item_id'])
g_tst = g_tst.groupby(['id'])
errors = g_tst.apply(lambda r: rmse(r['sales'], r['preds'])).sort_values(ascending=False)
top_50 = errors[:50]
items = [x[:-11] for x in top_50.index]

plt.figure(figsize=(15, 7))
plt.bar(items, top_50.values)
plt.xticks(rotation=90)
plt.title('Top 50 items with the highest RMSE values.')
plt.ylabel('RMSE')
plt.show()