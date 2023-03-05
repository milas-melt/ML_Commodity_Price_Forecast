import pandas as pd
import holidays
import numpy as np
import itertools
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from statsmodels.tsa.stattools import adfuller

param_grid = {  
                'changepoint_prior_scale': [0.001, 0.05, 0.2, 0.5],
                'changepoint_range': [0.8, 0.9, 0.95],
                'seasonality_prior_scale':[0.01, 0.1, 1.0, 10.0],
                'holidays_prior_scale':[0.01, 0.1, 1.0, 10.0],
                'seasonality_mode': ['multiplicative', 'additive'],
              }

def to_string(length):
    
    return f'{length} days'

def make_monthly_data(filepath):
    # Read in the csv file
    df = pd.read_excel(filepath)
    df.dropna(inplace = True)
    df.columns = ['ds', 'y']
    
    # Set the date column as the index and convert the date column to a date format
    df.set_index(df['ds'], inplace=True)
    df.index = pd.to_datetime(df.index)
    
    # Resample the data by month and fill forward
    df_resampled = df.resample('m').ffill()
    df_resampled['ds'] = df_resampled.index

    # Calculate the last 18 months of data as the test set
    num_months = df_resampled.shape[0]
    test_start = num_months - 18
    test = df_resampled.iloc[test_start:, :]
    test.columns = ['ds', 'y_test']
    train = df_resampled.iloc[:test_start, :]

    # Return the train and test sets
    return train, test


def data_process(fileName = "./IndexPrices__US93.xlsx", cap = None, floor = None, fill = False, split_idx = 548):
    df = pd.read_excel(fileName)
    
    df.columns = ['ds', 'y']
    df.ds = pd.to_datetime(df.ds, unit='s')
    df = remove_outliers(df)
    
    df['date'] = pd.to_datetime(df['ds'])
    df.set_index("date", inplace = True)

    df = df.dropna().drop_duplicates()

    if fill:
        df = df.resample('M').ffill()

    if cap is None:
        df['cap'] = df.y.max()
    else:
        df['cap'] = cap

    if floor is None:
        df['floor'] = df.y.min()
    else:
        df['floor'] = floor

    train_set = df[ :-split_idx]
    test_set = df[-split_idx: ]

    return train_set, test_set

def make_forecast(model_obj, train, periods = 548, cap = None, floor = None):
    
    model_obj.fit(train)
    future = model_obj.make_future_dataframe(periods, freq ="D")
    forecast = model_obj.predict(future)

    return forecast

def get_future(df, periods = 548):

    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods)

    return future


def timeseries_evaluation_metrics_func(y_true, y_pred):

    print('Evaluation metric results:-')
    print(f'MSE is : {mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true,y_pred)}')
    print(f'R2 is : {r2_score(y_true, y_pred)}',end='\n')

    if mean_absolute_percentage_error(y_true,y_pred) > 0.1:
        
        print('Your model is not good enough. Please adjust it.')

def is_nfl_season(ds):
    date = pd.to_datetime(ds)
    return (date.month > 8 or date.month < 2)

def nfl_sunday(ds):
    date = pd.to_datetime(ds)
    if date.weekday() == 6 and (date.month > 8 or date.month < 2):
        return 1
    else:
        return 0

def get_holidays():
    holidays_df = pd.DataFrame([], columns = ['ds','holiday'])
    ldates = []
    lnames = []
    for date, name in sorted(holidays.US(years=np.arange(2001, 2022 + 1)).items()):
        ldates.append(date)
        lnames.append(name)
    ldates = np.array(ldates)
    lnames = np.array(lnames)
    holidays_df.loc[:,'ds'] = ldates
    holidays_df.loc[:,'holiday'] = lnames
    holidays_df.loc[:,'holiday'] = holidays_df.loc[:,'holiday'].apply(lambda x : x.replace(' (Observed)',''))

    return holidays_df

def make_verif(forecast, data_train, data_test): 

    forecast.index = pd.to_datetime(forecast.ds)    
    data_train.index = pd.to_datetime(data_train.ds)
    data_test.index = pd.to_datetime(data_test.ds)
    data = pd.concat([data_train, data_test], axis=0)
    forecast.loc[:,'y'] = data.loc[:,'y']
    
    return forecast

def make_plot_block(verif, start_date, end_date, ax=None): 
    
    df = verif.loc[start_date:end_date,:]
    df.loc[:,'yhat'].plot(lw=2, ax=ax, color='r', ls='-', label='forecasts')
    ax.fill_between(df.index, df.loc[:,'yhat_lower'], df.loc[:,'yhat_upper'], color='coral', alpha=0.3)
    df.loc[:,'y'].plot(lw=2, ax=ax, color='steelblue', ls='-', label='observations')
    ax.grid(ls=':')
    ax.legend(fontsize=15)
    [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
    [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]
    ax.set_ylabel('Price', fontsize=15)
    ax.set_xlabel('', fontsize=15)
    ax.set_title(f'{start_date} to {end_date}', fontsize=18)

def get_best_parameter(data, method = 'mae'):

    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    error = []  

    for params in all_params:
        m = Prophet(**params).fit(data)  
        df_cv = cross_validation(m, initial='1000 days', period='1000 days',\
            horizon='548 days', parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)
        error.append(df_p[method].values[0])
        

    tuning_results = pd.DataFrame(all_params)
    tuning_results[method] = error
    print(tuning_results)

    return all_params[np.argmin(method)]

def remove_outliers(df):    

    df_sub = df.y
    lim = np.abs((df_sub - df_sub.mean()) / df_sub.std(ddof=0)) < 3
    df_sub = df.where(lim, np.nan)

    return df

              
def create_param_combinations(**param_dict):
    param_iter = itertools.product(*param_dict.values())
    params =[]
    for param in param_iter:
        params.append(param) 
    params_df = pd.DataFrame(params, columns=list(param_dict.keys()))
    return params_df

def single_cv_run(history_df, metrics, param_dict, horizon = None):
    m = Prophet(**param_dict)
    m.add_country_holidays(country_name='US')
    m.fit(history_df)
    df_cv = cross_validation(m, initial = to_string(len(history_df)), \
        period = to_string(len(history_df)), horizon = horizon)
    df_p = performance_metrics(df_cv).mean().to_frame().T
    df_p['params'] = str(param_dict)
    df_p = df_p.loc[:, metrics]
    return df_p

metrics = ['horizon', 'rmse', 'mape', 'mse', 'mae', 'mdape', 'smape', 'params'] 

def get_best_para(df, method = 'mape', metric = metrics, horizon = None):
    results = []
    params_df = create_param_combinations(**param_grid)
    for param in params_df.values:
        param_dict = dict(zip(params_df.keys(), param))
        cv_df = single_cv_run(df,  metric, param_dict, horizon)
        results.append(cv_df)
    results_df = pd.concat(results).reset_index(drop=True)
    best_param = results_df.loc[results_df[method] == min(results_df[method]), ['params']]
    print(f'\n The best param combination is {best_param.values[0][0]}')

    return results_df

def add_weekname(df):
    
    df['week'] = df.ds.dt.day_name()

    return df

def Augmented_Dickey_Fuller_Test_func(series , column_name):
    print (f'Results of Dickey-Fuller Test for column: {column_name}')
    dftest = adfuller(series, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic',\
        'p-value','No Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    if dftest[1] <= 0.05:
        print("Conclusion:====>")
        print("Reject the null hypothesis")
        print("Data is stationary")
    else:
        print("Conclusion:====>")
        print("Fail to reject the null hypothesis")
        print("Data is non-stationary")

colors = {'Monday':'red', 'Tuesday':'green', 'Wednesday':'blue', 'Thursday':'yellow', \
    'Friday': 'darkorange', 'Saturday': 'darkslategrey', 'Sunday': 'purple'}

def weekly_plot(filepath = "IndexPrices__US2792.xlsx", colors = colors, title = 'US2792 Avocados'):
    train, US2792_test = data_process(filepath)
    train['week'] = train.ds.dt.day_name()
    fig, ax = plt.subplots(figsize=(18,12))
    ax.scatter(train['ds'], train['y'], s = 5, c = train['week'].map(colors))
    plt.xlabel("time")
    plt.ylabel("price")
    ax.set_title(title,fontsize=20)
    plt.show()