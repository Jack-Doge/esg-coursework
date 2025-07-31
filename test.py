import os
import pandas as pd
import numpy as np
from loguru import logger


def preprocessing(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [  'ETFG Date', 
                    'As of Date', 
                    'Composite Ticker', 
                    'Constituent Ticker', 
                    'Constituent Name', 
                    'Weight', 
                    'Market Value', 
                    'CUSIP', 
                    'ISIN', 
                    'FIGI', 
                    'Sedol', 
                    'Country of Exchange', 
                    'Exchange', 
                    'Shares Held', 
                    'Asset Class', 
                    'Security Type', 
                    'Currency Traded'
                    ]
    df = df[df['Composite Ticker'].isin(['NURE', 'VNQ'])]
    df = df[['ETFG Date',
            'Composite Ticker', 
            'Constituent Ticker', 
            'Weight', 
            'Shares Held',
            'Market Value']]
    # drop rows with missing values 
    df.dropna(inplace=True)
    df = df[df['Constituent Ticker'] != '-']
    df = df[df['Weight'] > 0.001]
    return df


def fundflow_data(path: str) -> pd.DataFrame:
    fundflow_df = pd.read_csv(path)
    fundflow_df.columns = ['ETFG Date', 
                           'As of Date', 
                           'Composite Ticker', 
                           'Fund Shares Outstanding',
                           'NAV', 
                           'Net Daily Fund Flow']
    fundflow_df.rename(columns={'ETFG Date': 'date'}, inplace=True)
    fundflow_df = fundflow_df[fundflow_df['Composite Ticker'].isin(['NURE', 'VNQ'])]
    fundflow_df = fundflow_df[['date', 'Composite Ticker', 'NAV', 'Fund Shares Outstanding', 'Net Daily Fund Flow']]
    return fundflow_df

week_list = pd.date_range(start='2017-04-01', end='2025-01-01', freq='W-MON').strftime('%Y%m%d').tolist()
folder = 'D:\etf_global\data\\fundflow_us'
files = [files for year in os.listdir(folder) for files in os.listdir(os.path.join(folder, year))]
# files = [file for file in files if file[:8] in week_list]
logger.info(f'Number of files to process: {len(files)}')
final_df = pd.DataFrame()
for file in files:
    logger.info(f'Processing file: {file}')
    path = os.path.join(folder, file[:4], file)
    df = fundflow_data(path)
    final_df = pd.concat([final_df, df], ignore_index=True)
final_df['NAV'] = final_df['NAV'] * final_df['Fund Shares Outstanding']
final_df.reset_index(drop=True, inplace=True)
final_df.to_csv('data/fundflow_data.csv', index=False)

def add_week_label():
    df = pd.read_csv('data/fundflow_data.csv')
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    week_list = pd.date_range(start='2017-04-01', end='2025-01-01', freq='W-MON').strftime('%Y-%m-%d').tolist()
    print(week_list)
    for i in range(len(df)):
        if df.loc[i, 'date'].strftime('%Y-%m-%d') in week_list:
            df.loc[i, 'week'] = df.loc[i, 'date'].strftime('%Y-%m-%d')
    df['week'] = df['week'].fillna(method='ffill')
    df['week'] = pd.to_datetime(df['week'], format='%Y-%m-%d')
    
    df['cum_fundflow'] = df.groupby(['Composite Ticker', 'week'])['Net Daily Fund Flow'].transform(lambda x: x.cumsum())
    df = df[['week', 'Composite Ticker', 'NAV', 'cum_fundflow']]
    df.rename(columns={
        'Composite Ticker': 'label', 
        'week': 'date'
    }, inplace=True)
    df['label'] = df['label'].replace({'VNQ': 'Traditional', 'NURE': 'Green'})
    df.to_csv('data/fundflow_data.csv', index=False)
add_week_label()

if __name__ == "__main__":
    ...
    # folder = 'D:\etf_global\data\constituents_us'
    # files  = [files for year in os.listdir(folder) for files in os.listdir(os.path.join(folder, year))]
    
    # week_list = pd.date_range(start='2017-04-01', end='2025-01-01', freq='W-MON').strftime('%Y%m%d').tolist()
    # print(week_list)

    # files = [file for file in files if file[:8] in week_list]
    # logger.info(f'Number of files to process: {len(files)}')

    # final_df = pd.DataFrame()
    # for file in files:
    #     logger.info(f'Processing file: {file}')
    #     path = os.path.join(folder, file[:4], file)
    #     df = preprocessing(path)
    #     final_df = pd.concat([final_df, df], ignore_index=True)

    # final_df.reset_index(drop=True, inplace=True)
    # final_df.to_csv('raw_data.csv', index=False)