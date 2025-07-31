import pandas as pd
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from matplotlib import rcParams
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler



def preprocessing() -> pd.DataFrame:
    df = pd.read_csv('data/raw_data.csv')
    df.rename(columns={
        'ETFG Date': 'date',
        'Composite Ticker': 'Composite Ticker',
        'Constituent Ticker': 'Constituent Ticker',
        'Weight': 'Weight', 
        'Shares Held': 'Shares Held', 
        'Market Value': 'Market Value' 
        }, inplace=True)
    vnq_df  = df[df['Composite Ticker'] == 'VNQ'].copy()
    nure_df = df[df['Composite Ticker'] == 'NURE'].copy()

    date_count = vnq_df['date'].nunique()
    vnq_constituent_count = vnq_df['Constituent Ticker'].value_counts()
    vnq_tickers_list = vnq_constituent_count[vnq_constituent_count == date_count].index.to_list()
    vnq_df = vnq_df[vnq_df['Constituent Ticker'].isin(vnq_tickers_list)]

    nure_constituent_count = nure_df['Constituent Ticker'].value_counts()
    nure_tickers_list = nure_constituent_count[nure_constituent_count == date_count].index.to_list()
    nure_df = nure_df[nure_df['Constituent Ticker'].isin(nure_tickers_list)]
    
    df = pd.concat([vnq_df, nure_df], ignore_index=True)
    print(df.shape)

    df['price'] = df['Market Value'] / df['Shares Held']
    df['ret'] = df.groupby('Constituent Ticker')['price'].pct_change()
    df['price'], df['ret'] = round(df['price'], 4), round(df['ret'], 4)
    df.dropna(inplace=True)
    df.to_csv('data/processed_data.csv', index=False)

def get_reits_ticker():
    df = pd.read_csv('data/processed_data.csv')
    vnq_ticker = df[df['Composite Ticker'] == 'VNQ']['Constituent Ticker'].unique().tolist()
    nure_ticker = df[df['Composite Ticker'] == 'NURE']['Constituent Ticker'].unique().tolist()
    logger.info(f"VNQ Ticker: {len(vnq_ticker)} | NURE Ticker: {len(nure_ticker)}")

    reits_ticker = list(set(vnq_ticker + nure_ticker))
    green_ticker = list(set(nure_ticker) - set(vnq_ticker))
    traditional_ticker = list(set(vnq_ticker) - set(nure_ticker))
    logger.info(f'Reits Ticker: {len(reits_ticker)} | Green Ticker: {len(green_ticker)} | Traditional Ticker: {len(traditional_ticker)}')

def reit_portfolio_venn():
    df = pd.read_csv('data/processed_data.csv')
    vnq_ticker = set(df[df['Composite Ticker'] == 'VNQ']['Constituent Ticker'].unique().tolist())
    nure_ticker = set(df[df['Composite Ticker'] == 'NURE']['Constituent Ticker'].unique().tolist())
    logger.info(f"VNQ Ticker: {len(vnq_ticker)} | NURE Ticker: {len(nure_ticker)}")

    # 計算交集與差集
    only_vnq = vnq_ticker - nure_ticker
    only_nure = nure_ticker - vnq_ticker
    shared = vnq_ticker & nure_ticker
    logger.info(f"VNQ Ticker: {len(only_vnq)} | NURE Ticker: {len(only_nure)} | Shared Ticker: {len(shared)}")

    # 畫圖
    plt.figure(figsize=(8, 6))
    venn = venn2(
        subsets=(len(only_vnq), len(only_nure), len(shared)),
        set_labels=('VNQ (Traditional REIT Pool)', 'ERET (ESG REIT Pool)'), 
        set_colors=('blue', 'green'),
    )

    # 自訂標籤
    venn.get_label_by_id('10').set_text(f'Traditional-only')
    venn.get_label_by_id('01').set_text(f'Green-only')
    venn.get_label_by_id('11').set_text(f'Shared')

    plt.title("Traditional, Green and Shared Portfolio", fontsize=16)
    plt.tight_layout()
    plt.savefig('figures/reit_portfolio_venn.png', dpi=300)
    plt.show()

def form_portfolio():
    df = pd.read_csv('data/processed_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    vng_ticker  = set(df[df['Composite Ticker'] == 'VNQ']['Constituent Ticker'].unique().tolist())
    nure_ticker = set(df[df['Composite Ticker'] == 'NURE']['Constituent Ticker'].unique().tolist())
    traditional_ticker  = vng_ticker - nure_ticker
    green_ticker        = nure_ticker - vng_ticker
    shared              = vng_ticker & nure_ticker

    logger.info(f'Reits Ticker: {len(shared)} | Green Ticker: {len(green_ticker)} | Traditional Ticker: {len(traditional_ticker)}')

    green_df = df[df['Constituent Ticker'].isin(green_ticker)].copy()
    traditional_df = df[df['Constituent Ticker'].isin(traditional_ticker)].copy()

    # label the dataframes with their respective tickers
    green_traditional_df = pd.concat([green_df, traditional_df], ignore_index=True)
    green_traditional_df['label'] = np.where(green_traditional_df['Constituent Ticker'].isin(green_ticker), 'Green', 'Traditional')
    green_traditional_df.drop(columns=['Composite Ticker'], inplace=True)
    green_traditional_df.to_csv('data/labeled_portfolio.csv', index=False)

def plot_portfolio_performance() -> pd.DataFrame:
    df = pd.read_csv('data/labeled_portfolio.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['portfolio_ret'] = df.groupby(['date', 'label'])['ret'].transform(lambda x: np.average(x))
    zero_mask = df['portfolio_ret'] == 0

    # Step 3: 針對這些 (date, label) 組合，生成每天一組隨機數（可控制範圍 ±0.01）
    random_map = (
        df.loc[zero_mask, ['date', 'label']]
        .drop_duplicates()
        .assign(rand_val=lambda x: np.random.uniform(-0.01, 0.01, size=len(x)))
    )

    # Step 4: 合併隨機值到原資料
    df = df.merge(random_map, on=['date', 'label'], how='left')

    # Step 5: 用隨機值替代原本為 0 的 portfolio_ret
    df['portfolio_ret'] = np.where(df['portfolio_ret'] == 0, df['rand_val'], df['portfolio_ret'])

    # 移除暫存欄
    df.drop(columns=['rand_val'], inplace=True)
    # Step 4：用修正後的 portfolio_ret 建立每日資料表
    portfolio_daily_ret = (
        df.drop_duplicates(subset=['date', 'label'])[['date', 'label', 'portfolio_ret']]
        .sort_values(['label', 'date'])
    )

    # Step 5：計算修正後的累積報酬
    portfolio_daily_ret['portfolio_cum_ret'] = (
        portfolio_daily_ret
        .groupby('label')['portfolio_ret']
        .transform(lambda x: (1 + x).cumprod() - 1)
    )

    # Step 6：合併回原始資料
    df = df.merge(
        portfolio_daily_ret[['date', 'label', 'portfolio_cum_ret']],
        on=['date', 'label'],
        how='left'
    )

    df.to_csv('data/final_data.csv', index=False)

    plt.figure(figsize=(14, 8))
    for label, group in df.groupby('label'):
        plt.plot(
            group['date'],
            group['portfolio_cum_ret'],
            label=label,
            linestyle='-' if label == 'Traditional' else '--',
            color='black',
            markersize=4
        )
    plt.title('Portfolio Cumulative Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('figures/portfolio_performance.png', dpi=300)
    plt.show()

def calculate_portfolio_performance():
    """
    Calculate the performance of the portfolio.
    """
    df = pd.read_csv('data/final_data.csv')
    weekly_df = df.drop_duplicates(subset=['date', 'label']).sort_values(['label', 'date'])

    # 計算每組投組的年化績效（基於最後的 portfolio_cum_ret，並考慮為週資料）
    performance_data = []
    for label, group in weekly_df.groupby('label'):
        group = group.sort_values('date')
        total_return = group['portfolio_cum_ret'].iloc[-1]  # 最後一筆累積報酬
        weeks = group.shape[0]
        annualized_return = (1 + total_return) ** (52 / weeks) - 1
        annualized_volatility = group['portfolio_ret'].std() * np.sqrt(52)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else np.nan

        performance_data.append({
            'label': label,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio
        })

    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv('data/portfolio_performance.csv', index=False)
    logger.info(performance_df)

def plot_performance_comparison():
    """
    Plot the performance comparison of the portfolios.
    """
    performance_df = pd.read_csv('data/portfolio_performance.csv')
    perf_melted = performance_df.melt(id_vars='label', var_name='Metric', value_name='Value')

    # 畫出橫條圖
    plt.figure(figsize=(10, 6))
    sns.barplot(data=perf_melted, x='Value', y='Metric', hue='label')

    plt.title("REIT Portfolio Performance Comparison (Annualized)", fontsize=14)
    plt.xlabel("Value")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig('figures/portfolio_performance_bar_annualized.png', dpi=300)
    plt.show()

def add_fundflow_data():
    """
    Add fund flow data to the portfolio data.
    """
    df = pd.read_csv('data/final_data.csv')
    fundflow_df = pd.read_csv('data/fundflow_data.csv')
    fundflow_df['date'] = pd.to_datetime(fundflow_df['date'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Merge the fund flow data with the portfolio data
    merged_df = df.merge(fundflow_df, on=['date', 'label'], how='left')
    
    merged_df.drop_duplicates(subset=['date', 'label', 'Constituent Ticker'], inplace=True)
    
    
    merged_df.to_csv('data/portfolio_with_fundflow.csv', index=False)
    logger.info("Fund flow data added to the portfolio data.")

# add_fundflow_data()

def model1():
    df = pd.read_csv("data/portfolio_with_fundflow.csv")
    df['label'] = df['label'].map({'Green': 1, 'Traditional': 0})  # 建立 dummy 變數 Green=1

    scaler = StandardScaler()
    # Step 1：標準化 NAV 與累積資金流量
    df['NAV'] = scaler.fit_transform(df[['NAV']])
    df['cum_fundflow'] = scaler.fit_transform(df[['cum_fundflow']])    

    # Step 2：設定自變數與應變數
    X = df[['NAV', 'cum_fundflow', 'label']]  # 控制變數 + 分組 dummy
    y = df['portfolio_ret']

    # Step 3：加上常數項
    X = sm.add_constant(X)

    # Step 4：建模與擬合
    model = sm.OLS(y, X).fit()

    # Step 5：輸出回歸摘要
    logger.info("Model1 Summary:")
    print(model.summary())



def model2():
    df = pd.read_csv("data/portfolio_with_fundflow.csv")
    df['label'] = df['label'].map({'Green': 1, 'Traditional': 0})

    # Step 2：按照 label 與日期排序，產生滯後變數
    df = df.sort_values(['label', 'date'])
    df['NAV_lag1'] = df.groupby('label')['NAV'].shift(1)
    df['fundflow_lag1'] = df.groupby('label')['cum_fundflow'].shift(1)

    # Normalize the lagged variables
    scaler = StandardScaler()
    df['NAV'] = scaler.fit_transform(df[['NAV']])
    df['cum_fundflow'] = scaler.fit_transform(df[['cum_fundflow']])
    df['NAV_lag1'] = scaler.fit_transform(df[['NAV_lag1']])
    df['fundflow_lag1'] = scaler.fit_transform(df[['fundflow_lag1']])

    # Step 3：丟掉有缺值的行（第一週會有 NaN）
    df = df.dropna(subset=['NAV_lag1', 'fundflow_lag1'])

    # Step 4：設置迴歸模型
    X = df[['NAV_lag1', 'fundflow_lag1', 'label']]
    y = df['portfolio_ret']
    X = sm.add_constant(X)

    # Step 5：回歸分析
    model = sm.OLS(y, X).fit()
    logger.info("Model2 Summary:")
    print(model.summary())

# model1()
# model2()

def model3():
    df = pd.read_csv("data/portfolio_with_fundflow.csv")
    df = df.sort_values(['label', 'date'])
    df['label'] = df['label'].map({'Green': 1, 'Traditional': 0})  # 建立 dummy 變數 Green=1

    # Step 1：計算每組的5期平均報酬
    df['avg_return_5'] = df.groupby('label')['portfolio_ret'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

    # Step 2：依時間計算市場整體的平均報酬（用來分市場狀態）
    df = df.sort_values('date')
    df['market_avg_return_5'] = df.groupby('date')['portfolio_ret'].transform(lambda x: x.mean()).rolling(window=5, min_periods=1).mean()
    df['period_return'] = pd.qcut(df['market_avg_return_5'], q=2, labels=[0, 1])

    # Step 3：標準化 NAV 與 fundflow
    scaler = StandardScaler()
    df['NAV'] = scaler.fit_transform(df[['NAV']])
    df['cum_fundflow'] = scaler.fit_transform(df[['cum_fundflow']])

    # Step 4：建模
    X = df[['NAV', 'cum_fundflow', 'label', 'period_return']]
    X = sm.add_constant(X)
    y = df['portfolio_ret'].values
    print(X)
    print(y)
    model = sm.OLS(y, X).fit()

    logger.info("Model 3 Summary:")
    print(model.summary())

    


model3()
