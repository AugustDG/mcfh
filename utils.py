import pandas as pd


def read_stock_data():
    # Read stock data from a CSV file
    df = pd.read_csv('hackathon_sample_v2.csv')

    # Filter out the columns that are not needed
    df = df[['date', 'cusip', 'prc', 'eps_actual', 'ni_me']]
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

    # Pivot the table to have stock names as columns and dates as index
    df_pivoted = df.pivot(index='date', columns=['cusip'], values=['prc', 'eps_actual', 'ni_me'])

    print(df_pivoted)

    # Replace missing values with a 0 price
    df_pivoted = df_pivoted.fillna(0.0)

    return df_pivoted