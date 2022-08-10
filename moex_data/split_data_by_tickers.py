import dask.dataframe as dd
import pandas as pd
import sys


def main():
    if len(sys.argv) >= 3:
        data_file = sys.argv[1]
        tickers_file = sys.argv[2]
        postfix = sys.argv[3] if len(sys.argv) == 4 else ''

        # read ticker list
        with open(tickers_file) as f:
            tickers = []
            for line in f:
                tickers += line.strip().split(sep=',')
        tickers = [ticker.strip() for ticker in tickers]

        # load data
        ddf = dd.read_csv(data_file, 
                          dtype={'VOLUME': 'float64'})
        for ticker in tickers:
            print(f"{ticker} in process")
            df = ddf[ddf['#SYMBOL'] == ticker].compute()  # read data 
            df.reset_index().to_feather(f'{ticker}{postfix}.feather')


if __name__ == "__main__":
    main()