import pandas as pd
import numpy as np
import numba


# All preprocessing in one function
def preprocess_deals_data(data):
    """
    Preprocess dataframe with deals
    Consolidate market orders, returns new dataframe
    Add features: Day of week, Hour, Time
    """
    
    # Add columns and rename
    data['DELTA_OI'] = data['OPEN_POS'].diff()  # OI difference
    data['Time'] = pd.to_datetime(data['MOMENT'], format='%Y%m%d%H%M%S%f')
    data.rename(columns={'PRICE_DEAL': 'PRICE'}, inplace=True)

    # aggregate to obtain market order instant impact
    # deals with the same time are caused by one market order
    data = data[['Time', 'MOMENT', 'PRICE', 'DIRECTION', 'VOLUME', 'DELTA_OI']] \
        .groupby(by=['Time', 'MOMENT', 'DIRECTION'], as_index=False) \
        .agg({'PRICE': ['min', 'max'], 'VOLUME': 'sum', 'DELTA_OI': 'sum'})

    # Flatten columns (!)
    data.columns = ['_'.join(z) if z[1] != '' else z[0] for z in data.columns]

    # Add datetime features
    data['Date'] = data['Time'].map(lambda d: d.date())  # only date
    data['TimeOnly'] = data['Time'].map(lambda d: d.time())  # only time
    data['DOW'] = data['Time'].map(lambda d: d.isoweekday())  # day of week
    data['H'] = data['Time'].map(lambda d: d.hour)  # hour (for filtering)

    # Impact calculation
    data['Impact'] = data['PRICE_max'] - data['PRICE_min']

    return data


def preprocess_lob_data(lob):
    '''
    Preprocess MOEX ticks DataFrame
    '''
    bid = lob[['MOMENT', 'PRICE', 'VOLUME']][(lob['TYPE']=='B') & (lob['DEAL_ID'].isna())] \
        .groupby(by=['MOMENT'], as_index=False).agg({'PRICE': 'last', 'VOLUME': 'last'})
    bid.rename(columns={'PRICE': 'BID_PRICE', 'VOLUME': 'BID_SIZE'}, inplace=True)
    
    ask = lob[['MOMENT', 'PRICE', 'VOLUME']][(lob['TYPE']=='S') & (lob['DEAL_ID'].isna())] \
        .groupby(by=['MOMENT'], as_index=False).agg({'PRICE': 'last', 'VOLUME': 'last'})
    ask.rename(columns={'PRICE': 'ASK_PRICE', 'VOLUME': 'ASK_SIZE'}, inplace=True)
    
    res_lob = pd.merge(left=bid, right=ask, on='MOMENT', how='outer')
    res_lob.fillna(method='ffill', inplace=True)
    
    return res_lob


@numba.njit()
def process_lob(order_time, time, bid, ask, bsize, asize):
    '''
    Function gets order time array (from deals file), and market bid, ask and time array (from LOB file)
    For each element in order time array it finds corresponding bid and ask
    Returns arrays of bids ands asks for market orders
    '''
    res_bid = np.zeros_like(order_time, dtype=np.float32)
    res_ask = np.zeros_like(order_time, dtype=np.float32)
    
    res_bsize = np.zeros_like(order_time, dtype=np.float32)
    res_asize = np.zeros_like(order_time, dtype=np.float32)
    
    j = 0
    for i, t in enumerate(order_time):
        while time[j] < t:
            j += 1
        res_bid[i] = bid[j]
        res_ask[i] = ask[j]
        res_bsize[i] = bsize[j]
        res_asize[i] = asize[j]

    return res_bid, res_ask, res_bsize, res_asize


def add_lob_prices(deals, lob):
    '''
    Function processes market order dataframe and add bid and ask columns
    Returns new dataframe
    '''
    # Extract series for numba
    order_time = deals['MOMENT'].to_numpy()

    lob_time = lob['MOMENT'].to_numpy()
    lob_bid =  lob['BID_PRICE'].to_numpy()
    lob_ask =  lob['ASK_PRICE'].to_numpy()
    lob_bsize = lob['BID_SIZE'].to_numpy()
    lob_asize = lob['ASK_SIZE'].to_numpy()
    
    order_bid, order_ask, order_bsize, order_asize = process_lob(order_time, lob_time, lob_bid, lob_ask, lob_bsize, lob_asize)
    
    deals['BID'] = order_bid
    deals['ASK'] = order_ask
    deals['BID_SIZE'] = order_bsize
    deals['ASK_SIZE'] = order_asize
    
    deals['MID'] = (deals['BID'] + deals['ASK']) / 2
    deals['Spread'] = np.round(deals['ASK'] - deals['BID'], decimals=4)
    
    return deals