DROP_COLS = ['TRIMA5', 'MOM20', 'HT_TRENDLINE', 'KAMA20', 'fastdsr', 'fastd', 'CCI5', 'SAR', 'ROC5', 'TRIMA20', 'MOM10', 'HT_DCPERIOD', 'Trange', 'PPO', 'ADX20', 'fastksr', 'KAMA30', 'ADX5', 'CCI10', 'slowd', 'TYPPRICE', 'CCI15', 'fastk', 'ULTOSC', 'ADX10', 'APO', 'slowk', 'BETA', 'WILLR', 'ATR', 'MOM15', 'TRIMA10', 'KAMA10', 'ROC10', 'ROC20']

COLS_TO_AGG = {
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'sma5': 'last', 'sma10': 'last', 'sma15': 'last', 'sma20': 'last',
            'ema5': 'last', 'ema10': 'last', 'ema15': 'last', 'ema20': 'last', 'upperband': 'last', 'middleband': 'last','lowerband': 'last','macd510': 'last',
            'macd520': 'last', 'macd1020': 'last', 'macd1520': 'last', 'macd1226': 'last', 'RSI14': 'last', 'RSI8': 'last'
        }

COLS_TO_RSML = ['open', 'high', 'low', 'close', 'volume', 'sma5', 'sma10',
                            'sma15', 'sma20', 'ema5', 'ema10', 'ema15', 'ema20',
                            'upperband', 'middleband', 'lowerband', 'macd510', 'macd520',
                            'macd1020', 'macd1520', 'macd1226', 'RSI14', 'RSI8']
