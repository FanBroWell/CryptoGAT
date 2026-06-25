import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def add_technical_indicators(df, base_window=14):
    df = df.sort_index().copy()
    
    features_dict = {}
    
    df['open_norm'] = df['open'] / df['close'].shift(1)
    df['high_norm'] = df['high'] / df['close'].shift(1)
    df['low_norm'] = df['low'] / df['close'].shift(1)
    df['close_norm'] = df['close'] / df['close'].shift(1)
    
    df['volume_ma5'] = df['volume'].rolling(window=5, min_periods=1).mean()
    df['volume_norm'] = df['volume'] / (df['volume_ma5'] + 1e-10)
    
    features_dict['open_norm'] = df['open_norm']
    features_dict['high_norm'] = df['high_norm']
    features_dict['low_norm'] = df['low_norm']
    features_dict['close_norm'] = df['close_norm']
    features_dict['volume_norm'] = df['volume_norm']
    
    sma_short = df['close'].rolling(window=7, min_periods=1).mean()
    sma_long = df['close'].rolling(window=base_window, min_periods=1).mean()
    
    features_dict['sma_ratio_7'] = df['close'] / sma_short
    features_dict['sma_ratio_14'] = df['close'] / sma_long
    features_dict['sma_cross'] = sma_short / sma_long
    
    ema_short = df['close'].ewm(span=7, adjust=False).mean()
    ema_long = df['close'].ewm(span=base_window, adjust=False).mean()
    
    features_dict['ema_ratio_7'] = df['close'] / ema_short
    features_dict['ema_ratio_14'] = df['close'] / ema_long
    
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    
    features_dict['macd'] = macd_line / (df['close'] + 1e-10)
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=base_window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=base_window, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    features_dict['rsi_14'] = rsi / 100
    
    gain_7 = (delta.where(delta > 0, 0)).rolling(window=7, min_periods=1).mean()
    loss_7 = (-delta.where(delta < 0, 0)).rolling(window=7, min_periods=1).mean()
    rs_7 = gain_7 / (loss_7 + 1e-10)
    rsi_7 = 100 - (100 / (1 + rs_7))
    
    features_dict['rsi_7'] = rsi_7 / 100
    
    features_dict['roc_7'] = (df['close'] - df['close'].shift(7)) / df['close'].shift(7)
    features_dict['roc_14'] = (df['close'] - df['close'].shift(base_window)) / df['close'].shift(base_window)
    
    features_dict['momentum_7'] = df['close'] / df['close'].shift(7)
    
    bb_middle = df['close'].rolling(window=20, min_periods=1).mean()
    bb_std = df['close'].rolling(window=20, min_periods=1).std()
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    
    features_dict['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
    features_dict['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-10)
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=base_window, min_periods=1).mean()
    
    features_dict['atr_norm'] = atr / df['close']
    
    returns = df['close'].pct_change()
    features_dict['volatility_7'] = returns.rolling(window=7, min_periods=1).std()
    features_dict['volatility_14'] = returns.rolling(window=base_window, min_periods=1).std()
    features_dict['volatility_30'] = returns.rolling(window=30, min_periods=1).std()
    
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    features_dict['obv_norm'] = obv / (obv.rolling(window=30, min_periods=1).mean() + 1e-10)
    
    vpt = (df['volume'] * df['close'].pct_change()).fillna(0).cumsum()
    features_dict['vpt_norm'] = vpt / (vpt.rolling(window=30, min_periods=1).mean() + 1e-10)
    
    features_dict['volume_change'] = df['volume'] / df['volume'].shift(1)
    
    vol_ma_long = df['volume'].rolling(window=30, min_periods=1).mean()
    features_dict['volume_trend'] = df['volume'] / vol_ma_long
    
    body = df['close'] - df['open']
    upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
    lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
    
    features_dict['upper_shadow'] = upper_shadow / (df['close'] + 1e-10)
    features_dict['lower_shadow'] = lower_shadow / (df['close'] + 1e-10)
    features_dict['body_ratio'] = body / (df['close'] + 1e-10)
    
    features_dict['daily_range'] = (df['high'] - df['low']) / df['close']
    
    features_dict['return_lag_1'] = df['close'].pct_change(1)
    features_dict['return_lag_3'] = df['close'].pct_change(3)
    features_dict['return_lag_7'] = df['close'].pct_change(7)
    features_dict['return_lag_14'] = df['close'].pct_change(14)
    features_dict['return_lag_30'] = df['close'].pct_change(30)
    
    features_df = pd.DataFrame(features_dict, index=df.index)
    
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.ffill().bfill()
    features_df = features_df.fillna(1.0)
    
    ratio_features = [
        'open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_norm',
        'sma_ratio_7', 'sma_ratio_14', 'sma_cross',
        'ema_ratio_7', 'ema_ratio_14', 'momentum_7',
        'obv_norm', 'vpt_norm', 'volume_change', 'volume_trend'
    ]
    
    small_value_features = [
        'bb_width', 'atr_norm', 
        'volatility_7', 'volatility_14', 'volatility_30',
        'upper_shadow', 'lower_shadow', 'body_ratio', 'daily_range'
    ]
    
    percentile_features = [
        'rsi_14', 'rsi_7', 'bb_position'
    ]
    
    return_features = [
        'roc_7', 'roc_14', 'macd',
        'return_lag_1', 'return_lag_3', 'return_lag_7', 'return_lag_14', 'return_lag_30'
    ]
    
    for col in features_df.columns:
        if col in ratio_features:
            features_df[col] = features_df[col].clip(0.5, 2.0)
        elif col in small_value_features:
            features_df[col] = features_df[col].clip(0, 1.0)
        elif col in percentile_features:
            features_df[col] = features_df[col].clip(0, 1.0)
        elif col in return_features:
            features_df[col] = features_df[col].clip(-0.5, 0.5)
            features_df[col] = (features_df[col] + 0.5)
        else:
            features_df[col] = features_df[col].clip(0, 2.0)
    
    return features_df

def get_feature_count():
    return 5 + 6 + 5 + 6 + 4 + 4 + 5

def get_feature_names():
    feature_names = [
        'open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_norm',
        'sma_ratio_7', 'sma_ratio_14', 'sma_cross',
        'ema_ratio_7', 'ema_ratio_14', 'macd_norm',
        'rsi_14', 'rsi_7', 'roc_7', 'roc_14', 'momentum_7',
        'bb_position', 'bb_width', 'atr_norm',
        'volatility_7', 'volatility_14', 'volatility_30',
        'obv_norm', 'vpt_norm', 'volume_change', 'volume_trend',
        'upper_shadow', 'lower_shadow', 'body_ratio', 'daily_range',
        'return_lag_1', 'return_lag_3', 'return_lag_7', 'return_lag_14', 'return_lag_30'
    ]
    
    return feature_names
