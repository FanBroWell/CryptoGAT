import pandas as pd
import numpy as np
import pickle
import os
from glob import glob

def load_and_normalize(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    df['open_norm'] = df['open'] / df['close'].shift(1)
    df['high_norm'] = df['high'] / df['close'].shift(1)
    df['low_norm'] = df['low'] / df['close'].shift(1)
    df['close_norm'] = df['close'] / df['close'].shift(1)
    
    df['volume_ma5'] = df['volume'].rolling(window=5, min_periods=1).mean()
    df['volume_norm'] = df['volume'] / df['volume_ma5']
    
    df = df.iloc[1:].copy()
    
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    
    features = df[['open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_norm']].values
    features = np.nan_to_num(features, nan=1.0, posinf=2.0, neginf=0.5)
    features = np.clip(features, 0.5, 2.0)
    
    close_prices = df['close'].values
    gt = np.zeros(len(close_prices))
    for i in range(1, len(close_prices)):
        if close_prices[i - 1] > 0:
            gt[i] = (close_prices[i] - close_prices[i - 1]) / close_prices[i - 1]
    
    mask = np.ones(len(close_prices), dtype=np.float32)
    mask[0] = 0.0
    
    return {
        'dates': df['date'].dt.date.tolist(),
        'features': features,
        'ground_truth': gt,
        'close_prices': close_prices,
        'mask': mask
    }

def main():
    base_dir = '../dataset/Binance_USDT'
    output_dir = '../dataset/CRYPTO_1D_ALL'
    
    csv_files = glob(os.path.join(base_dir, '*USDT_daily_1000days.csv'))
    all_symbols = sorted([os.path.basename(f).replace('_daily_1000days.csv', '') 
                          for f in csv_files])
    
    exclude = ['USDCUSDT', 'TUSDUSDT']
    coin_symbols = [c for c in all_symbols if c not in exclude]
    
    crypto_data = []
    failed = []
    
    for i, symbol in enumerate(coin_symbols, 1):
        file_path = os.path.join(base_dir, f"{symbol}_daily_1000days.csv")
        try:
            data = load_and_normalize(file_path)
            data['coin_name'] = symbol.replace('USDT', '')
            crypto_data.append(data)
        except Exception as e:
            print(f"Error loading {symbol}: {str(e)[:50]}")
            failed.append(symbol)
    
    if failed:
        print(f"Failed to load {len(failed)} coins")
    
    all_dates = [set(d['dates']) for d in crypto_data]
    common_dates = sorted(list(set.intersection(*all_dates)))
    
    aligned_data = []
    for data in crypto_data:
        date_to_idx = {date: idx for idx, date in enumerate(data['dates'])}
        indices = [date_to_idx[date] for date in common_dates]
        
        aligned_data.append({
            'coin_name': data['coin_name'],
            'features': data['features'][indices],
            'ground_truth': data['ground_truth'][indices],
            'close_prices': data['close_prices'][indices],
            'mask': data['mask'][indices]
        })
    
    n_coins = len(aligned_data)
    n_days = len(common_dates)
    n_features = 5
    
    eod_data = np.zeros((n_coins, n_days, n_features), dtype=np.float32)
    mask_data = np.zeros((n_coins, n_days), dtype=np.float32)
    gt_data = np.zeros((n_coins, n_days), dtype=np.float32)
    price_data = np.zeros((n_coins, n_days), dtype=np.float32)
    
    for i, data in enumerate(aligned_data):
        eod_data[i] = data['features']
        mask_data[i] = data['mask']
        gt_data[i] = data['ground_truth']
        price_data[i] = data['close_prices']
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "eod_data.pkl"), "wb") as f:
        pickle.dump(eod_data, f)
    
    with open(os.path.join(output_dir, "mask_data.pkl"), "wb") as f:
        pickle.dump(mask_data, f)
    
    with open(os.path.join(output_dir, "gt_data.pkl"), "wb") as f:
        pickle.dump(gt_data, f)
    
    with open(os.path.join(output_dir, "price_data.pkl"), "wb") as f:
        pickle.dump(price_data, f)
    
    coin_names = [d['coin_name'] for d in aligned_data]
    with open(os.path.join(output_dir, "coin_names.txt"), "w") as f:
        for name in coin_names:
            f.write(f"{name}\n")
    
    print(f"Processing complete. Saved {n_coins} coins, {n_days} days to {output_dir}")

if __name__ == '__main__':
    main()
