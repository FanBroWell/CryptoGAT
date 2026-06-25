import os
import pickle
import numpy as np
import pandas as pd
import random
from feature_engineer import add_technical_indicators, get_feature_count

def load_raw_ohlcv_from_csv(coin_name, csv_dir, target_days=999):
    csv_filename = f"{coin_name}USDT_daily_1000days.csv"
    csv_path = os.path.join(csv_dir, csv_filename)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if 'date' not in df.columns or 'close' not in df.columns:
        raise ValueError(f"CSV file format error: {csv_path}")
    
    df = df.tail(target_days).reset_index(drop=True)
    
    if len(df) < target_days:
        print(f"Warning: {coin_name}: actual data {len(df)} days < target {target_days} days")
    
    return df[['open', 'high', 'low', 'close', 'volume']]

def main():
    RANDOM_SEED = 123456789
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    input_dir = '../dataset/CRYPTO_1D_ALL'
    output_dir = '../dataset/CRYPTO_1D_ENHANCED'
    
    with open(os.path.join(input_dir, 'eod_data.pkl'), 'rb') as f:
        eod_data = pickle.load(f)
    
    with open(os.path.join(input_dir, 'price_data.pkl'), 'rb') as f:
        price_data = pickle.load(f)
    
    with open(os.path.join(input_dir, 'mask_data.pkl'), 'rb') as f:
        mask_data = pickle.load(f)
    
    with open(os.path.join(input_dir, 'gt_data.pkl'), 'rb') as f:
        gt_data = pickle.load(f)
    
    with open(os.path.join(input_dir, 'coin_names.txt'), 'r') as f:
        coin_names = [line.strip() for line in f]
    
    n_coins, n_days, n_orig_features = eod_data.shape
    
    n_features = get_feature_count()
    enhanced_features = np.zeros((n_coins, n_days, n_features), dtype=np.float32)
    
    csv_dir = '../dataset/Binance_USDT'
    failed_coins = []
    
    for i, coin_name in enumerate(coin_names):
        try:
            df = load_raw_ohlcv_from_csv(coin_name, csv_dir, target_days=n_days)
            features_df = add_technical_indicators(df, base_window=14)
            
            if features_df.shape[0] != n_days:
                if features_df.shape[0] < n_days:
                    pad_rows = n_days - features_df.shape[0]
                    pad_data = np.repeat(features_df.iloc[0:1].values, pad_rows, axis=0)
                    features_array = np.vstack([pad_data, features_df.values])
                else:
                    features_array = features_df.iloc[:n_days].values
                enhanced_features[i, :, :] = features_array
            else:
                enhanced_features[i, :, :] = features_df.values
            
        except Exception as e:
            print(f"Error processing {coin_name}: {str(e)[:50]}")
            failed_coins.append(coin_name)
            enhanced_features[i, :, :5] = eod_data[i, :, :]
            enhanced_features[i, :, 5:] = 1.0
    
    if failed_coins:
        print(f"Failed coins ({len(failed_coins)}): {', '.join(failed_coins)}")
    
    nan_count = np.isnan(enhanced_features).sum()
    inf_count = np.isinf(enhanced_features).sum()
    
    if nan_count > 0 or inf_count > 0:
        for i in range(n_coins):
            for j in range(n_features):
                col = enhanced_features[i, :, j]
                col[np.isinf(col)] = np.nan
                
                if np.isnan(col).any():
                    mask = ~np.isnan(col)
                    if mask.sum() > 0:
                        indices = np.arange(len(col))
                        col[~mask] = np.interp(indices[~mask], indices[mask], col[mask])
                    else:
                        col[:] = 1.0
                
                enhanced_features[i, :, j] = col
        
        nan_count_after = np.isnan(enhanced_features).sum()
        inf_count_after = np.isinf(enhanced_features).sum()
        
        if nan_count_after > 0 or inf_count_after > 0:
            enhanced_features = np.nan_to_num(enhanced_features, nan=1.0, posinf=2.0, neginf=0.5)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'eod_data.pkl'), 'wb') as f:
        pickle.dump(enhanced_features, f)
    
    with open(os.path.join(output_dir, 'mask_data.pkl'), 'wb') as f:
        pickle.dump(mask_data, f)
    
    with open(os.path.join(output_dir, 'gt_data.pkl'), 'wb') as f:
        pickle.dump(gt_data, f)
    
    with open(os.path.join(output_dir, 'price_data.pkl'), 'wb') as f:
        pickle.dump(price_data, f)
    
    with open(os.path.join(output_dir, 'coin_names.txt'), 'w') as f:
        for coin in coin_names:
            f.write(f"{coin}\n")
    
    print(f"Processing complete. Enhanced features: {enhanced_features.shape}, saved to {output_dir}")

if __name__ == '__main__':
    main()
