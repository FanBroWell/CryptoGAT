import random
import numpy as np
import os
import torch
import torch.nn as nn
from models.gat import GAT, build_correlation_matrix
from evaluator import evaluate
import pickle
from datetime import datetime

RANDOM_SEED = 123456789

def set_random_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(RANDOM_SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

market_name = 'CRYPTO_1D_ENHANCED'
lookback_length = 30
data_path = '../dataset'
stock_num = 66
fea_num = 35
steps = 1
valid_index = 599
test_index = 799

hidden_size = 64
num_layers = 2
dropout = 0
num_graph_layer = 2
epochs = 100
learning_rate = 0.00025
weight_decay = 0
use_scheduler = True
correlation_threshold = 0.5

def load_data():
    dataset_path = os.path.join(data_path, market_name)
    
    if not os.path.exists(dataset_path):
        print(f"Error: Data directory does not exist {dataset_path}")
        exit(1)
    
    import sys
    if not hasattr(np, '_core'):
        sys.modules['numpy._core'] = np.core
    
    try:
        with open(os.path.join(dataset_path, "eod_data.pkl"), "rb") as f:
            eod_data = pickle.load(f)
        with open(os.path.join(dataset_path, "mask_data.pkl"), "rb") as f:
            mask_data = pickle.load(f)
        with open(os.path.join(dataset_path, "gt_data.pkl"), "rb") as f:
            gt_data = pickle.load(f)
        with open(os.path.join(dataset_path, "price_data.pkl"), "rb") as f:
            price_data = pickle.load(f)
    except ModuleNotFoundError as e:
        if 'numpy._core' in str(e):
            with open(os.path.join(dataset_path, "eod_data.pkl"), "rb") as f:
                eod_data = pickle.load(f, encoding='latin1')
            with open(os.path.join(dataset_path, "mask_data.pkl"), "rb") as f:
                mask_data = pickle.load(f, encoding='latin1')
            with open(os.path.join(dataset_path, "gt_data.pkl"), "rb") as f:
                gt_data = pickle.load(f, encoding='latin1')
            with open(os.path.join(dataset_path, "price_data.pkl"), "rb") as f:
                price_data = pickle.load(f, encoding='latin1')
        else:
            raise
    
    try:
        with open(os.path.join(dataset_path, "coin_names.txt"), "r") as f:
            coin_names = [line.strip() for line in f]
    except:
        coin_names = [f"Coin{i+1}" for i in range(stock_num)]
    
    trade_dates = min(mask_data.shape[1], gt_data.shape[1])
    if trade_dates > 0 and np.all(gt_data[:, trade_dates-1] == 0):
        trade_dates = trade_dates - 1
    
    if eod_data.shape[2] != fea_num:
        print(f"Error: Feature count mismatch!")
        exit(1)
    
    return {
        'eod_data': eod_data,
        'mask_data': mask_data,
        'gt_data': gt_data,
        'price_data': price_data,
        'coin_names': coin_names,
        'trade_dates': trade_dates
    }

def get_batch(data, offset):
    eod_data = data['eod_data']
    mask_data = data['mask_data']
    price_data = data['price_data']
    gt_data = data['gt_data']
    
    seq_len = lookback_length
    mask_batch = mask_data[:, offset: offset + seq_len + steps]
    mask_batch = np.min(mask_batch, axis=1)
    
    return (
        eod_data[:, offset:offset + seq_len, :],
        np.expand_dims(mask_batch, axis=1),
        np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
        np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1)
    )

def build_relation_graph(data):
    price_data = data['price_data']
    relation_matrix = build_correlation_matrix(price_data, threshold=correlation_threshold)
    return relation_matrix

def validate(model, data, relation_matrix, start_index, end_index):
    model.eval()
    
    with torch.no_grad():
        cur_pred = []
        cur_gt = []
        cur_mask = []
        loss = 0.0
        
        for offset in range(start_index - lookback_length - steps + 1, 
                           end_index - lookback_length - steps + 1):
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device),
                get_batch(data, offset)
            )
            
            prediction = model(data_batch, relation_matrix)
            cur_loss = nn.MSELoss()(prediction * mask_batch, gt_batch * mask_batch)
            loss += cur_loss.item()
            
            cur_pred.append(prediction[:, 0].cpu().numpy())
            cur_gt.append(gt_batch[:, 0].cpu().numpy())
            cur_mask.append(mask_batch[:, 0].cpu().numpy())
        
        loss /= (end_index - start_index)
        
        cur_pred = np.array(cur_pred).T
        cur_gt = np.array(cur_gt).T
        cur_mask = np.array(cur_mask).T
        
        perf = evaluate(cur_pred, cur_gt, cur_mask)
    
    return loss, perf

def train():
    data = load_data()
    
    relation_matrix_np = build_relation_graph(data)
    relation_matrix = torch.FloatTensor(relation_matrix_np).to(device)
    
    model = GAT(
        d_feat=fea_num,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_graph_layer=num_graph_layer
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
    
    best_valid_loss = np.inf
    best_valid_perf = None
    best_test_perf = None
    best_epoch = 0
    
    trade_dates = data['trade_dates']
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_indices = list(range(0, valid_index - lookback_length - steps + 1))
        random.shuffle(train_indices)
        
        for offset in train_indices:
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device),
                get_batch(data, offset)
            )
            
            optimizer.zero_grad()
            prediction = model(data_batch, relation_matrix)
            cur_loss = nn.MSELoss()(prediction * mask_batch, gt_batch * mask_batch)
            cur_loss.backward()
            optimizer.step()
            train_loss += cur_loss.item()
        
        train_loss /= len(train_indices)
        
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            valid_loss, valid_perf = validate(
                model, data, relation_matrix, valid_index, test_index
            )
            
            test_loss, test_perf = validate(
                model, data, relation_matrix, test_index, trade_dates
            )
            
            if scheduler is not None:
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(valid_loss)
                new_lr = optimizer.param_groups[0]['lr']
                lr_changed = new_lr < old_lr
            else:
                lr_changed = False
                new_lr = learning_rate
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_valid_perf = valid_perf
                best_test_perf = test_perf
                best_epoch = epoch + 1
                improved = " *"
            else:
                improved = ""
            
            print(f"Epoch {epoch+1:3d}/{epochs} | " +
                  f"Train: {train_loss:.2e} | " +
                  f"Valid Loss: {valid_loss:.2e} | " +
                  f"Valid IC: {valid_perf['IC']:.4f} | " +
                  f"Test IC: {test_perf['IC']:.4f} | " +
                  f"Test Sharpe: {test_perf['sharpe5']:+.4f}{improved}")
            
            if lr_changed:
                print(f"    LR reduced: {old_lr:.6f} → {new_lr:.6f}")
    
    print(f"\nBest Model (Epoch {best_epoch})")
    print(f"Validation Performance:")
    print(f"  MSE:      {best_valid_perf['mse']:.2e}")
    print(f"  IC:       {best_valid_perf['IC']:.4f}")
    print(f"  ICIR:     {best_valid_perf['ICIR']:.4f}")
    print(f"  Prec@10:  {best_valid_perf['prec_10']:.4f}")
    print(f"  Sharpe:   {best_valid_perf['sharpe5']:+.4f}")
    
    print(f"\nTest Performance:")
    print(f"  MSE:      {best_test_perf['mse']:.2e}")
    print(f"  IC:       {best_test_perf['IC']:.4f}")
    print(f"  ICIR:     {best_test_perf['ICIR']:.4f}")
    print(f"  Prec@10:  {best_test_perf['prec_10']:.4f}")
    print(f"  Sharpe:   {best_test_perf['sharpe5']:+.4f}")
    
    save_results(best_epoch, best_valid_perf, best_test_perf, num_params)
    
    return {
        'best_epoch': best_epoch,
        'best_valid_perf': best_valid_perf,
        'best_test_perf': best_test_perf,
        'num_params': num_params
    }

def save_results(best_epoch, best_valid_perf, best_test_perf, num_params):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = '../results'
    os.makedirs(output_dir, exist_ok=True)
    
    result_path = os.path.join(output_dir, f'gat_enhanced_{timestamp}.txt')
    
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(f"Dataset: {market_name}\n")
        f.write(f"Features: {fea_num}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n")
        f.write(f"Model parameters: {num_params:,}\n\n")
        f.write(f"Best Model (Epoch {best_epoch}):\n")
        f.write(f"  Valid IC: {best_valid_perf['IC']:.4f}\n")
        f.write(f"  Test IC: {best_test_perf['IC']:.4f}\n")
        f.write(f"  Test Sharpe: {best_test_perf['sharpe5']:+.4f}\n")
        f.write(f"  Test ICIR: {best_test_perf['ICIR']:.4f}\n")
        f.write(f"  Test Prec@10: {best_test_perf['prec_10']:.4f}\n")

if __name__ == '__main__':
    results = train()
