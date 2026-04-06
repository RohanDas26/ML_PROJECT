import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Batch First: (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.lstm(x)
        # Take the output of the last time step
        last_out = out[:, -1, :] 
        return self.fc(last_out)


class SklearnLSTM(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible wrapper for the PyTorch LSTM.
    Allows easy integration into GridSearchCV and cross_val_score.
    """
    def __init__(
        self, 
        lookback=12, 
        hidden_size=64, 
        num_layers=2, 
        dropout=0.3, 
        lr=0.001, 
        epochs=100, 
        batch_size=32, 
        patience=15, 
        random_state=42
    ):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None

    def _create_sequences(self, X, y):
        """Converts 2D tabular data into 3D sequential tensors (batch, time, features)"""
        Xs, ys = [], []
        # We need at least 'lookback' steps to make a prediction
        for i in range(len(X) - self.lookback):
            Xs.append(X[i : i + self.lookback])
            ys.append(y[i + self.lookback])
        return np.array(Xs), np.array(ys)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # 1. Sequence Generation
        X_seq, y_seq = self._create_sequences(X, y)
        
        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_seq, dtype=torch.float32).view(-1, 1).to(self.device)
        
        # 2. DataLoaders
        # We use a 80/20 train/val split internally for Early Stopping
        val_size = int(len(X_t) * 0.2)
        train_size = len(X_t) - val_size
        
        if val_size == 0 or train_size == 0:
            # Fallback for very small datasets (just train on everything, no early stopping)
            train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=False)
            val_loader = None
        else:
            X_tr, y_tr = X_t[:train_size], y_t[:train_size]
            X_val, y_val = X_t[train_size:], y_t[train_size:]
            
            train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=self.batch_size, shuffle=False)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=self.batch_size, shuffle=False)
        
        # 3. Model Initialization
        input_size = X_t.shape[2]
        self.model = LSTMForecaster(
            input_size=input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            dropout=self.dropout
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        
        # 4. Training Loop with Early Stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
            
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        pred = self.model(batch_x)
                        loss = criterion(pred, batch_y)
                        val_loss += loss.item() * batch_x.size(0)
                
                val_loss /= len(X_val)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best weights
                    self.best_weights = self.model.state_dict()
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.patience:
                    # Early stopping triggered
                    self.model.load_state_dict(self.best_weights)
                    break
        
        if val_loader is not None and hasattr(self, 'best_weights'):
            self.model.load_state_dict(self.best_weights)
            
        return self

    def predict(self, X):
        self.model.eval()
        # Since prediction usually expects the same length array back from sklearn, 
        # but sequences consume 'lookback' rows, we need to handle the mismatch.
        # For evaluation/scoring, we just pad the beginning with the first known prediction.
        
        # Generate sequences
        # Note: If X is len N, X_seq will be len N-lookback
        Xs, _ = self._create_sequences(X, np.zeros(len(X)))
        
        # If the test set is smaller than lookback, we can't sequence it.
        if len(Xs) == 0:
            return np.zeros(len(X))
            
        X_t = torch.tensor(Xs, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            preds = self.model(X_t).cpu().numpy().ravel()
            
        # Pad the missing 'lookback' predictions at the start with the first valid prediction
        # to maintain shape parity for sklearn metrics
        pad = np.full(self.lookback, preds[0])
        return np.concatenate([pad, preds])
