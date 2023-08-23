import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

x_tensor = torch.tensor(x_train.values, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32)
dataset = TensorDataset(x_tensor, y_tensor)

class ScorePredictor(pl.LightningModule):
    def __init__(self, input_dim):
        super(ScorePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5), # 正則化のためのドロップアウト層を追加
            nn.Linear(32, 1)
        )
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.MSELoss()(y_pred, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5) # L2正則化を追加

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.MSELoss()(y_pred, y)
        pred = {'val_loss': loss}
        self.validation_step_outputs.append(pred)
        return pred

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        self.log('val_loss', avg_loss)


kf = KFold(n_splits=5, shuffle=True, random_state=42)


for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    train_data = torch.utils.data.Subset(dataset, train_idx)
    val_data = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_data, batch_size=32)
    val_loader = DataLoader(val_data, batch_size=32)

    model = ScorePredictor(input_dim=len(correct_features))
    
    # 早期停止コールバックを定義
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=True,
        mode='min'
    )
    trainer = pl.Trainer(max_epochs=5, callbacks=[early_stop_callback])
    trainer.fit(model, train_loader, val_loader)