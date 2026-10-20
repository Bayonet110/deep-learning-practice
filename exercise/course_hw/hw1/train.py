import random
from pathlib import Path

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED=42
BATCH_SIZE=64
LEARNING_RATE=1e-3
MAX_EPOCHS=500
VALID_RATIO=0.2
PATIENCE=40

ROOT=Path(__file__).resolve().parent
DATA_DIR=ROOT/"data"
OUT_DIR=ROOT/"outputs"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data():
    train_df=pd.read_csv(DATA_DIR/"covid_train.csv")
    test_df=pd.read_csv(DATA_DIR/"covid_test.csv")

    print("训练集形状：",train_df.shape)
    print("测试集形状：",test_df.shape)
    print("前五列：",train_df.columns[:5].tolist())
    print("后五列：",train_df.columns[-5:].tolist())

    target="tested_positive.2"

    features=[
        col for col in train_df.columns
        if col in test_df.columns
        and col not in ("id",target)
    ]

    X=train_df[features].to_numpy(dtype=np.float32)
    y=train_df[target].to_numpy(dtype=np.float32)
    X_test=test_df[features].to_numpy(dtype=np.float32)

    return X,y,X_test,features

def prepare_data(X,y,X_test):
    # 打乱索引
    rng=np.random.default_rng(SEED)
    indices=rng.permutation(len(X))

    # 划分训练集和验证集
    n_valid=int(len(X)*VALID_RATIO)

    valid_idx=indices[:n_valid]
    train_idx=indices[n_valid:]

    X_train=X[train_idx]
    y_train=y[train_idx]
    X_valid=X[valid_idx]
    y_valid=y[valid_idx]

    # 特征标准化
    mean=X_train.mean(axis=0)
    std=X_train.std(axis=0)

    std[std<1e-8]=1.0
    X_train=(X_train-mean)/std
    X_valid=(X_valid-mean)/std
    X_test=(X_test-mean)/std

    # 转换成pytorch TensorDataset
    train_dataset=TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )

    valid_dataset=TensorDataset(
        torch.from_numpy(X_valid),
        torch.from_numpy(y_valid)
    )

    # 创建DataLoader
    train_loader=DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    valid_loader=DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader,valid_loader,X_test,mean,std

class MLP(nn.Module):
    def __init__(self,input_dim):
        super().__init__()

        self.layers=nn.Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

    def forward(self,x):
        return self.layers(x).squeeze(-1)

def evaluate(model,valid_loader,criterion,device):
    model.eval()

    total_loss=0.0
    total_count=0

    with torch.no_grad():
        for xb,yb in valid_loader:
            xb=xb.to(device)
            yb=yb.to(device)
            pred=model(xb)
            loss=criterion(pred,yb)

            total_loss+=loss.item()*len(yb)
            total_count+=len(yb)

    return total_loss/total_count

def predict_test(model,X_test,device):
    model.eval()

    with torch.no_grad():
        X_test_tensor=torch.from_numpy(X_test).to(device)
        predictions=model(X_test_tensor).cpu().numpy()

    return predictions


def plot_loss(train_losses, valid_losses, save_path):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_losses,
        label="Train MSE"
    )

    plt.plot(
        epochs,
        valid_losses,
        label="Validation MSE"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    set_seed(SEED)

    device=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("当前使用设备：",device)

    if torch.cuda.is_available():
        print("GPU名称：",torch.cuda.get_device_name(0))

    X,y,X_test,features= load_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader,valid_loader,X_test,mean,std=prepare_data(
        X,y,X_test
    )

    print("训练集样本数：",len(train_loader.dataset))
    print("验证集样本数：",len(valid_loader.dataset))

    xb,yb=next(iter(train_loader))

    print("Batch特征现状：",xb.shape)
    print("Batch标签形状：",yb.shape)

    model=MLP(input_dim=X.shape[1]).to(device)
    criterion=nn.MSELoss()

    optimizer=torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_loss=float("inf")
    best_state=None
    best_epoch=0
    stale_epochs=0
    train_losses=[]
    valid_losses=[]

    process_bar=tqdm(range(MAX_EPOCHS),desc="Training")

    for epoch in process_bar:
        model.train()
        train_total=0.0
        train_count=0

        for xb,yb in train_loader:
            xb=xb.to(device)
            yb=yb.to(device)
            optimizer.zero_grad()

            pred=model(xb)
            loss=criterion(pred,yb)
            loss.backward()
            optimizer.step()

            train_total+=loss.item()*len(yb)
            train_count+=len(yb)

        train_mse=train_total/train_count
        valid_mse=evaluate(model,valid_loader, criterion,device)

        train_losses.append(train_mse)
        valid_losses.append(valid_mse)

        if valid_mse<best_loss-1e-6:
            best_loss=valid_mse
            best_state=copy.deepcopy(model.state_dict())
            best_epoch=epoch+1
            stale_epochs=0
        else:
            stale_epochs+=1

        process_bar.set_postfix(
            train_MSE=f"{train_mse:.5f}",
            valid_MSE=f"{valid_mse:.5f}",
            best_MSE=f"{best_loss:.5f}"
        )

        if stale_epochs>=PATIENCE:
            print(f"\nEarly Stopping：第{epoch+1}轮停止训练")
            break

    model.load_state_dict(best_state)
    print(f"最佳Epoch：{best_epoch}")
    print(f"最佳验证集MSE：{best_loss:.5f}")
    print(f"最佳验证集RMSE：{np.sqrt(best_loss):.5f}")

    torch.save(
        {
            "model_state_dict":{
                k:v.detach().cpu()
                for k,v in model.state_dict().items()
            },
            "input_dim":X.shape[1],
            "best_epoch":best_epoch,
            "best_valid_mse":best_loss,
            "mean":mean,
            "std":std,
            "features":features
        },
        OUT_DIR/"best_model.pt"
    )

    print("最佳模型已保存")

    # 测试集预测
    predictions=predict_test(model,X_test,device)
    test_df=pd.read_csv(DATA_DIR/"covid_test.csv")
    submission=pd.DataFrame({
        "id":test_df["id"],
        "tested_positive":predictions
    })

    submission.to_csv(
        OUT_DIR/"submission.csv",
        index=False
    )

    print("测试集预测完成，结果已保存")

    plot_loss(
        train_losses,
        valid_losses,
        OUT_DIR / "loss_curve.png"
    )

    print("损失曲线已保存")