from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'outputs'
MODEL_DIR = OUTPUT_DIR / 'models'
SEED = 42
EPOCHS = 200
BATCH_SIZE = 16

def load_data(batch_size=16,seed=SEED):
    # 读取数据
    train_data=pd.read_csv(BASE_DIR/"iris_training.csv",skiprows=1,header=None)
    test_data=pd.read_csv(BASE_DIR/"iris_test.csv",skiprows=1,header=None)

    # 分离特征和标签
    X_train=train_data.iloc[:,:4].values
    y_train=train_data.iloc[:,4].values
    X_test=test_data.iloc[:,:4].values
    y_test=test_data.iloc[:,4].values

    # 标准化
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    # 转换为Pytroch Tensor
    X_train=torch.tensor(X_train,dtype=torch.float32).unsqueeze(1)
    X_test=torch.tensor(X_test,dtype=torch.float32).unsqueeze(1)
    y_train=torch.tensor(y_train,dtype=torch.long)
    y_test=torch.tensor(y_test,dtype=torch.long)

    generator=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(TensorDataset(X_train,y_train),batch_size=batch_size,shuffle=True,generator=generator)
    test_loader=DataLoader(TensorDataset(X_test,y_test),batch_size=batch_size,shuffle=False)

    return train_loader,test_loader

class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 卷积层
        self.conv1=nn.Conv1d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=1
        )

        # 激活函数
        self.relu=nn.ReLU()

        # 最大池化层
        self.pool=nn.MaxPool1d(kernel_size=2)

        # 全连接分类层
        self.fc=nn.Linear(8*2,3)

    def forward(self,x):
        x=self.pool(self.relu(self.conv1(x)))
        return self.fc(torch.flatten(x,start_dim=1))

class VGG(nn.Module):
    def __init__(self):
        super().__init__()

        self.features=nn.Sequential(
            nn.Conv1d(1,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv1d(16,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv1d(32,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.classifier=nn.Linear(32,3)

    def forward(self,x):
        return self.classifier(torch.flatten(self.features(x),start_dim=1))

def train_model(model, train_loader, criterion, optimizer, epochs=EPOCHS, label='Training'):
    progress_bar=tqdm(range(1,epochs+1),desc=label)
    history = []
    for epoch in progress_bar:
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            correct += (output.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        avg_loss = total_loss / total
        accuracy = correct / total
        history.append({'epoch': epoch, 'train_loss': avg_loss, 'train_accuracy': accuracy})
        progress_bar.set_postfix(
            loss=f"{avg_loss:.4f}",
            train_acc=f"{accuracy:.4f}"
        )

        if epoch in (1,50,100,150,epochs):
            tqdm.write(
                f"{label}:Epoch {epoch}/{epochs}, "
                f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}"
            )
    return history

def evaluate_model(model,data_loader,criterion):
    model.eval()

    total_loss=0.0
    correct=0
    total=0

    with torch.no_grad():
        for X,y in data_loader:
            output=model(X)
            loss=criterion(output,y)
            total_loss+=loss.item()*X.size(0)
            predicted=output.argmax(dim=1)
            correct+=(predicted==y).sum().item()
            total+=y.size(0)

    avg_loss=total_loss/total
    accuracy=correct/total

    return avg_loss,accuracy

def run_experiment(model_name,optimizer_name,learning_rate):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_loader,test_loader=load_data(BATCH_SIZE,seed=SEED)
    model=BasicCNN() if model_name=="BasicCNN" else VGG()
    criterion=nn.CrossEntropyLoss()
    optimizer_type=torch.optim.SGD if optimizer_name=="SGD" else torch.optim.Adam
    optimizer=optimizer_type(model.parameters(),lr=learning_rate)

    title=f'{model_name} + {optimizer_name}'
    print(f"\n========== {title} (lr={learning_rate})==========")
    history=train_model(model,train_loader, criterion, optimizer,EPOCHS,title)
    train_loss,train_acc=evaluate_model(model,train_loader,criterion)
    test_loss,test_acc=evaluate_model(model,test_loader,criterion)
    print(f'Train Loss:{train_loss:.4f} | Train Accuracy:{train_acc:.4%}')
    print(f'Test Loss:{test_loss:.4f} | Test Accuracy:{test_acc:.4%}')

    model_file=MODEL_DIR/f'{model_name.lower()}_{optimizer_name.lower()}.pth'
    torch.save(model.state_dict(),model_file)

    for record in history:
        record.update({'model':model_name,'optimizer':optimizer_name,'lr':learning_rate})
    result = {'model': model_name, 'optimizer': optimizer_name, 'lr': learning_rate,
              'epochs': EPOCHS, 'train_loss': train_loss, 'train_accuracy': train_acc,
              'test_loss': test_loss, 'test_accuracy': test_acc}

    return result,history

def save_outputs(results, histories):
    results_df = pd.DataFrame(results)
    history_df = pd.DataFrame(histories)
    results_df.to_csv(OUTPUT_DIR / 'experiment_results.csv', index=False, encoding='utf-8-sig')
    history_df.to_csv(OUTPUT_DIR / 'training_history.csv', index=False, encoding='utf-8-sig')
    lines = ['Iris CNN classification - four experiments', 'Iris: 120 training / 30 test',
             'VGG here is a 1D VGG-style adaptation, not VGG16.', '']
    for r in results:
        lines.extend([f"========== {r['model']} + {r['optimizer']} ==========" ,
                      f"Learning Rate: {r['lr']}", f"Epochs: {r['epochs']}",
                      f"Train Loss: {r['train_loss']:.4f}",
                      f"Train Accuracy: {r['train_accuracy']:.4%}",
                      f"Test Loss: {r['test_loss']:.4f}",
                      f"Test Accuracy: {r['test_accuracy']:.4%}", ''])
    (OUTPUT_DIR / 'results.txt').write_text('\n'.join(lines), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for (model_name, optimizer_name), group in history_df.groupby(['model', 'optimizer'], sort=False):
        name = f'{model_name} + {optimizer_name}'
        axes[0].plot(group['epoch'], group['train_loss'], label=name)
        axes[1].plot(group['epoch'], group['train_accuracy'], label=name)
    axes[0].set(title='Training Loss', xlabel='Epoch', ylabel='Cross Entropy Loss')
    axes[1].set(title='Training Accuracy', xlabel='Epoch', ylabel='Accuracy', ylim=(0, 1.02))
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'training_curves.png', dpi=180)
    plt.close(fig)
    print('\n========== Summary ==========')
    print(results_df.to_string(index=False, formatters={
        'train_loss': '{:.4f}'.format, 'test_loss': '{:.4f}'.format,
        'train_accuracy': '{:.2%}'.format, 'test_accuracy': '{:.2%}'.format}))
    print(f'\nOutputs saved in: {OUTPUT_DIR}')


if __name__=="__main__":
    OUTPUT_DIR.mkdir(exist_ok=True,parents=True)
    MODEL_DIR.mkdir(exist_ok=True,parents=True)
    configs=[
        ('BasicCNN','SGD',0.05),
        ('BasicCNN','Adam',0.001),
        ('VGG','SGD',0.01),
        ('VGG','Adam',0.001)
    ]

    results,histories=[],[]
    for model_name,optimizer_name,lr in configs:
        result,history=run_experiment(model_name,optimizer_name,lr)
        results.append(result)
        histories.extend(history)

    save_outputs(results, histories)