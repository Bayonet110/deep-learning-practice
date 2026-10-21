import csv
import json
import random
import time
from pathlib import Path
from collections import Counter

from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
RANDOM_SEED=42

MODEL_TYPES=["RNN","GRU","LSTM"]
OPTIMIZER_TYPES=["SGD","Adam"]

PAD_TOKEN="<PAD>"
UNK_TOKEN="<UNK>"
PAD_IDX=0
UNK_IDX=1
MAX_VOCAB_SIZE=20000
MAX_LENGTH=100
BATCH_SIZE=64

EMBEDDING_DIM=128
HIDDEN_DIM=128
LEARNING_RATE_SGD=0.1
LEARNING_RATE_ADAM=0.001

EPOCHS=5

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(split):
    texts=[]
    labels=[]

    for label in [0,1]:
        file_path=DATA_DIR/f"sentiment.{split}.{label}"

        with open(file_path,"r",encoding="utf-8") as f:
            for line in f:
                text=line.strip()

                if not text:
                    continue

                texts.append(text)
                labels.append(label)

    return texts,labels

def tokenize(text):
    return text.lower().split()

def build_vocab(texts,max_vocab_size=MAX_VOCAB_SIZE):
    counter=Counter()

    for text in texts:
        tokens=tokenize(text)
        counter.update(tokens)

    vocab={
        PAD_TOKEN:PAD_IDX,
        UNK_TOKEN:UNK_IDX
    }

    most_common_words=counter.most_common(max_vocab_size-2)

    for word,_ in most_common_words:
        if word not in vocab:
            vocab[word]=len(vocab)

    return vocab

def encode_text(text,vocab):
    tokens=tokenize(text)

    token_ids=[
        vocab.get(token,UNK_IDX)
        for token in tokens
    ]

    return token_ids

class YelpDataset(Dataset):
    def __init__(self,texts,labels,vocab,max_length=MAX_LENGTH):
        self.texts=texts
        self.labels=labels
        self.vocab=vocab
        self.max_length=max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text=self.texts[idx]
        label=self.labels[idx]

        token_ids=encode_text(text,self.vocab)
        token_ids=token_ids[:self.max_length]

        if not token_ids:
            token_ids=[UNK_IDX]

        return (
            torch.tensor(token_ids,dtype=torch.long),
            torch.tensor(label,dtype=torch.long)
        )

def collate_fn(batch):
    sequences,labels=zip(*batch)
    lengths=torch.tensor(
        [len(seq) for seq in sequences],
        dtype=torch.long
    )

    padded_sequences=pad_sequence(sequences,batch_first=True,padding_value=PAD_IDX)
    labels=torch.stack(labels)
    return padded_sequences,lengths,labels

def create_dataloader(texts,labels,vocab,batch_size=BATCH_SIZE,shuffle=False):
    dataset=YelpDataset(texts=texts,labels=labels,vocab=vocab,max_length=MAX_LENGTH)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,collate_fn=collate_fn)

class RNNClassifier(nn.Module):
    def __init__(self,vocab_size,embedding_dim=128,hidden_dim=128,num_classes=2,model_type="RNN"):
        super().__init__()

        self.model_type=model_type.upper()

        # Embedding 层
        self.embedding=nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_IDX
        )

        # 循环神经网络层
        rnn_types={
            "RNN":nn.RNN,
            "GRU":nn.GRU,
            "LSTM":nn.LSTM
        }

        if self.model_type not in rnn_types:
            raise ValueError(f"不支持的模型类型：{model_type}")

        self.rnn = rnn_types[self.model_type](
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # 全连接分类层
        self.fc=nn.Linear(hidden_dim,num_classes)

    def forward(self, x, lengths):
        embedded=self.embedding(x)

        packed=pack_padded_sequence(embedded,lengths.cpu(),batch_first=True,enforce_sorted=False)

        if self.model_type=="LSTM":
            _,(hidden,cell)=self.rnn(packed)
        else:
            _,hidden=self.rnn(packed)

        final_hidden=hidden[-1]
        logits=self.fc(final_hidden)

        return logits

def train_model(model,dataLoader,criterion,optimizer,device):
    model.train()

    total_loss=0.0
    total_correct=0
    total_samples=0
    progress_bar=tqdm(dataLoader,desc="Training",leave=False)

    for inputs,lengths,labels in progress_bar:
        inputs=inputs.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()
        outputs=model(inputs,lengths)
        loss=criterion(outputs,labels)
        loss.backward()

        # 梯度裁剪，缓解RNN梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)

        optimizer.step()

        batch_size=labels.size(0)
        total_loss+=loss.item()*batch_size
        total_correct+=(outputs.argmax(dim=1)==labels).sum().item()
        total_samples+=batch_size

        progress_bar.set_postfix(
            loss=f"{total_loss/total_samples:.4f}",
            acc=f"{total_correct/total_samples:.4f}"
        )

    avg_loss=total_loss/total_samples
    accuracy=total_correct/total_samples

    return avg_loss,accuracy

def evaluate(model,dataloader,criterion,device):
    model.eval()
    total_loss,total_correct,total_samples=0.0,0,0

    with torch.no_grad():
        for inputs,lengths,labels in dataloader:
            inputs=inputs.to(device)
            labels=labels.to(device)

            outputs=model(inputs,lengths)
            loss=criterion(outputs,labels)
            batch_size=labels.size(0)

            total_loss+=loss.item()*batch_size
            total_correct+=(outputs.argmax(dim=1)==labels).sum().item()
            total_samples+=batch_size

    avg_loss=total_loss/total_samples
    accuracy=total_correct/total_samples

    return avg_loss,accuracy

def save_history(history,output_dir):
    file_path=output_dir/"training_history.csv"

    filenames=["epoch","train_loss","train_acc","dev_loss","dev_acc"]

    with open(file_path,"w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=filenames)
        writer.writeheader()
        writer.writerows(history)

def plot_history(history, output_dir):
    """绘制 Loss 和 Accuracy 曲线"""

    epochs = [item["epoch"] for item in history]

    train_loss = [item["train_loss"] for item in history]
    dev_loss = [item["dev_loss"] for item in history]

    train_acc = [item["train_acc"] for item in history]
    dev_acc = [item["dev_acc"] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss 曲线
    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[0].plot(epochs, dev_loss, marker="o", label="Dev Loss")

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy 曲线
    axes[1].plot(epochs, train_acc, marker="o", label="Train Acc")
    axes[1].plot(epochs, dev_acc, marker="o", label="Dev Acc")

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()

    fig.savefig(
        output_dir / "training_curves.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

def save_summary(results):
    if not results:
        return

    file_path=OUTPUT_DIR/"experiment_summary.csv"

    with open(file_path,"w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def run_experiment(model_type,optimizer_type,train_loader,dev_loader,test_loader,vocab_size,device,epochs=EPOCHS):
    set_seed()
    experiment_name=f"{model_type}_{optimizer_type}"
    experiment_dir=OUTPUT_DIR/experiment_name
    experiment_dir.mkdir(parents=True,exist_ok=True)
    print(f"\n========== {experiment_name} ==========")

    start_time=time.perf_counter()

    model=RNNClassifier(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        model_type=model_type
    ).to(device)

    criterion=nn.CrossEntropyLoss()

    if optimizer_type=="SGD":
        learning_rate=LEARNING_RATE_SGD
        optimizer=torch.optim.SGD(model.parameters(),lr=LEARNING_RATE_SGD)
    elif optimizer_type=="Adam":
        learning_rate=LEARNING_RATE_ADAM
        optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE_ADAM)
    else:
        raise ValueError(f"不支持的优化器：{optimizer_type}")

    history=[]
    log_lines=[]
    best_dev_loss=float("inf")
    best_epoch=0
    best_model_path=experiment_dir/"best_model.pt"

    def write_log(message):
        print(message)
        log_lines.append(message)

        with open(experiment_dir/"training_log.txt","w",encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")

    write_log(f"Experiment: {experiment_name}")
    write_log(f"Device: {device}")
    write_log(f"Train samples: {len(train_loader.dataset)}")
    write_log(f"Dev samples: {len(dev_loader.dataset)}")
    write_log(f"Test samples: {len(test_loader.dataset)}")
    write_log(f"Learning rate: {learning_rate}")

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        train_loss,train_acc=train_model(model,train_loader,criterion, optimizer, device)
        dev_loss,dev_acc=evaluate(model,dev_loader,criterion, device)
        epoch_time = time.perf_counter() - epoch_start

        history.append({
            "epoch":epoch+1,
            "train_loss":train_loss,
            "train_acc":train_acc,
            "dev_loss":dev_loss,
            "dev_acc":dev_acc
        })

        message=(
            f"Epoch[{epoch+1}/{epochs}] "
            f"Train Loss:{train_loss:.4f}, "
            f"Train Acc:{train_acc:.4f} | "
            f"Dev Loss:{dev_loss:.4f}, "
            f"Dev Acc:{dev_acc:.4f} | "
            f"Time:{epoch_time:.1f}s"
        )

        write_log(message)

        if dev_loss<best_dev_loss:
            best_dev_loss=dev_loss
            best_epoch=epoch+1

            torch.save(model.state_dict(),best_model_path)

            write_log(f"  Saved best model at Epoch {best_epoch}")

        save_history(history,experiment_dir)

    model.load_state_dict(torch.load(best_model_path,map_location=device,weights_only=True))
    train_loss,train_acc=evaluate(model,train_loader,criterion, device)
    dev_loss,dev_acc=evaluate(model,dev_loader,criterion, device)
    test_loss,test_acc=evaluate(model,test_loader,criterion, device)

    total_time=time.perf_counter()-start_time

    # 最终结果
    result = {
        "model": model_type,
        "optimizer": optimizer_type,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "train_samples": len(train_loader.dataset),
        "train_loss": round(train_loss, 6),
        "train_acc": round(train_acc, 6),
        "dev_loss": round(dev_loss, 6),
        "dev_acc": round(dev_acc, 6),
        "test_loss": round(test_loss, 6),
        "test_acc": round(test_acc, 6),
        "time_seconds": round(total_time, 2)
    }

    write_log(f"\nBest Epoch: {best_epoch}")
    write_log(
        f"Final Train Loss: {train_loss:.4f}, "
        f"Train Acc: {train_acc:.4f}"
    )
    write_log(
        f"Final Dev Loss: {dev_loss:.4f}, "
        f"Dev Acc: {dev_acc:.4f}"
    )
    write_log(
        f"Final Test Loss: {test_loss:.4f}, "
        f"Test Acc: {test_acc:.4f}"
    )
    write_log(f"Total Time: {total_time:.1f}s")

    # 保存训练曲线
    plot_history(history, experiment_dir)

    # 保存最终指标及超参数
    metrics = {
        **result,
        "vocab_size": vocab_size,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "seed": RANDOM_SEED
    }

    with open(
            experiment_dir / "metrics.json",
            "w",
            encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"\n实验结果已保存至：{experiment_dir}")

    return result

def main():
    set_seed()
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    # 读取数据集
    print("========== 加载 Yelp 数据集 ==========")
    train_texts,train_labels=load_data("train")
    dev_texts,dev_labels=load_data("dev")
    test_texts,test_labels=load_data("test")

    print(f"训练集样本数：{len(train_texts)}")
    print(f"验证集样本数：{len(dev_texts)}")
    print(f"测试集样本数：{len(test_texts)}")

    # 构建词表
    print("\n========== 构建词表 ==========")
    vocab=build_vocab(train_texts)
    print(f"词表大小：{len(vocab)}")

    # 测试文本编码
    sample_text=train_texts[0]
    print(f"\n原始文本：{sample_text}")
    print(f"分词结果：{tokenize(sample_text)}")
    print(f"编码结果：{encode_text(sample_text,vocab)}")
    print(f"未知单词编码：{encode_text('thisisaveryrareword',vocab)}")

    # 创建DataLoader
    print("\n========== 创建 DataLoader ==========")
    train_loader=create_dataloader(train_texts,train_labels,vocab,shuffle=True)
    dev_loader=create_dataloader(dev_texts,dev_labels,vocab)
    test_loader=create_dataloader(test_texts,test_labels,vocab)

    print(f"训练集 Batch 数量：{len(train_loader)}")
    print(f"验证集 Batch 数量：{len(dev_loader)}")
    print(f"测试集 Batch 数量：{len(test_loader)}")

    # 训练
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n当前实验设备：{device}")

    results=[]
    for model_type in MODEL_TYPES:
        for optimizer_type in OPTIMIZER_TYPES:
            result=run_experiment(
                model_type=model_type,
                optimizer_type=optimizer_type,
                train_loader=train_loader,
                dev_loader=dev_loader,
                test_loader=test_loader,
                vocab_size=len(vocab),
                device=device,
                epochs=EPOCHS
            )

            results.append(result)
            save_summary(results)

    print("\n========== 实验结果汇总 ==========")
    for result in results:
        print(
            f"{result['model']:5s} + "
            f"{result['optimizer']:4s} | "
            f"Train Acc: {result['train_acc']:.4f} | "
            f"Dev Acc: {result['dev_acc']:.4f} | "
            f"Test Acc: {result['test_acc']:.4f}"
        )

    print(f"\n全部实验完成！结果保存在：{OUTPUT_DIR}")


if __name__=="__main__":
    main()