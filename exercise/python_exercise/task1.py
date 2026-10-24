from collections import Counter
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent
DATA_PATH=BASE_DIR/"data"/"text8"
OUTPUT_DIR=BASE_DIR/"outputs"
VOCAB_PATH=OUTPUT_DIR/"vocab.txt"
IDS_PATH=OUTPUT_DIR/"corpus_ids.txt"

def load_corpus(path:Path):
    with path.open("r",encoding="utf-8") as file:
        return file.read().split()

def build_vocabulary(words):
    counts=Counter(words)
    sorted_words=counts.most_common()
    word_to_id={word: index for index,(word,_) in enumerate(sorted_words)}

    with VOCAB_PATH.open("w",encoding="utf-8") as file:
        for word_id,(word,frequency) in enumerate(sorted_words):
            file.write(f"{word_id}\t{word}\t{frequency}\n")

    assert sum(counts.values())==len(words)
    print(f"词表大小：{len(word_to_id):,}")
    print("高频次 Top 10 (ID / 单词 / 频次)：")
    for word,frequency in sorted_words[:10]:
        print(word_to_id[word],word,frequency)

    return word_to_id

def encode_word(word,word_to_id,position):
    if word not in word_to_id:
        raise ValueError(f"OOV:位置{position}的单词‘{word}'不在词表中")
    return word_to_id[word]

def encode_corpus(words,word_to_id):
    preview=[]
    buffer=[]
    batch_size=10000

    with IDS_PATH.open("w",encoding="utf-8") as file:
        for position,word in enumerate(words):
            word_id=encode_word(word,word_to_id, position)
            if position<30:
                preview.append(word_id)
            buffer.append(str(word_id))
            if len(buffer)==batch_size:
                file.write(" ".join(buffer) + " ")
                buffer.clear()
        if buffer:
            file.write(" ".join(buffer) + "\n")

    print(f"已编码 Token 数：{len(words):,}")
    print("前30个token：",words[:30])
    print("前30个ID：",preview)

def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"找不到数据集：{DATA_PATH}")
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    words=load_corpus(DATA_PATH)
    print(f"语料Token总数：{len(words):,}")

    print("\n========== Task 1.1:构建词表 ==========")
    word_to_id=build_vocabulary(words)
    print("词表输出路径：",VOCAB_PATH)

    print("\n========== Task 1.2: 语料编码 ==========")
    encode_corpus(words,word_to_id)
    print("编码输出路径：",IDS_PATH)
    print("\nTask 1完成")

if __name__=="__main__":
    main()
