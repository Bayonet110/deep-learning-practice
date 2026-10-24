import csv
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent
IDS_PATH=BASE_DIR/"outputs"/"corpus_ids.txt"
VOCAB_PATH=BASE_DIR/"outputs"/"vocab.txt"
RESULT_PATH = BASE_DIR / "outputs" / "distinct_n_results.csv"

def load_token_ids(path):
    with path.open("r",encoding="utf-8") as file:
        return [int(token) for token in file.read().split()]

def calculate_distinct_n_ids(token_ids,n,vocab_size):
    if n<1:
        raise ValueError("N必须大于等于1")

    total=len(token_ids)-n+1

    if total<=0:
        return 0.0

    unique_ngrams=set()
    code=0

    for i in range(n):
        code=code*vocab_size+token_ids[i]

    unique_ngrams.add(code)

    highest_power=vocab_size**(n-1)

    for i in range(n,len(token_ids)):
        code=(
            (code%highest_power)*vocab_size+token_ids[i]
        )
        unique_ngrams.add(code)

    unique_count = len(unique_ngrams)

    print(f"Total {n}-grams: {total:,}")
    print(f"Unique {n}-grams: {unique_count:,}")

    return total,unique_count,unique_count/total


def main():
    if not IDS_PATH.is_file():
        raise FileNotFoundError(f"找不到编码文件：{IDS_PATH}")

    if not VOCAB_PATH.is_file():
        raise FileNotFoundError(f"找不到词表文件：{VOCAB_PATH}")

    with VOCAB_PATH.open("r",encoding="utf-8") as file:
        vocab_size=sum(1 for _ in file)

    token_ids=load_token_ids(IDS_PATH)

    print(f"Vocabulary Size: {vocab_size:,}")
    print(f"Total Tokens:{len(token_ids):,}")

    print("\n========== Task 2: Distinct-N ==========")

    results = []

    for n in range(1, 5):
        total, unique_count, score = calculate_distinct_n_ids(
            token_ids, n, vocab_size
        )

        results.append([n, total, unique_count, f"{score:.6f}"])

        print(f"Total {n}-grams: {total:,}")
        print(f"Unique {n}-grams: {unique_count:,}")
        print(f"Distinct-{n}: {score:.6f}")

    # 保存 CSV 实验结果
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULT_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["N", "Total N-grams", "Unique N-grams", "Distinct-N"])
        writer.writerows(results)

    print(f"\n实验结果已保存到：{RESULT_PATH}")

if __name__=="__main__":
    main()