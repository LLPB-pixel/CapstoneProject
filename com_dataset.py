import pandas as pd

for split in ["train", "val", "test"]:
    df = pd.read_csv(f"Data/{split}.csv")
    print(f"\n{split}:")
    print(df["label"].value_counts(dropna=False))
    print("Nulls:", df["label"].isna().sum())