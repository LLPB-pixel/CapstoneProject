import pandas as pd

VALID_LABELS = {"bad", "good"}  # descartamos también 'unknown'

for split in ["train", "val", "test"]:
    path = f"Data/{split}.csv"
    df = pd.read_csv(path)
    
    original_len = len(df)
    # Quedarse solo con filas cuya label es válida
    df = df[df["label"].isin(VALID_LABELS)].copy()
    
    # Mapear a 0/1
    df["label"] = df["label"].map({"good": 0, "bad": 1})
    
    df.to_csv(path, index=False)
    print(f"{split}: {original_len} → {len(df)} filas ({original_len - len(df)} eliminadas)")