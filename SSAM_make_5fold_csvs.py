import os
import pandas as pd

SSAM = "/mnt/hdd2/task2/ssam"
POOL_CSVS = [f"{SSAM}/All_nrrdname_train.csv", f"{SSAM}/All_nrrdname_val.csv"]
OUT_DIR = f"{SSAM}/folds"

# 0-indexed val patients per fold (authoritative: preprocess/split_dataset.log)
VAL_PATIENTS = {
    0: ['72', '29', '7', '52', '64'],
    1: ['11', '40', '68', '30', '74'],
    2: ['31', '41', '67', '57', '30'],
    3: ['50', '61', '12', '38', '29'],
    4: ['26', '14', '61', '7', '53'],
}
TEST_PATIENTS = ['19', '24', '71', '76', '78']

# Expected mask (row) counts from split_dataset.log, for validation
EXPECTED = {
    0: (177280, 21946),
    1: (179452, 19774),
    2: (177553, 21673),
    3: (175620, 23606),
    4: (177731, 21495),
}


def patient_of(png_path: str) -> str:
    # /mnt/hdd2/task2/{train,val}_set/{pid}/C?-XXXXX.png  -> pid
    return str(png_path).rstrip("/").split("/")[-2]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.concat([pd.read_csv(p) for p in POOL_CSVS], ignore_index=True)
    df["__pid"] = df["png_filename"].map(patient_of)

    pool_pids = sorted(df["__pid"].unique(), key=str)
    print(f"[pool] {len(df)} rows, {len(pool_pids)} patients: {pool_pids}")
    leaked = [p for p in pool_pids if p in TEST_PATIENTS]
    if leaked:
        print(f"  !! WARNING: test patients found in pool: {leaked}")

    all_ok = True
    for k in range(5):
        val_pids = set(VAL_PATIENTS[k])
        is_val = df["__pid"].isin(val_pids)
        val_df = df[is_val].drop(columns="__pid")
        train_df = df[~is_val].drop(columns="__pid")

        train_out = f"{OUT_DIR}/train_fold{k}.csv"
        val_out = f"{OUT_DIR}/val_fold{k}.csv"
        train_df.to_csv(train_out, index=False)
        val_df.to_csv(val_out, index=False)

        exp_tr, exp_va = EXPECTED[k]
        ok = (len(train_df) == exp_tr) and (len(val_df) == exp_va)
        all_ok &= ok
        tag = "OK" if ok else f"MISMATCH (expected train={exp_tr}, val={exp_va})"
        # patient-level sanity: train and val patients disjoint, val == requested
        got_val_pids = sorted(val_df["png_filename"].map(patient_of).unique(), key=str)
        print(f"[fold {k}] train={len(train_df)} val={len(val_df)}  {tag}")
        print(f"         val patients: {got_val_pids}")

    print("\nALL FOLDS MATCH split_dataset.log" if all_ok
          else "\n!! Some folds differ from split_dataset.log mask counts -- check above")


if __name__ == "__main__":
    main()
