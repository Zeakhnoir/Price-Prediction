# Price-Prediction
Kaggle Competition
# House Prices – Random-Forest Baseline

A lightweight baseline for Kaggle’s **House Prices: Advanced Regression Techniques** competition.  
The notebook cleans the data, trains a shallow Random Forest, and exports a ready-to-submit CSV in just a few minutes on any laptop.

---

## What the script does

1. **Load** `train.csv`, `test.csv`, `sample_submission.csv`.
2. **Prune noisy columns** & the ID (`Alley`, `PoolQC`, `Id`).
3. **Handle missing data**  
   * Numeric → mean imputation  
   * Categorical → `pd.get_dummies()` (one-hot, NaNs become all-zero vectors)
4. **Align feature space** between train & test (fills any gaps with 0).
5. **Train** a RandomForestRegressor  
   * 300 trees, shallow depth (`max_depth=1`) for speed & overfit control  
   * Out-of-bag score enabled for quick sanity check
6. **Predict** house prices for the test set.
7. **Evaluate** locally with Log-RMSE against the hidden targets in `sample_submission.csv`.
8. **Save** `submission3131.csv` ready for upload.

---

## Quick start

```bash
# 1. Clone repo & install deps
pip install -r requirements.txt

# 2. Drop Kaggle data files into the project root
#    ├── train.csv
#    ├── test.csv
#    └── sample_submission.csv

# 3. Run the script
python main.py

