import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ─── Load Data ───────────────────────────────────────────────────────────────
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

# Preserve actual test targets from the sample submission file
y_test = sample_sub['SalePrice'].copy()

# ─── Drop Uninformative Columns ──────────────────────────────────────────────
# Now also drop 'Id' early so it's not used as a feature
drop_cols = ['Id', 'Alley', 'PoolQC']
train = train.drop(columns=drop_cols)
test  = test.drop(columns=drop_cols)

# ─── Identify Feature Types ───────────────────────────────────────────────────
# Numeric features exclude the target now that 'Id' is dropped
numeric_feats = train.select_dtypes(include=[np.number]).columns.tolist()
numeric_feats.remove('SalePrice')

# Categorical features
categorical_feats = train.select_dtypes(include=['object']).columns.tolist()

# ─── Impute Numeric Missing Values ───────────────────────────────────────────
for col in numeric_feats:
    mean_val = train[col].mean()
    train[col] = train[col].fillna(mean_val)
    test[col] = test[col].fillna(mean_val)

# ─── Encode Categorical Features ─────────────────────────────────────────────
# Combine for consistent dummy encoding
train_features = train.drop(columns=['SalePrice'])
all_data = pd.concat([train_features, test], keys=['train', 'test'])

# Create dummy variables for categories (NaNs → all-zero columns)
all_data = pd.get_dummies(all_data, columns=categorical_feats)

# Split back into train/test
X_train = all_data.xs('train')
X_test  = all_data.xs('test')
y_train = train['SalePrice']

# Align columns between train and test
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# ─── Train Random Forest ──────────────────────────────────────────────────────
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=1,       # maximum depth of the tree(ı tried to keep it shallow) 
    min_samples_split=2,  # minimum samples to split an internal node
    min_samples_leaf=1,   # minimum samples at a leaf node
    max_features='sqrt',  # number of features to consider when looking for the best split
    bootstrap=True,       # whether to use bootstrap samples when building trees
    oob_score=True,      # whether to use out-of-bag samples to estimate the generalization accuracy
    n_jobs=-1,           # number of jobs to run in paralles
    random_state=42
)
model.fit(X_train, y_train)

# ─── Predict and Evaluate ─────────────────────────────────────────────────────
y_pred = model.predict(X_test)
log_rmse = np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(y_pred)))
print(f'Log RMSE: {log_rmse:.4f}')

# ─── Prepare Submission ───────────────────────────────────────────────────────
submission = sample_sub.copy()
submission['SalePrice'] = y_pred
submission.to_csv('submission3131.csv', index=False)

print("Submission file 'submission.csv' created.")
