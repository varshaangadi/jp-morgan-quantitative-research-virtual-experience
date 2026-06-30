import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
# ---------------------------------------
# 1. LOAD DATA
# ---------------------------------------

df = pd.read_csv("data/Task 3 and 4_Loan_Data.csv")

# Display dataset structure
print("\nDataset Preview:\n", df.head())

# ---------------------------------------
# 2. SEPARATE FEATURES & TARGET
# ---------------------------------------

# Assuming the default column is named "default"
X = df.drop("default", axis=1)
y = df["default"]

# ---------------------------------------
# 3. TRAIN TEST SPLIT
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ---------------------------------------
# 4. TRAIN LOGISTIC REGRESSION MODEL
# ---------------------------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------------------------------
# 5. MODEL PERFORMANCE CHECK
# ---------------------------------------

pd_predictions = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, pd_predictions)
print("\nModel AUC Score:", auc)

# ---------------------------------------
# 6. EXPECTED LOSS FUNCTION
# ---------------------------------------

def expected_loss_function(borrower_features, loan_amount, recovery_rate=0.10):
    """
    borrower_features : list of input features in correct order
    loan_amount       : loan exposure (EAD)
    recovery_rate     : assumed recovery rate (default 10%)

    Returns:
        PD and Expected Loss
    """
    
    borrower_df = pd.DataFrame([borrower_features], columns=X.columns)
    prob_default = model.predict_proba(borrower_df)[0][1]
    expected_loss = loan_amount * prob_default * (1 - recovery_rate)

    return prob_default, expected_loss

# ---------------------------------------
# 7. TEST THE FUNCTION
# ---------------------------------------

# Example borrower (must match column order in your CSV)
sample_borrower = list(X.columns.map(lambda c: df[c].mean()))

loan_amount = 500000  # Example loan amount

pd_value, loss_value = expected_loss_function(sample_borrower, loan_amount)

print("\nPredicted Probability of Default (PD):", round(pd_value, 4))
print("Expected Loss (EL): ₹", round(loss_value, 2))
df["default"].value_counts().plot(kind="bar")
plt.title("Default vs Non-Default Borrowers")
plt.xlabel("Default")
plt.ylabel("Count")
plt.show()
