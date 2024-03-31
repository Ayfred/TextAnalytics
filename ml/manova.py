import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

# Read the data
data = pd.read_csv("./dataset/preprocessed_data.csv")

# Shuffle the rows
data = data.sample(frac=1).reset_index(drop=True)

data = data.dropna()

# Keep only the first 10 columns
data = data.iloc[:, :10]

# Define the dependent variables
dependent_vars = data.columns.tolist()[1:]

# Define the independent variable
independent_var = "gender"

formula = " + ".join(dependent_vars) + " ~ " + independent_var

y, X = dmatrices(formula, data, return_type='dataframe')*
# Check for multicollinearity
vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)

# Perform MANOVA if multicollinearity is not severe
if vif["VIF"].max() < 10:  # You can adjust this threshold as needed
    maov = MANOVA(endog=y, exog=X)
    print(maov.mv_test())
else:
    print("Multicollinearity is too severe for MANOVA.")
