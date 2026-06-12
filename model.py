import kagglehub
import pandas as pd
import numpy as np
import re
import ast
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve, KFold
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================================
# 0. LOAD DATA
# =====================================================================
path = kagglehub.dataset_download("sagyamthapa/nepali-housing-price-dataset")
print("Path to dataset files:", path)

file_path = r"C:\Users\HP\.cache\kagglehub\datasets\sagyamthapa\nepali-housing-price-dataset\versions\1\2020-4-27.csv"

df = pd.read_csv(file_path)


# =====================================================================
# 1. AREA / BUILD AREA -> SQUARE FEET
# =====================================================================
def convert_area_to_sqft(area_str):
    if pd.isna(area_str) or not re.search(r'\d', str(area_str)):
        return None

    area_str = str(area_str).lower().strip()

    try:
        if '-' in area_str:
            m = re.findall(r'[\d\.]+(?:-[\d\.]+)+', area_str)
            if not m:
                return None
            parts = [float(x) for x in m[0].split('-')]
            if len(parts) == 4:
                r, a, p, d = parts
                return r * 5476 + a * 342.25 + p * 85.56 + d * 21.39
            elif len(parts) == 3:
                r, a, p = parts
                return r * 5476 + a * 342.25 + p * 85.56

        dot_match = re.fullmatch(r'(\d+(?:\.\d+){2,3})\s*aana', area_str)
        if dot_match:
            parts = [float(x) for x in dot_match.group(1).split('.')]
            if len(parts) == 4:
                r, a, p, d = parts
                return r * 5476 + a * 342.25 + p * 85.56 + d * 21.39
            elif len(parts) == 3:
                r, a, p = parts
                return r * 5476 + a * 342.25 + p * 85.56

        haat_match = re.search(
            r'(\d+(?:\.\d+)?)\s*[*x/]\s*(\d+(?:\.\d+)?)\s*haat', area_str
        )
        if haat_match:
            length = float(haat_match.group(1))
            width = float(haat_match.group(2))
            return length * width * 2.25

        frac_match = re.search(r'(\d+)\s*/\s*(\d+)', area_str)
        if frac_match:
            val = float(frac_match.group(1)) / float(frac_match.group(2))
        else:
            num_match = re.search(r'[\d\.]+', area_str)
            val = float(num_match.group()) if num_match else None

        if val is not None:
            if 'ropani' in area_str:
                return val * 5476
            elif 'aana' in area_str or 'ana' in area_str:
                return val * 342.25
            elif 'dhur' in area_str:
                return val * 182.25
            elif 'kattha' in area_str:
                return val * 3645
            elif 'bigha' in area_str:
                return val * 72900
            elif 'haat' in area_str:
                return val * 2.25
            elif 'sq. meter' in area_str or 'meter' in area_str:
                return val * 10.76
            elif 'sq. feet' in area_str or 'feet' in area_str:
                return val

    except Exception:
        return None

    return None


df['Area_SqFt'] = df['Area'].apply(convert_area_to_sqft)
df['Build_Area_SqFt'] = df['Build Area'].apply(convert_area_to_sqft)
df['Has_Build_Area'] = df['Build_Area_SqFt'].notna().astype(int)


# =====================================================================
# 2. ROAD WIDTH -> NUMERIC FEET
# =====================================================================
df['Road_Width_Feet'] = (
    df['Road Width'].astype(str).str.extract(r'(\d+(?:\.\d+)?)')[0].astype(float)
)


# =====================================================================
# 3. ROAD TYPE -> FILL MISSING
# =====================================================================
df['Road Type'] = df['Road Type'].astype(str).str.strip()
df['Road Type'] = df['Road Type'].replace({'nan': 'Unknown', 'N/A': 'Unknown'})
df['Road Type'] = df['Road Type'].fillna('Unknown')


# =====================================================================
# 4. VIEWS -> NUMERIC  (e.g. "2.6K" -> 2600)
# =====================================================================
def parse_views(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip().upper()
    if val.endswith('K'):
        return float(val[:-1]) * 1000
    if val.endswith('M'):
        return float(val[:-1]) * 1_000_000
    try:
        return float(val)
    except ValueError:
        return np.nan


df['Views_Num'] = df['Views'].apply(parse_views)


# =====================================================================
# 5. POSTED -> "DAYS SINCE POSTED"
# =====================================================================
def parse_posted(val):
    if pd.isna(val):
        return np.nan
    val = str(val).lower().strip()
    m = re.match(r'(\d+)\s*(hour|day|week|month|year)s?\s*ago', val)
    if not m:
        return np.nan
    num, unit = float(m.group(1)), m.group(2)
    multiplier = {'hour': 1 / 24, 'day': 1, 'week': 7, 'month': 30, 'year': 365}
    return num * multiplier[unit]


df['Days_Since_Posted'] = df['Posted'].apply(parse_posted)


# =====================================================================
# 6. AMENITIES -> #AMENITIES + KEY AMENITY FLAGS
# =====================================================================
def parse_amenities(val):
    if pd.isna(val):
        return []
    try:
        items = ast.literal_eval(val)
        if isinstance(items, list):
            return items
    except (ValueError, SyntaxError):
        pass
    return []


amenity_lists = df['Amenities'].apply(parse_amenities)
df['Num_Amenities'] = amenity_lists.apply(len)

key_amenities = ['Parking', 'Garden', 'Garage', 'Solar Water', 'Water Tank']
for amenity in key_amenities:
    col_name = 'Amenity_' + amenity.replace(' ', '_')
    df[col_name] = amenity_lists.apply(lambda lst, a=amenity: int(a in lst))


# =====================================================================
# 7. CITY -> TOP CITIES + "OTHER"
# =====================================================================
top_cities = df['City'].value_counts().nlargest(10).index
df['City_Grouped'] = df['City'].where(df['City'].isin(top_cities), 'Other')


# =====================================================================
# 8. OUTLIER HANDLING (clip rather than drop, to keep dataset size)
# =====================================================================
for col, lower_q, upper_q in [
    ('Price', 0.01, 0.99),
    ('Bedroom', 0.0, 0.99),
    ('Bathroom', 0.0, 0.99),
]:
    lower = df[col].quantile(lower_q)
    upper = df[col].quantile(upper_q)
    df[col] = df[col].clip(lower=lower, upper=upper)


# =====================================================================
# 9. MISSING NUMERIC DATA -> MEDIAN IMPUTATION
# =====================================================================
for col in ['Floors', 'Area_SqFt', 'Build_Area_SqFt', 'Road_Width_Feet',
            'Views_Num', 'Days_Since_Posted']:
    df[col] = df[col].fillna(df[col].median())


# =====================================================================
# 10. ONE-HOT ENCODING
# =====================================================================
categorical_cols = ['Face', 'Road Type', 'City_Grouped']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# =====================================================================
# 11. DROP UNUSED COLUMNS
# =====================================================================
drop_cols = [
    'Title', 'Address', 'Road', 'Road Width', 'Year',
    'Area', 'Build Area', 'Posted', 'Views', 'Amenities', 'City',
]
df_final = df_encoded.drop(columns=drop_cols)


# =====================================================================
# 12. TRAIN / VALIDATION SPLIT
#     Target is log1p(Price) since price is heavily right-skewed -
#     this stabilizes variance and makes linear regression more
#     appropriate.
# =====================================================================
X = df_final.drop(columns=['Price'])
y = np.log1p(df_final['Price'])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =====================================================================
# 13. MODEL: RIDGE REGRESSION (L2-REGULARIZED LINEAR REGRESSION)
#     RidgeCV picks the best regularization strength (alpha) via
#     built-in cross-validation on the training set only.
# =====================================================================
alphas = np.logspace(-3, 3, 13)  # 0.001 ... 1000

model = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('ridge', RidgeCV(alphas=alphas, cv=5)),
])

model.fit(X_train, y_train)

best_alpha = model.named_steps['ridge'].alpha_
print(f"Best regularization strength (alpha): {best_alpha:.4f}")


# =====================================================================
# 14. EVALUATION
# =====================================================================
def evaluate(model, X, y, label):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, preds)
    print(f"{label:>12} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | R^2: {r2:.4f}")
    return mse, rmse, r2


print("\nModel performance (target = log1p(Price)):")
evaluate(model, X_train, y_train, "Train")
evaluate(model, X_val, y_val, "Validation")


# =====================================================================
# 15. LEARNING CURVE
#     Shows training vs. validation error as the training set size
#     grows - diagnoses under/overfitting.
# =====================================================================
train_sizes, train_scores, val_scores = learning_curve(
    estimator=model,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='neg_mean_squared_error',
    n_jobs=-1,
)

train_mse = -train_scores.mean(axis=1)
val_mse = -val_scores.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mse, 'o-', color='tab:blue', label='Training MSE')
plt.plot(train_sizes, val_mse, 'o-', color='tab:orange', label='Validation MSE')
plt.xlabel('Training set size')
plt.ylabel('Mean Squared Error (log1p(Price))')
plt.title('Learning Curve - Ridge Regression')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=150)
plt.show()

print("\nLearning curve saved to 'learning_curve.png'")