# train_models.py

import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ─── 1. Load & Clean ───────────────────────────────────────────────────────────
df = pd.read_csv('INDICATORS.csv')
if 'label success' in df.columns:
    df = df.rename(columns={'label success':'success'})

df = df.rename(columns={
    'gdp_growth_rate (billions of us dollars)':    'gdp_growth',
    'gdp (percent change)':                        'gdp_total',
    'gdp_per_capita (us dollar per capita)':       'gdp_pc',
    'rail lines (total route-km)':                 'rail_km',
    'hsr_network_length':                          'hsr_km',
    'total_population (thousands)':                'pop_thousands',
    'population_density (people/square kilometer)': 'pop_density',
    'urbanization_rate':                           'urban_rate'
})

# coerce to numeric
for c in ['rail_km','hsr_km','gdp_growth','urban_rate']:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace(',',''),
                          errors='coerce')

# ─── 2. Feature Engineering ─────────────────────────────────────────────────────
def make_features(X):
    X = X.copy()
    X['hsr_per_thousand']           = X['hsr_km'] / X['pop_thousands']
    X['rail_per_thousand']          = X['rail_km'] / X['pop_thousands']
    X['hsr_share']                  = X['hsr_km'] / X['rail_km']
    X['gdp_pc_urban_interaction']   = X['gdp_pc'] * X['urban_rate']
    X['density_growth_interaction'] = X['pop_density'] * X['gdp_growth']
    return X[[
        'gdp_growth','gdp_total','gdp_pc','pop_density','urban_rate',
        'hsr_per_thousand','rail_per_thousand','hsr_share',
        'gdp_pc_urban_interaction','density_growth_interaction'
    ]]

fe = FunctionTransformer(make_features, validate=False)

# ─── 3. Define Pipelines ────────────────────────────────────────────────────────
pipe_lr = Pipeline([
    ('fe',     fe),
    ('impute', SimpleImputer(strategy='median')),
    ('scale',  StandardScaler()),
    ('clf',    LogisticRegression(
                   penalty='l2', solver='saga',
                   class_weight='balanced', max_iter=2000,
                   random_state=0
               ))
])

pipe_smote_lr = ImbPipeline([
    ('fe',     fe),
    ('impute', SimpleImputer(strategy='median')),
    ('scale',  StandardScaler()),
    ('smote',  SMOTE(k_neighbors=3, random_state=0)),
    ('clf',    LogisticRegression(
                   penalty='l2', solver='saga',
                   max_iter=2000, random_state=0
               ))
])

pipe_svm = Pipeline([
    ('fe',     fe),
    ('impute', SimpleImputer(strategy='median')),
    ('scale',  StandardScaler()),
    ('clf',    SVC(
                   C=0.1, gamma=0.01, kernel='rbf',
                   class_weight='balanced', probability=True,
                   random_state=0
               ))
])

pipe_rf = Pipeline([
    ('fe',     fe),
    ('impute', SimpleImputer(strategy='median')),
    ('clf',    RandomForestClassifier(
                   n_estimators=50, max_depth=2,
                   min_samples_leaf=4,
                   class_weight='balanced',
                   random_state=0
               ))
])

voting = VotingClassifier(
    estimators=[
        ('lr',    pipe_smote_lr),
        ('svm',   pipe_svm),
        ('rf',    pipe_rf)
    ],
    voting='soft'
)

pipelines = {
    'Logistic Regression':       pipe_lr,
    'SMOTE + Logistic Regression': pipe_smote_lr,
    'SVM (RBF)':                 pipe_svm,
    'Random Forest':             pipe_rf,
    'Voting Ensemble':           voting
}

# ─── 4. Fit, Save & Evaluate ────────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
loo = LeaveOneOut()

for name, pipe in pipelines.items():
    # 4A) Fit full-data
    pipe.fit(df, df['success'])

    # 4B) Save
    fname = f"model/{name.replace(' ', '_').lower()}.joblib"
    joblib.dump(pipe, fname)

    # 4C) LOOCV accuracy
    acc = cross_val_score(pipe, df, df['success'],
                          cv=loo, scoring='accuracy', n_jobs=-1).mean()
    print(f"{name:25s} → saved to {fname} · LOOCV accuracy: {acc:.2%}")
