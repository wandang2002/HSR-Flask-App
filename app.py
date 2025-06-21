# app.py

import pandas as pd
import joblib
from flask import Flask, render_template, request
from sklearn.model_selection import LeaveOneOut, cross_val_score

app = Flask(__name__)

# ─── Load country metadata ──────────────────────────────────────────────────────
metrics_df = pd.read_csv('dataset/country.csv')

# ─── Load all trained pipelines ─────────────────────────────────────────────────
model_files = {
  'Logistic Regression':        'model/model/logistic_regression.joblib',
  'SMOTE + Logistic Regression':'model/model/smote_+_logistic_regression.joblib',
  'SVM (RBF)':                  'model/model/svm_(rbf).joblib',
  'Random Forest':              'model/model/random_forest.joblib',
  'Voting Ensemble':            'model/model/voting_ensemble.joblib'
}
def make_features(X):
    X = X.copy()
    X['hsr_per_thousand']           = X['hsr_km'] / X['pop_thousands']
    X['rail_per_thousand']          = X['rail_km'] / X['pop_thousands']
    X['hsr_share']                  = X['hsr_km'] / X['rail_km']
    X['gdp_pc_urban_interaction']   = X['gdp_pc'] * X['urban_rate']
    X['density_growth_interaction'] = X['pop_density'] * X['gdp_growth']
    return X[[
      'gdp_growth','gdp_total','gdp_pc',
      'pop_density','urban_rate',
      'hsr_per_thousand','rail_per_thousand','hsr_share',
      'gdp_pc_urban_interaction','density_growth_interaction'
    ]]
pipelines = {name: joblib.load(path) for name, path in model_files.items()}

# ─── Compute LOOCV accuracies once at startup ───────────────────────────────────
# ─── Load & preprocess the original INDICATORS for cross-val ─────────────────
df = pd.read_csv('model/INDICATORS.csv')

# 1) rename the success label
if 'label success' in df.columns:
    df.rename(columns={'label success':'success'}, inplace=True)

# 2) rename all raw feature columns to match your training script
df.rename(columns={
    'gdp_growth_rate (billions of us dollars)':    'gdp_growth',
    'gdp (percent change)':                        'gdp_total',
    'gdp_per_capita (us dollar per capita)':       'gdp_pc',
    'rail lines (total route-km)':                 'rail_km',
    'hsr_network_length':                          'hsr_km',
    'total_population (thousands)':                'pop_thousands',
    'population_density (people/square kilometer)': 'pop_density',
    'urbanization_rate':                           'urban_rate'
}, inplace=True)

# 3) coerce those columns to numeric (remove commas if any)
for col in ['rail_km','hsr_km','gdp_growth','gdp_total','gdp_pc',
            'pop_thousands','pop_density','urban_rate']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',',''),
                            errors='coerce')

X_all = df            # your pipelines start with make_features(impute,scale,…)
y_all = df['success']
loo = LeaveOneOut()
model_options = []
for name, pipe in pipelines.items():
    acc = cross_val_score(pipe, X_all, y_all,
                          cv=loo, scoring='accuracy', n_jobs=-1).mean()
    model_options.append({'name': name, 'accuracy': acc})

# ─── Flask routes ───────────────────────────────────────────────────────────────
@app.route('/', methods=['GET','POST'])
def index():
    probability = None
    selected_country = None
    selected_model   = None
    region = income_level = None

    if request.method == 'POST':
        # User inputs
        selected_country = request.form['country']
        selected_model   = request.form['model']
        hsr_length = float(request.form['hsr_length'])
        budget     = float(request.form['budget'])  # still unused by model

        # Lookup country stats
        row = metrics_df.loc[metrics_df['country'] == selected_country].iloc[0]
        region = row['region']
        income_level = row['income_level']

        # Build the one-row DataFrame
        data = {
          'gdp_growth':    row['gdp_growth'],
          'gdp_total':     row['gdp_total'],
          'gdp_pc':        row['gdp_pc'],
          'pop_thousands': row['pop_thousands'],
          'pop_density':   row['pop_density'],
          'urban_rate':    row['urban_rate'],
          'rail_km':       row['rail_km'],
          'hsr_km':        hsr_length
        }
        df_input = pd.DataFrame([data])

        # Predict with the chosen pipeline
        pipe = pipelines[selected_model]
        proba = pipe.predict_proba(df_input)[0,1]
        probability = f"{proba*100:.1f}%"

    return render_template(
        'index.html',
        countries=metrics_df['country'].tolist(),
        model_options=model_options,
        selected_country=selected_country,
        selected_model=selected_model,
        region=region,
        income_level=income_level,
        probability=probability
    )

if __name__ == '__main__':
    app.run(debug=True)
