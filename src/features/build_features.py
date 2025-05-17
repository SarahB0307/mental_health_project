import argparse
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def build_features(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(input_dir, 'processed_data.csv'))
    # Séparation X/y
    X = df.drop('depression_score', axis=1)
    y = df['depression_score']
    # Standardisation
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    # Sélection de variables par importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(20).index.tolist()
    # Nouveau DataFrame
    df_feat = pd.concat([X_scaled[top_features], y.reset_index(drop=True)], axis=1)
    out_path = os.path.join(output_dir, 'features.csv')
    df_feat.to_csv(out_path, index=False)
    print(f"Features sauvegardées dans {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    build_features(args.input_dir, args.output_dir)