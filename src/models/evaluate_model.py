import argparse
import os
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def evaluate(config_path, data_dir, model_dir):
    # Chargement modèle + test
    path = os.path.join(model_dir, 'rf_model.pkl')
    data = joblib.load(path)
    model = data['model']
    X_test = data['X_test']
    y_test = data['y_test']
    # Prédictions
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    # Rapport
    report = f"RMSE: {rmse:.3f}\nR2: {r2:.3f}\n"
    report_path = os.path.join(model_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Rapport d'évaluation sauvegardé dans {report_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_dir', required=True)
    args = parser.parse_args()
    evaluate(args.config, args.data_dir, args.model_dir)