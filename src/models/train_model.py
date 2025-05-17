import argparse
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib


def train(config_path, data_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    # Lecture config
    config = yaml.safe_load(open(config_path))
    # Chargement données
    df = pd.read_csv(os.path.join(data_dir, 'features.csv'))
    X = df.drop('depression_score', axis=1)
    y = df['depression_score']
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config['seed']
    )
    # Modèle
    params = config['model']['random_forest']
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    # Sauvegarde
    model_path = os.path.join(model_dir, 'rf_model.pkl')
    joblib.dump({'model': rf, 'X_test': X_test, 'y_test': y_test}, model_path)
    print(f"Modèle et données de test sauvegardés dans {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_dir', required=True)
    args = parser.parse_args()
    train(args.config, args.data_dir, args.model_dir)