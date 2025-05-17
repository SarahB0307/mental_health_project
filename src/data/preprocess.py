import argparse
import os
import pandas as pd
from sklearn.impute import SimpleImputer


def preprocess(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Lecture
    df = pd.read_csv(os.path.join(input_dir, 'mental_health.csv'))
    # Imputation numérique
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    imp_num = SimpleImputer(strategy='median')
    df[num_cols] = imp_num.fit_transform(df[num_cols])
    # Imputation catégorielle + encodage
    cat_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    # Sauvegarde
    out_path = os.path.join(output_dir, 'processed_data.csv')
    df.to_csv(out_path, index=False)
    print(f"Données prétraitées sauvegardées dans {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    preprocess(args.input_dir, args.output_dir)