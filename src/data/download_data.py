import argparse
import shutil
import os

def main(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    src = 'C:\\Users\\beldj\\Downloads\\mental_health_dataset.csv'
    dst = os.path.join(output_dir, 'mental_health.csv')
    shutil.copyfile(src, dst)
    print(f"Données copiées dans {dst}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    main(args.output_dir)