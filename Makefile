.PHONY: all data preprocess features train evaluate clean

all: data preprocess features train evaluate

data:
	python src/data/download_data.py --output_dir data/raw

preprocess:
	python src/data/preprocess.py \
	    --input_dir data/raw \
	    --output_dir data/processed

features:
	python src/features/build_features.py \
	    --input_dir data/processed \
	    --output_dir data/processed

train:
	python src/models/train_model.py \
	    --config config/config.yaml \
	    --data_dir data/processed \
	    --model_dir models

evaluate:
	python src/models/evaluate_model.py \
	    --config config/config.yaml \
	    --data_dir data/processed \
	    --model_dir models

clean:
	rm -rf data/processed models/*.pkl