python preprocess_dataset.py \
	--data_dir 'data/messidor2' \
	--raw_dir 'data/messidor2/raw/Messidor-2' \
	--proc_dir 'data/messidor2/processed' \
	--labels 'data/messidor2/MessidorLabelsBinary.csv'


python preprocess_dataset.py \
	--data_dir 'data/kaggle' \
	--raw_dir 'data/kaggle/raw/train' \
	--proc_dir 'data/kaggle/processed' \
	--labels 'data/kaggle/trainLabelsBinary.csv' \
	--img_type 'jpeg'