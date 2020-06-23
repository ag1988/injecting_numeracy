### Finetuning on the SQuAD v1.1 dataset

First download SQuAD v1 using:
```
bash download.sh
```
Files will be placed inside `./data` (a dir inside the `squad_finetuning` dir).

#### Conventional finetuning for BERT.

Training:
```
CUDA_VISIBLE_DEVICES=0 python run_squad.py  --bert_model bert-base-uncased  --do_train  --do_predict --do_lower_case --train_file data/train-v1.1.json   --predict_file data/dev-v1.1.json   --train_batch_size 13   --predict_batch_size 256   --learning_rate 3e-5   --num_train_epochs 3.0   --max_seq_length 384   --doc_stride 128   --overwrite_output_dir   --output_dir out_squad_bert_orig   --n_best_size 10   --eval_period 800  --load_from_cache --use_segment_ids --num_train_samples -1
```
Inference:
```
CUDA_VISIBLE_DEVICES=0 python run_squad.py  --bert_model out_squad_bert_orig --do_evaluate --do_predict --do_lower_case --predict_file data/dev-v1.1.json  --predict_batch_size 256   --max_seq_length 384   --doc_stride 128  --use_segment_ids  --output_dir out_squad_bert_orig  --n_best_size 10
```
This will save `predictions.json` in `out_squad_bert_orig`.

#### Using digit tokenization and omitting segment ids for pre-trained models.

Training (initialize using a pre-trained model with out dir ../out_pre_training):
```
CUDA_VISIBLE_DEVICES=0 python run_squad.py  --bert_model ../out_pre_training  --do_train  --do_predict --do_lower_case --train_file data/train-v1.1.json   --predict_file data/dev-v1.1.json   --train_batch_size 13   --predict_batch_size 256   --learning_rate 3e-5   --num_train_epochs 3.0   --max_seq_length 384   --doc_stride 128   --overwrite_output_dir  --indiv_digits  --output_dir out_squad_pre_training  --n_best_size 10   --eval_period 800  --load_from_cache --num_train_samples -1
```
Inference:
```
CUDA_VISIBLE_DEVICES=0 python run_squad.py  --bert_model out_squad_pre_training --do_evaluate --do_predict --do_lower_case --predict_file data/dev-v1.1.json  --predict_batch_size 256   --max_seq_length 384   --doc_stride 128  --indiv_digits  --output_dir out_squad_pre_training   --n_best_size 10
```
This will save `predictions.json` in `out_squad_pre_training`.

#### Evaluation:
```
python evaluate-v1.1.py --dataset_file data/dev-v1.1.json --prediction_file out_squad_pre_training/predictions.json
```

##### Notes:  
1) To use a pre-trained model from a directory use --bert_model path_to_pre_train_dir .  
2) To use segment ids like in conventional BERT finetuning use --use_segment_ids .  
3) To use digit tokenization on top of word-piece tokenization use --indiv_digits .   

Contact: Ankit Gupta


