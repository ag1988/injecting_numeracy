# Pre-training/finetuning models on DROP-like datasets

You can skip the steps accordingly. E.g. if you're not training on synthetic data, you dont need to create features for it. Similarly, you can ignore step 2) if you do not wish to re-train the trained models that we provide.

1) Create features from drop-like data (Digit Tokenizaiton will be used by default):
```
# DROP
$ python create_examples_n_features.py --split train --drop_json ../data/drop_dataset_train.json --output_dir data/examples_n_features --max_seq_length 512
$ python create_examples_n_features.py --split eval --drop_json ../data/drop_dataset_dev.json --output_dir data/examples_n_features --max_seq_length 512
```
The second command will save 2 files eval_examples.pkl and eval_features.pkl in the output_dir and similarly the first command will use prefix train. Similarly, 
```
# numeric data
$ python create_examples_n_features.py --split train --drop_json ../data/synthetic_numeric_train_drop_format.json --output_dir data/examples_n_features_numeric --max_seq_length 50 --max_decoding_steps 11 --max_n_samples -1
$ python create_examples_n_features.py --split eval --drop_json ../data/synthetic_numeric_dev_drop_format.json --output_dir data/examples_n_features_numeric --max_seq_length 50 --max_decoding_steps 11

# texual synthetic data
$ python create_examples_n_features.py --split train --drop_json ../data/synthetic_textual_mixed_min3_max6_up0.7_train_drop_format.json --output_dir data/examples_n_features_syntext --max_seq_length 160 --max_n_samples -1
$ python create_examples_n_features.py --split eval --drop_json ../data/synthetic_textual_mixed_min3_max6_up0.7_dev_drop_format.json --output_dir data/examples_n_features_syntext --max_seq_length 160
```
As the training sets are large, some of these can consume a lot of RAM.


2) Training GenBERT using the above generated features (you can reduce the number of gpu's as long as you're not getting OOM errors):

**Pretraining:**  

*GenBERT + ND*
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_on_drop.py  --do_train   --do_eval  --mlm_dir ../data/MLM_train/ --examples_n_features_dir ./data/examples_n_features_numeric/ --train_batch_size 800 --mlm_batch_size 48 --mlm_scale 0.5 --eval_batch_size 1200 --learning_rate 6e-5  --max_seq_length 50 --num_train_epochs 60.0 --warmup_proportion 0.1 --output_dir out_numeric_finetune_bert --random_shift --num_train_samples -1
```

*GenBERT + TD*
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_on_drop.py   --do_train   --do_eval  --mlm_dir ../data/MLM_train/ --examples_n_features_dir ./data/examples_n_features_syntext/ --train_batch_size 240 --mlm_batch_size 48 --mlm_scale 0.5 --eval_batch_size 1000 --learning_rate 1e-5  --max_seq_length 160 --num_train_epochs 5.0 --warmup_proportion 0.1 --output_dir out_syntext_finetune_bert --random_shift --num_train_samples -1
```

*GenBERT + ND + TD*
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_textual_with_numeric.py  --do_train   --do_eval --mlm_dir ../data/MLM_train/ --examples_n_features_dir_syntext ./data/examples_n_features_syntext/ --examples_n_features_dir_numeric ./data/examples_n_features_numeric/ --train_batch_size_syntext 240 --train_batch_size_numeric 624 --mlm_batch_size 48 --mlm_scale 0.5 --eval_batch_size 1000 --learning_rate 1e-5 --num_train_epochs 5.0 --warmup_proportion 0.1 --output_dir out_syntext_and_numeric_finetune_numeric --init_weights_dir out_numeric_finetune_bert --random_shift --num_train_samples -1
```
---

**Finetuning:** 

*GenBERT + DROP*
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune_on_drop.py   --do_train   --do_eval  --examples_n_features_dir ./data/examples_n_features/ --train_batch_size 16 --mlm_batch_size -1 --eval_batch_size 360 --learning_rate 3e-5  --max_seq_length 512 --num_train_epochs 30.0 --warmup_proportion 0.1 --output_dir out_drop_finetune_bert --num_train_samples -1
```

*GenBERT + ND + DROP*
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune_on_drop.py   --do_train   --do_eval  --examples_n_features_dir ./data/examples_n_features/ --train_batch_size 16 --mlm_batch_size -1 --eval_batch_size 360 --learning_rate 3e-5  --max_seq_length 512 --num_train_epochs 30.0 --warmup_proportion 0.1 --init_weights_dir out_numeric_finetune_bert --output_dir out_drop_finetune_numeric --num_train_samples -1
```

*GenBERT + TD + DROP*
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune_on_drop.py   --do_train   --do_eval  --examples_n_features_dir ./data/examples_n_features/ --train_batch_size 14 --mlm_batch_size -1 --eval_batch_size 400 --learning_rate 3e-5  --max_seq_length 512 --num_train_epochs 30.0 --warmup_proportion 0.1 --init_weights_dir out_syntext_finetune_bert --output_dir out_drop_finetune_syntext --num_train_samples -1
```

*GenBERT + ND + TD + DROP*
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune_on_drop.py   --do_train   --do_eval  --examples_n_features_dir ./data/examples_n_features/ --train_batch_size 14 --mlm_batch_size -1 --eval_batch_size 400 --learning_rate 3e-5  --max_seq_length 512 --num_train_epochs 30.0 --warmup_proportion 0.1 --init_weights_dir out_syntext_and_numeric_finetune_numeric --output_dir out_drop_finetune_syntext_and_numeric --num_train_samples -1
```
For each training, its output dir also contains informative files such as `training_args.bin` (can loaded via torch.load()), Tensorboard logs inside `log`, etc.


We provide our trained models with the following output dirs.

|                       | BERT                   | +ND                       | +TD                       | +ND+TD                                   | +ND-LM                           | +ND-LM-RS                                 | +ND-LM-DT                           |
|----|----|----|----|----|----|----|----|
| pre-trained           | -                      | out_numeric_finetune_bert | out_syntext_finetune_bert | out_syntext_and_numeric_finetune_numeric | out_no_mlm_numeric_finetune_bert | out_no_mlm_no_shift_numeric_finetune_bert | out_wp_no_mlm_numeric_finetune_bert |
| finetuned on DROP     | out_drop_finetune_bert | out_drop_finetune_numeric | out_drop_finetune_syntext | out_drop_finetune_syntext_and_numeric    | out_drop_finetune_no_mlm_numeric | out_drop_finetune_no_mlm_no_shift_numeric | -                                   |
| finetuned on SQuAD v1 | out_squad_bert_orig    | out_squad_numeric         | out_squad_syntext         | out_squad_syntext_and_numeric            | out_squad_no_mlm_numeric         | -                                         | -                                   |

3) Inference.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune_on_drop.py --do_eval --do_inference --examples_n_features_dir ./data/examples_n_features/ --eval_batch_size 800 --init_weights_dir out_drop_finetuned_model  --output_dir preds
```
This will save `predictions.jsonl` in `./preds` . Here `out_drop_finetuned_model` is one of the models finetuned on DROP as above.
Note that we use the eval examples and features in the `examples_n_features_dir` and so if the answers are not known (for test set), please create features using some defaut answer annotation like {'number': '0',...} and then do inference.


4) Evaluation (using DROP eval script): 
```
python drop_eval.py --gold_path ../data/drop_dataset_dev.json --prediction_path ./preds/predictions.jsonl --answer_key prediction
```
In case the prediction file supplied in --prediction_path is a .jsonl (list of dicts), to identify which key in these dicts corresponds to the prediction string, one can supply the key name using --answer_key. Otherwise, one can simply supply a .json containing query_id's as keys and prediction strings as values.

5) **SQuAD finetuning** : You can finetune the pre-trained models on SQuAD v1 by following the README inside `squad_finetuning` dir.

Notes:
1) To load a pre-trained model use --init_weights_dir out_your_previous_outdir .  
2) To exclude MLM task from training use --mlm_batch_size -1 .  
3) While pre-training on short inputs its good to use --random_shift.  
4) max_seq_length param is not necessary for finetune_on_drop.py as this parameter is stored inside the generated feaures. But its good to specify it (the same one used to create the features) for transparency.  
5) Tensorboard log dir for a training is inside its output dir.  


Contact: Ankit Gupta
