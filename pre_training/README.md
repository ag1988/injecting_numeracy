# Pre-training BERT on datasets in DROP format

**Download the data + trained models** : To download our data & models and then pre-process it, run `bash download.sh`. This will also place the downloaded data inside appropriate sub-dirs.  

You're not required to follow the instructions exactly - feel free to skip the steps accordingly. E.g. if you're not interested in pre-training and only want to evaluate our finetuned models on DROP you can simply skip the steps involving the synthetic and MLM data.  

**Masked LM Task** : Wiki pages inside `./data/MLM_paras.jsonl` can be tokenized and masked via `gen_train_data_MLM.py` (to create `MLM_paras.jsonl`, after [downloading & preparing](https://hotpotqa.github.io/wiki-readme.html) Wikipedia, we concatenated the paras after placing a string `■ .` to indicate end of para. We only kept the pages corresponding to the titles in `./data/wiki_titles_used.jsonl`). You can create masked MLM instances via:
```
python gen_train_data_MLM.py --train_corpus ./data/MLM_paras.jsonl --bert_model bert-base-uncased --output_dir ./data/MLM_train/ --do_lower_case --max_predictions_per_seq 65 --digitize
```
The instances will be saved in `./data/MLM_train/` dir. 

---

**Numeric Data (ND)** :  You can generate numeric data by following the README inside `numeric_data_generation` dir. The numeric data will be stored as `synthetic_numeric.jsonl` in `./data` dir. For finetuning, this should then be converted to DROP format as follows:
```
python convert_synthetic_numeric_to_drop.py --data_jsonl ./data/synthetic_numeric.jsonl
```
This will output `synthetic_numeric_train_drop_format.json`, `synthetic_numeric_dev_drop_format.json` in `./data` dir.

---

**Textual Data (ND)** :  Although we provide the textual data files `synthetic_textual_mixed_min3_max6_up0.7_train.json` and `synthetic_textual_mixed_min3_max6_up0.7_dev.json` that we used for our training, you can generate textual data again by following the README inside `../textual_data_generation` dir. The resulting `.json`s must be moved to `./data`. For finetuning, these should then be converted to DROP format as follows:
```
python convert_synthetic_texual_to_drop.py --data_json ./data/synthetic_textual_mixed_min3_max6_up0.7_train.json
python convert_synthetic_texual_to_drop.py --data_json ./data/synthetic_textual_mixed_min3_max6_up0.7_dev.json
```
This will output `synthetic_textual_mixed_min3_max6_up0.7_train_drop_format.json`, `synthetic_textual_mixed_min3_max6_up0.7_dev_drop_format.json` in `./data` dir.

---

**This concludes the data generation part - you can now pre-train/finetune by following the README inside `./gen_bert` dir.**  

Code was tested on Python 3.7.6 with `requirements.txt` containing the list of libraries.   

Contact: Ankit Gupta