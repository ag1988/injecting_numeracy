# downloaded genbert_data_and_models.tar.gz
echo "downloading genbert_data_and_models.tar.gz (7.7G) ... this can take some time ..."
wget https://storage.googleapis.com/ai2i/genbert/genbert_data_and_models.tar.gz

# untar the downloaded data to get pre_training_data.tar.gz, gen_bert_models.tar.gz, squad_models.tar.gz
echo "inflating genbert_data_and_models.tar.gz ..."
tar -xzvf genbert_data_and_models.tar.gz

# move the files to respective sub-dirs
mv gen_bert_models.tar.gz  ./gen_bert
mv squad_models.tar.gz  ./gen_bert/squad_finetuning

# untar to get ./data dir inside pre_training dir
echo "inflating pre_training_data.tar.gz ..."
tar -xzvf pre_training_data.tar.gz

cd ./data
echo "downloading DROP ..."
wget https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip
unzip drop_dataset.zip
mv ./drop_dataset/*.json .
mv drop_dataset.zip drop_dataset
rm -rf drop_dataset
cd ..

cd ./gen_bert
echo "inflating gen_bert_models.tar.gz inside ./gen_bert ..."
tar -xzvf gen_bert_models.tar.gz

cd ./squad_finetuning
echo "inflating squad_models.tar.gz inside ./gen_bert/squad_finetuning ..."
tar -xzvf squad_models.tar.gz
