"""A Transformer with a BERT encoder and BERT decoder with extensive weight tying."""
# In each decoder layer, the self attention params are also used for source attention, 
# thereby allowing us to use BERT as a decoder as well.
# Most of the code is taken from HuggingFace's repo.

from __future__ import absolute_import

import argparse
import logging
import os, sys, random, jsonlines, shutil, time
import ujson as json
from scipy.special import softmax
from io import open
from collections import namedtuple
from pathlib import Path
from tqdm import tqdm, trange

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling import BertTransformer, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.tokenization import BertTokenizer

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

from create_examples_n_features import DropExample, DropFeatures, read_file, write_file, split_digits
from squad_utils import exact_match_score, get_final_text
from allennlp.training.metrics.drop_em_and_f1 import DropEmAndF1

START_TOK, END_TOK, SPAN_SEP, IGNORE_IDX, MAX_SPANS = '@', '\\', ';', 0, 6


LMInputFeatures = namedtuple("LMInputFeatures", "input_ids input_mask lm_label_ids")

class MLMDataset(TensorDataset):
    def __init__(self, training_path, epoch=1, tokenizer=None, num_data_epochs=1):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        data_file = training_path / f"epoch_{self.data_epoch}.jsonl"
        metrics_file = training_path / f"epoch_{self.data_epoch}_metrics.jsonl"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        self.seq_len = seq_len = metrics['max_seq_len']
        
        input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
        input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
        lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=IGNORE_IDX) 
        # ignore index == 0
        logging.info(f"Loading MLM examples for epoch {epoch}")
        with jsonlines.open(data_file, 'r') as reader:
            for i, example in enumerate(tqdm(reader.iter(), total=num_samples, desc="MLM examples")):
                features = self.convert_example_to_features(example)
                input_ids[i] = features.input_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
#                 if i == 1000:
#                     break
        print()
        assert i == num_samples - 1  # Assert that the sample count metric was true
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.lm_label_ids = lm_label_ids

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item]).long(),
                torch.tensor(self.input_masks[item]).long(),
                torch.tensor(self.lm_label_ids[item]).long())
    
    def convert_example_to_features(self, example):
        tokens = example["tokens"]
        masked_lm_positions = example["masked_lm_positions"]
        masked_lm_labels = example["masked_lm_labels"]
        max_seq_length = self.seq_len
        
        assert len(tokens) <= max_seq_length
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        masked_label_ids = self.tokenizer.convert_tokens_to_ids(masked_lm_labels)

        input_array = np.zeros(max_seq_length, dtype=np.int)
        input_array[:len(input_ids)] = input_ids

        mask_array = np.zeros(max_seq_length, dtype=np.bool)
        mask_array[:len(input_ids)] = 1

        lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=IGNORE_IDX)
        lm_label_array[masked_lm_positions] = masked_label_ids

        features = LMInputFeatures(input_ids=input_array, input_mask=mask_array,
                                   lm_label_ids=lm_label_array)
        return features


ModelFeatures = namedtuple("ModelFeatures", "example_id input_ids input_mask segment_ids label_ids head_type q_spans p_spans")

class DropDataset(TensorDataset):
    def __init__(self, args, split='train'):
        logging.info(f"Loading {split} examples and features.")
        direc = args.examples_n_features_dir
        if split == 'train':
            examples = read_file(direc + '/train_examples.pkl')
            drop_features = read_file(direc + '/train_features.pkl')
        else:
            examples = read_file(direc + '/eval_examples.pkl')
            drop_features = read_file(direc + '/eval_features.pkl')
        
        num_samples = len(examples)
        self.max_dec_steps = len(drop_features[0].decoder_label_ids)
        
        features = []
        for i, (example, drop_feature) in enumerate(zip(examples, drop_features)):
            features.append(self.convert_to_input_features(example, drop_feature))
            if split == 'train' and args.num_train_samples >= 0 and len(features) >= args.num_train_samples:
                break
        print()
#         assert i == num_samples - 1
        self.num_samples = len(features)
        self.seq_len = drop_features[0].max_seq_length
        self.examples = examples
        self.drop_features = drop_features
        self.features = features
        self.example_ids = [f.example_id for f in features]
        self.input_ids = torch.tensor([f.input_ids for f in features]).long()
        self.input_mask = torch.tensor([f.input_mask for f in features]).long()
        self.segment_ids = torch.tensor([f.segment_ids for f in features]).long()
        self.label_ids = torch.tensor([f.label_ids for f in features]).long()
        self.head_type = torch.tensor([f.head_type for f in features]).long()
        self.q_spans = torch.tensor([f.q_spans for f in features]).long()
        self.p_spans = torch.tensor([f.p_spans for f in features]).long()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (self.input_ids[item], self.input_mask[item], self.segment_ids[item], self.label_ids[item], 
                self.head_type[item], self.q_spans[item], self.p_spans[item])
    
    def convert_to_input_features(self, drop_example, drop_feature):
        max_seq_len = drop_feature.max_seq_length
        
        # input ids are padded by 0
        input_ids = drop_feature.input_ids
        input_ids += [IGNORE_IDX] * (max_seq_len - len(input_ids))
        
        # input mask is padded by 0
        input_mask = drop_feature.input_mask
        input_mask += [0] * (max_seq_len - len(input_mask))
        
        # segment ids are padded by 0
        segment_ids = drop_feature.segment_ids
        segment_ids += [0] * (max_seq_len - len(segment_ids))
        
        # we assume dec label ids are already padded by 0s
        decoder_label_ids = drop_feature.decoder_label_ids
        assert len(decoder_label_ids) == self.max_dec_steps 
        #decoder_label_ids += [0] * (MAX_DECODING_STEPS - len(decoder_label_ids))
        
        # for span extraction head, ignore idx == -1
        question_len = segment_ids.index(1) if 1 in segment_ids else len(segment_ids)
        starts, ends = drop_feature.start_indices, drop_feature.end_indices
        q_spans, p_spans = [], []
        for st, en in zip(starts, ends):
            if any([x < 0 or x >= max_seq_len for x in [st, en]]):
                continue
            elif all([x < question_len for x in [st, en]]):
                q_spans.append([st, en])
            elif all([question_len <= x for x in [st, en]]):
                p_spans.append([st, en])
        q_spans, p_spans = q_spans[:MAX_SPANS], p_spans[:MAX_SPANS]
        head_type = 1 if q_spans or p_spans else -1
        q_spans += [[-1,-1]]*(MAX_SPANS - len(q_spans))
        p_spans += [[-1,-1]]*(MAX_SPANS - len(p_spans))
                
        return ModelFeatures(drop_feature.example_index, input_ids, input_mask, segment_ids,
                             decoder_label_ids, head_type, q_spans, p_spans)

    
def make_output_dir(args, scripts_to_save=[sys.argv[0]]):
    # scripts_to_save are relative paths of files to save
    os.makedirs(args.output_dir, exist_ok=True)
    tb_dir = os.path.join(args.output_dir, 'log')
    # remove prev tensorboard logs
    shutil.rmtree(tb_dir, ignore_errors=True)
    code_dir = os.path.join(args.output_dir, 'scripts')
    os.makedirs(code_dir, exist_ok=True)
    for script in scripts_to_save:
        dst_file = os.path.join(code_dir, os.path.basename(script))
        shutil.copyfile(script, dst_file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--examples_n_features_dir",
                        default='data/examples_n_features/',
                        type=str,
                        help="Dir containing drop examples and features.")
    parser.add_argument("--mlm_dir",
                        default='../data/MLM_train/',
                        type=Path,
                        help="The data dir with MLM taks. Should contain the .jsonl files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default='./out_drop_finetune',
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--model", default="bert-base-uncased", type=str)
    parser.add_argument("--init_weights_dir",
                        default='',
                        type=str,
                        help="The directory where init model wegihts an config are stored.")
    parser.add_argument("--max_seq_length",
                        default=-1,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_inference",
                        action='store_true',
                        help="Whether to run inference on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--mlm_batch_size",
                        default=-1,
                        type=int,
                        help="Total batch size for mlm train data.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_samples",
                        default=-1,
                        type=int,
                        help="Total number of training samples used.")
    parser.add_argument("--num_train_epochs",
                        default=6.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--freeze_encoder',
                        action='store_true',
                        help="Whether to freeze the bert encoder, embeddings.")
#     parser.add_argument('--indiv_digits',
#                         action='store_true',
#                         help="Whether to tokenize numbers as digits.")
    parser.add_argument('--rand_init',
                        action='store_true',
                        help="Whether to use random init instead of BERT.")
    parser.add_argument('--random_shift',
                        action='store_true',
                        help="Whether to randomly shift position ids of encoder input.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--mlm_scale',
                        type=float, default=1.0,
                        help="mlm loss scaling factor.")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.do_train:
        make_output_dir(args, scripts_to_save=[sys.argv[0], 'modeling.py', 'create_examples_n_features.py'])

    #tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    if args.init_weights_dir:
        model = BertTransformer.from_pretrained(args.init_weights_dir)
    else:
        # prepare model
        model = BertTransformer.from_pretrained(args.model,
            cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank)))
    
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    eval_data = DropDataset(args, 'eval')
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_data))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    if args.do_eval and args.do_inference:
        inference(args, model, eval_dataloader, device, tokenizer)
        exit()
        
    if args.do_train:
        # Prepare data loader
        train_data = DropDataset(args, 'train')
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        
        # Prepare optimizer
        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=args.learning_rate,
                     warmup=args.warmup_proportion,
                     t_total=num_train_optimization_steps)
        
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        
        '''
        ------------------------------------------------------------------------------
        TODO: check training resume, fp16, use --random_shift for short inputs
        ------------------------------------------------------------------------------
        '''
        # using fp16
        fp16 = False
#         try:
#             from apex import amp
#             model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
#             fp16 = True
#         except ImportError:
#             logger.info("Not using 16-bit training due to apex import error.")
        
#         if n_gpu > 1:
#             model = torch.nn.DataParallel(model)
        
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'log'))  # tensorboard
        
        # masked LM data
        do_mlm_task = False 
        if args.mlm_batch_size > 0:
            mlm_dataset = MLMDataset(training_path=args.mlm_dir, tokenizer=tokenizer)
            mlm_dataloader = DataLoader(mlm_dataset, sampler=RandomSampler(mlm_dataset), batch_size=args.mlm_batch_size)
            mlm_iter = iter(mlm_dataloader)
            do_mlm_task = True
            
        model.train()
        (global_step, all_losses, all_errors, all_dec_losses, all_dec_errors, eval_errors,
         best, best_mlm, t_prev, do_eval) = 0, [], [], [], [], [], 1000, 1000, time.time(), False
        mlm_losses, mlm_errors, all_span_losses, all_span_errors = [], [], [], []
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                # grads wrt to train data
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, head_type, q_spans, p_spans = batch
                
                losses = model(input_ids, segment_ids, input_mask, random_shift=args.random_shift, target_ids=label_ids,
                               target_mask=None, answer_as_question_spans=q_spans, answer_as_passage_spans=p_spans,
                               head_type=head_type)
                loss, errs, dec_loss, dec_errors, span_loss, span_errors, type_loss, type_errors, type_preds = losses
                
                # aggregate on multi-gpu
                take_mean = lambda x: x.mean() if x is not None and sum(x.size()) > 1 else x
                take_sum = lambda x: x.sum() if x is not None and sum(x.size()) > 1 else x
                [loss, dec_loss, span_loss, type_loss] = list(map(take_mean, [loss, dec_loss, span_loss, type_loss]))
                [errs, dec_errors, span_errors, type_errors] = list(map(take_sum, [errs, dec_errors, span_errors, type_errors]))
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                all_losses.append(loss.item()); all_dec_losses.append(dec_loss.item()); 
                all_errors.append(errs.item() / input_ids.size(0))
                all_dec_errors.append(dec_errors.item() / input_ids.size(0))
                all_span_losses.append(span_loss.item()); all_span_errors.append(span_errors.item() / input_ids.size(0))
         
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                if do_mlm_task:
                    # grads wrt to mlm data
                    while True:
                        try:
                            batch = next(mlm_iter)  # sample next mlm batch
                            break
                        except StopIteration:       # end of epoch: reset and shuffle
                            mlm_iter = iter(mlm_dataloader)
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, label_ids = batch
                    loss, errs = model(input_ids, None, input_mask, target_ids=label_ids, 
                                       ignore_idx=IGNORE_IDX, task='mlm')
                    loss, err_sum = take_mean(loss), take_sum(errs)      # for multi-gpu
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    loss = args.mlm_scale * loss
                    mlm_losses.append(loss.item()); mlm_errors.append(err_sum.item() / input_ids.size(0))

                    if fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                else:
                    mlm_losses.append(-1); mlm_errors.append(-1)
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()          # update step
                    optimizer.zero_grad()
                    global_step += 1
                    
                train_result = {'trn_loss': all_losses[-1], 'trn_dec_loss': all_dec_losses[-1], 
                                'trn_err': all_errors[-1], 'trn_dec_err': all_dec_errors[-1], 
                                'lr': optimizer.get_lr()[0], 'trn_span_loss': all_span_losses[-1], 
                                'trn_span_err': all_span_errors[-1], 'epoch': epoch}
                mlm_result = {'trn_mlm_loss': mlm_losses[-1], 'trn_mlm_err': mlm_errors[-1]}
                tb_writer.add_scalars('train', train_result, len(all_losses))
                tb_writer.add_scalars('mlm', mlm_result, len(all_losses)) if do_mlm_task else None
    
                if time.time() - t_prev > 60*60: # evaluate every hr
                    do_eval = True
                    t_prev = time.time()
            
                if do_eval:
                    do_eval = False
                    eval_result = evaluate(args, model, eval_dataloader, device, len(train_data))
                    eval_err = eval_result['eval_err']
                    if eval_err < best or (eval_err < best + 0.005 and np.mean(mlm_errors[-1000:]) < best_mlm):
                        # if eval err is in range of best, look at MLM err
                        train_state = {'global_step': global_step, 'optimizer_state_dict': optimizer.state_dict()}
                        train_state.update(train_result)
                        save(args, model, tokenizer, train_state)
                        best_mlm = min(best_mlm, np.mean(mlm_errors[-1000:]))
                    best = min(best, eval_err)
                    eval_errors.append((len(all_losses), eval_err))
                    model.train()
            
                    tb_writer.add_scalars('eval', eval_result, len(all_losses))
#                     for name, param in model.named_parameters():
#                         tb_writer.add_histogram(name, param.clone().cpu().data.numpy(), len(all_losses))
            # end of epoch
            do_eval = True

        # training complete
        tb_writer.export_scalars_to_json(os.path.join(args.output_dir, 'training_scalars.json'))
        tb_writer.close()
        
        
def save(args, model, tokenizer, train_state_dict):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    output_args_file = os.path.join(args.output_dir, 'training_args.bin')
    train_state_file = os.path.join(args.output_dir, 'training_state.bin')
    
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)
    torch.save(args, output_args_file)
#     torch.save(train_state_dict, train_state_file)


def evaluate(args, model, eval_dataloader, device, n_train):
    model.eval()
    eval_examples = eval_dataloader.dataset.examples
    predictions, eval_loss, eval_accuracy, eval_err_sum = {}, 0, 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    sample_accs = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, head_type, q_spans, p_spans = batch
        
        with torch.no_grad():
            losses = model(input_ids, segment_ids, input_mask, random_shift=args.random_shift, target_ids=label_ids,
                           target_mask=None, answer_as_question_spans=q_spans, answer_as_passage_spans=p_spans,
                           head_type=head_type)
            loss, errs, dec_loss, dec_errors, span_loss, span_errors, type_loss, type_errors, type_preds = losses 
            tmp_eval_loss = loss

        for i, sample_acc in enumerate((errs == 0).cpu().tolist()):
            sample_accs.append({'qid': eval_examples[i+nb_eval_examples].qas_id, 'em': sample_acc})
        
        tmp_eval_accuracy = (errs == 0).sum().item()

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        eval_err_sum += dec_errors.sum().item()

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    eval_err_sum /= nb_eval_examples

    result = {'eval_loss': eval_loss,
              'eval_acc': eval_accuracy,
              'eval_dec_err_sum': eval_err_sum,
              'eval_err': 1-eval_accuracy}

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "a+") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\t" % (key, str(result[key])))
        writer.write("n_train = %d\t" % n_train)
        writer.write("\n")

    write_file(sample_accs, os.path.join(args.output_dir, "ems.jsonl"))

    return result


def inference(args, model, eval_dataloader, device, tokenizer):
    model.eval()
    eval_examples = eval_dataloader.dataset.examples
    eval_drop_features = eval_dataloader.dataset.drop_features
    [start_tok_id, end_tok_id] = tokenizer.convert_tokens_to_ids([START_TOK, END_TOK]) # [1030, 1032]
    all_dec_ids, all_label_ids, all_type_preds, all_start_preds, all_end_preds, all_type_logits = [], [], [], [], [], []
    all_input_ids = []
    nb_eval_examples, eval_accuracy, eval_err_sum = 0, 0, 0
    
    for batch in tqdm(eval_dataloader, desc="Inference"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, head_type, q_spans, p_spans = batch
        
        with torch.no_grad():
            out = model(input_ids, segment_ids, task='inference')
            # here segment_ids are only used to get the best span prediction
            dec_preds, type_preds, start_preds, end_preds, type_logits = tuple(t.cpu() for t in out)
            # dec_preds: [bsz, max_deocoding_steps], has start_tok
#           # type_preds, start_preds, end_preds, type_logits : [bsz], [bsz], [bsz], [bsz, 2]
        assert dec_preds.size() == label_ids.size()
        assert dec_preds.dim() == 2
        
#         bch_errs = ((bch_preds != target_ids).float() * (target_ids != IGNORE_IDX).float()).sum(dim=-1)
#         bch_eval_accuracy = (bch_errs == 0).sum().item()
#         eval_accuracy += bch_eval_accuracy
#         eval_err_sum += bch_errs.sum().item() # all errors
        nb_eval_examples += input_ids.size(0)
        all_dec_ids.append(dec_preds); all_label_ids.append(label_ids); all_type_preds.append(type_preds)
        all_start_preds.append(start_preds); all_end_preds.append(end_preds); all_type_logits.append(type_logits)
        all_input_ids.append(input_ids)
        #break
#     eval_accuracy /= nb_eval_examples
#     eval_err_sum /= nb_eval_examples
#     result = {'eval_accuracy': eval_accuracy,
#               'eval_err_sum': eval_err_sum}

#     logger.info("***** Eval results *****")
#     for key in sorted(result.keys()):
#         logger.info("  %s = %s", key, str(result[key]))
    
    tup = all_dec_ids, all_label_ids, all_type_preds, all_start_preds, all_end_preds, all_type_logits, all_input_ids
    all_dec_ids, all_label_ids, all_type_preds, all_start_preds, all_end_preds, all_type_logits, all_input_ids = \
                                                                tuple(torch.cat(t, dim=0).tolist() for t in tup)
    def trim(ids):
        # remove start tok
        ids = ids[1:] if ids[0] == start_tok_id else ids
        # only keep predictions until the first pad/end token
        _ids = []
        for id in ids:
            if id in [IGNORE_IDX, end_tok_id]:
                break
            else:
                _ids.append(id)
        return _ids
    def process(text):
        processed = '.'.join([x.strip() for x in text.split('.')]) # remove space around decimal
        try:
            float(processed)  #'.' is a decimal only if final str is a number
        except ValueError:
            processed = text
        return '-'.join([x.strip() for x in processed.split('-')]) # remove space around "-"
        
    predictions, ems, drop_ems = [], [], []
    for i in range(len(all_dec_ids)):
        example = eval_examples[i]
        drop_feature = eval_drop_features[i]
        answer_text = (SPAN_SEP+' ').join(example.answer_texts).strip().lower()
        processed_answer_text = process(answer_text)
        # generator prediction
        dec_ids = trim(all_dec_ids[i])
        dec_toks = tokenizer.convert_ids_to_tokens(dec_ids)
        dec_text = detokenize(dec_toks)
        dec_processed = process(dec_text)
        # span prediction
        start_pred, end_pred, input_ids = all_start_preds[i], all_end_preds[i], all_input_ids[i]
        [start_pred, end_pred] = sorted([start_pred, end_pred])
        span_ids = [x for x in input_ids[start_pred:end_pred+1] if x != 0]
        span_toks = tokenizer.convert_ids_to_tokens(span_ids)
        span_text = detokenize(span_toks)
        span_processed = process(span_text)
        
        span_pred, used_orig = wrapped_get_final_text(example, drop_feature, start_pred, end_pred)
        if not used_orig:
            span_pred = process(span_pred)
    
        prediction = span_pred if all_type_preds[i] else dec_processed
        head_pred = 'span_extraction' if all_type_preds[i] else 'generator'
        
        # compute drop em and f1
        drp = DropEmAndF1()
        drp(prediction, example.answer_annotations)
        drop_em, drop_f1 = drp.get_metric()
        em = exact_match_score(prediction, processed_answer_text)
        
        predictions.append({'query_id': example.qas_id, 'passage_id':example.passage_id, 
                            'processed_dec_out': dec_processed, 'prediction': prediction, 
                            'ans_used': processed_answer_text, 'type_logits': all_type_logits[i],
                            'head_pred': head_pred, 'processed_span_out': span_processed,
                            'dec_out': dec_text, 'span_out': span_text, 'span_pred': span_pred,
                            'drop_em': drop_em, 'drop_f1': drop_f1, 'em': em})
        ems.append(em); drop_ems.append(drop_em)
        if i < 20:
            print(prediction, processed_answer_text, end=' || ')
    logger.info(f'EM: {np.mean(ems)}, Drop EM: {np.mean(drop_ems)}')
    logger.info('saving predictions.jsonl in ' + args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    write_file(predictions, os.path.join(args.output_dir, "predictions.jsonl"))

    
def detokenize(tok_tokens):
    tok_text = " ".join(tok_tokens)

    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    return tok_text


def wrapped_get_final_text(example, feature, start_index, end_index, do_lower_case=True, verbose_logging=False, logger=None):
    tok_tokens = feature.tokens[start_index:(end_index + 1)]
    tok_tokens = [tok for tok in tok_tokens if tok not in ['[PAD]', '[CLS]', '[SEP]']]
    tok_text = detokenize(tok_tokens)
    
    if start_index in feature.doc_token_to_orig_map and end_index in feature.doc_token_to_orig_map:
        orig_doc_start = feature.doc_token_to_orig_map[start_index]
        orig_doc_end = feature.doc_token_to_orig_map[end_index]
        orig_tokens = example.orig_passage_tokens[orig_doc_start:(orig_doc_end + 1)]
    elif start_index in feature.que_token_to_orig_map and end_index in feature.que_token_to_orig_map:
        orig_que_start = feature.que_token_to_orig_map[start_index]
        orig_que_end = feature.que_token_to_orig_map[end_index]
        orig_tokens = example.orig_question_tokens[orig_que_start:(orig_que_end + 1)]
    else:
        orig_tokens = None
        return tok_text, False
    
    orig_text = " ".join(orig_tokens)
    final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging, logger)
    return final_text, True


if __name__ == "__main__":
    main()



'''
(for pre-training on short texts use --random_shift, for finetuning use mlm_batch_size -1)

CUDA_VISIBLE_DEVICES=4,5,6,7 python finetune_on_drop.py --do_eval --do_inference --examples_n_features_dir ./data/examples_n_features/ --eval_batch_size 1000 --init_weights_dir out_drop_finetune_bert  --output_dir preds
'''

# export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir=./log
