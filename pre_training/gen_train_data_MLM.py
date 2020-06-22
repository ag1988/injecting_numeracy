from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
import collections, jsonlines

from random import random, randrange, randint, shuffle, choice, sample
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np
import ujson as json


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        if (whole_word_mask and len(cand_indices) >= 1 and token.startswith("##")):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    shuffle(cand_indices)
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
#         # If adding a whole-word mask would exceed the maximum number of
#         # predictions, then just skip this candidate.
#         if len(masked_lms) + len(index_set) > num_to_mask:
#             continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
            
        op = np.random.choice(['mask', 'orig', 'random'], p=[0.8, 0.1, 0.1])
        for index in index_set:
            covered_indexes.add(index)

            # 80% of the time, replace with [MASK]
            if op == 'mask':
                masked_token = "[MASK]"
            elif op == 'orig':
                # 10% of the time, keep original
                masked_token = tokens[index]
            else:
                # 10% of the time, replace with random word
                masked_token = choice(vocab_list)
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
            tokens[index] = masked_token

#     assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    return tokens, mask_indices, masked_token_labels


def create_instances_from_document(
        document, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    # document is a list of toknzd sents
    # Account for [CLS], [SEP]
    max_num_tokens = max_seq_length - 2

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random() < short_seq_prob:
        target_seq_length = randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment) # append to sents to current_chunk until target_seq_length
        current_length += len(segment)

        if i == len(document) - 1 or current_length >= target_seq_length:
            tokens = sum(current_chunk, [])  
            truncate_seq(tokens, max_num_tokens)
            if tokens:
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                # The segment IDs are 0 for the [CLS] token, the A tokens and the first [SEP]
                # They are 1 for the B tokens and the final [SEP]
                segment_ids = [0 for _ in range(len(tokens))]

                tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list)

                instance = {
                    "tokens": tokens,
                    "segment_ids": segment_ids,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_labels": masked_lm_labels}
                instances.append(instance)
            # reset and start new chunk
            current_chunk = []
            current_length = 0
        i += 1

    return instances


def truncate_seq(tokens, max_num_tokens):
    """Truncates a list to a maximum sequence length"""
    while len(tokens) > max_num_tokens:
        assert len(tokens) >= 1
        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            del tokens[0]
        else:
            tokens.pop()


def split_digits(wps):
    # further split numeric wps
    toks = []
    for wp in wps:
        if set(wp).issubset(set('#0123456789')) and set(wp) != {'#'}: # numeric wp - split digits
            for i, dgt in enumerate(list(wp.replace('#', ''))):
                prefix = '##' if (wp.startswith('##') or i > 0) else ''
                toks.append(prefix + dgt)
        else:
            toks.append(wp)
    return toks


def main():
    parser = ArgumentParser(description='''Creates whole-word-masked instances for MLM task. MLM_paras.jsonl is a list of dicts each with a key 'sents' and val a list of sentences of some document.\n
    Usage: python gen_train_data_MLM.py --train_corpus MLM_paras.jsonl --bert_model bert-base-uncased --output_dir data/MLM_train/ --do_lower_case --max_predictions_per_seq 65 --do_whole_word_mask --digitize ''')
    
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True,
                        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
                                 "bert-base-multilingual", "bert-base-chinese"])
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--do_whole_word_mask", action="store_true",
                        help="Whether to use whole word masking rather than per-WordPiece masking.")
    parser.add_argument("--digitize", action="store_true",
                        help="Whether to further split a numeric wp into digits.")
    parser.add_argument("--epochs_to_generate", type=int, default=1,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=65,
                        help="Maximum number of tokens to mask in each sequence")
    
    args = parser.parse_args()
    
    
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    digit_tokenize = lambda s: split_digits(tokenizer.tokenize(s))
    vocab_list = list(tokenizer.vocab.keys())
    
    with jsonlines.open(args.train_corpus, 'r') as reader:
        data = [d for d in tqdm(reader.iter())]
    docs = []
    for d in tqdm(data):
        doc = [digit_tokenize(sent) if args.digitize else tokenizer.tokenize(sent)
               for sent in d['sents']]
        if doc: docs.append(doc)
        
    # docs is a list of docs - each doc is a list of sents - each sent is list of tokens
    args.output_dir.mkdir(exist_ok=True)
    for epoch in trange(args.epochs_to_generate, desc="Epoch"):
        epoch_filename = args.output_dir / f"epoch_{epoch}.jsonl"
        num_instances = 0
        with epoch_filename.open('w') as epoch_file:
            for doc_idx in trange(len(docs), desc="Document"):
                doc_instances = create_instances_from_document(
                    docs[doc_idx], max_seq_length=args.max_seq_len, short_seq_prob=args.short_seq_prob,
                    masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq,
                    whole_word_mask=args.do_whole_word_mask, vocab_list=vocab_list)
                for instance in doc_instances:
                    epoch_file.write(json.dumps(instance) + '\n')
                    num_instances += 1
        metrics_file = args.output_dir / f"epoch_{epoch}_metrics.jsonl"
        with metrics_file.open('w') as metrics_file:
            metrics = {
                "num_training_examples": num_instances,
                "max_seq_len": args.max_seq_len
            }
            metrics_file.write(json.dumps(metrics))


if __name__ == '__main__':
    main()

'''python gen_train_data_MLM.py --train_corpus ./data/MLM_paras.jsonl --bert_model bert-base-uncased --output_dir ./data/MLM_train/ --do_lower_case --max_predictions_per_seq 65 --digitize'''