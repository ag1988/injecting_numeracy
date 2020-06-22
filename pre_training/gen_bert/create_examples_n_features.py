import os, json, copy, string, itertools, logging, pickle, argparse, jsonlines
import numpy as np
from random import choice
from decimal import Decimal
from typing import Any, Dict, List, Tuple, Callable
import collections
from collections import defaultdict

from allennlp.common.file_utils import cached_path
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.dataset_readers.reading_comprehension.util import IGNORED_TOKENS, STRIPPED_CHARACTERS
from tqdm import tqdm

from pytorch_pretrained_bert.tokenization import BertTokenizer
from w2n import word_to_num

# create logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

START_TOK, END_TOK, SPAN_SEP, IGNORE_IDX, MAX_DECODING_STEPS = '@', '\\', ';', 0, 20 

class DropExample(object):
    def __init__(self,
                 qas_id,
                 passage_id,
                 question_tokens,
                 passage_tokens,
                 orig_question_tokens,
                 orig_passage_tokens,
                 numbers_in_passage=None,
                 number_indices=None,
                 answer_type=None,
                 number_of_answer=None,
                 passage_spans=None,
                 question_spans=None,
                 answer_annotations=None,
                 answer_texts=None
                 ):
        self.qas_id = qas_id
        self.passage_id = passage_id
        self.question_tokens = question_tokens
        self.passage_tokens = passage_tokens
        self.orig_question_tokens = orig_question_tokens
        self.orig_passage_tokens = orig_passage_tokens
        self.numbers_in_passage = numbers_in_passage
        self.number_indices = number_indices
        self.answer_type = answer_type
        self.number_of_answer = number_of_answer
        self.passage_spans = passage_spans
        self.question_spans = question_spans
        self.answer_annotations = answer_annotations
        self.answer_texts = answer_texts
        
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % self.qas_id
        s += ", \nquestion: %s" % (" ".join(self.question_tokens))
        s += ", \npassage: %s" % (" ".join(self.passage_tokens))
        if self.numbers_in_passage:
            s += ", \nnumbers_in_passage: {}".format(self.numbers_in_passage)
        if self.number_indices:
            s += ", \nnumber_indices: {}".format(self.number_indices)
        if self.answer_type:
            s += ", \nanswer_type: {}".format(self.answer_type)
        if self.number_of_answer:
            s += ", \nnumber_of_answer: {}".format(self.number_of_answer)
        if self.passage_spans:
            s += ", \npassage_spans: {}".format(self.passage_spans)
        if self.question_spans:
            s += ", \nquestion_spans: {}".format(self.question_spans)
        if self.answer_annotations:
            s += ", \nanswer_annotations: {}".format(self.answer_annotations)
        if self.answer_type:
            s += ", \nanswer_type: {}".format(self.answer_type)
        if self.answer_texts:
            s += ", \nanswer_texts: {}".format(self.answer_texts)
        return s


class DropFeatures(object):
    def __init__(self,
                 unique_id,
                 example_index,
                 max_seq_length,
                 tokens,
                 que_token_to_orig_map,
                 doc_token_to_orig_map,
                 input_ids,
                 input_mask,
                 segment_ids,
                 number_indices,
                 start_indices=None,
                 end_indices=None,
                 number_of_answers=None,
                 decoder_label_ids=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.max_seq_length = max_seq_length
        self.tokens = tokens
        self.que_token_to_orig_map = que_token_to_orig_map
        self.doc_token_to_orig_map = doc_token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.number_indices = number_indices
        self.start_indices = start_indices
        self.end_indices = end_indices
        self.number_of_answers = number_of_answers
        self.decoder_label_ids = decoder_label_ids

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "unique_id: %s" % (self.unique_id)
        s += ", \nnumber_indices: {}".format(self.number_indices)
        if self.start_indices:
            s += ", \nstart_indices: {}".format(self.start_indices)
        if self.end_indices:
            s += ", \nend_indices: {}".format(self.end_indices)
        if self.number_of_answers:
            s += ", \nnumber_of_answers: {}".format(self.number_of_answers)
        return s


WORD_NUMBER_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                   "five": 5, "six": 6, "seven": 7, "eight": 8,
                   "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                   "thirteen": 13, "fourteen": 14, "fifteen": 15,
                   "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}


def split_token_by_delimiter(token: Token, delimiter: str) -> List[Token]:
    split_tokens = []
    char_offset = token.idx
    for sub_str in token.text.split(delimiter):
        if sub_str:
            split_tokens.append(Token(text=sub_str, idx=char_offset))
            char_offset += len(sub_str)
        split_tokens.append(Token(text=delimiter, idx=char_offset))
        char_offset += len(delimiter)
    if split_tokens:
        split_tokens.pop(-1)
        char_offset -= len(delimiter)
        return split_tokens
    else:
        return [token]


def split_tokens_by_hyphen(tokens: List[Token]) -> List[Token]:
    hyphens = ["-", "–", "~"]
    new_tokens: List[Token] = []

    for token in tokens:
        if any(hyphen in token.text for hyphen in hyphens):
            unsplit_tokens = [token]
            split_tokens: List[Token] = []
            for hyphen in hyphens:
                for unsplit_token in unsplit_tokens:
                    if hyphen in token.text:
                        split_tokens += split_token_by_delimiter(unsplit_token, hyphen)
                    else:
                        split_tokens.append(unsplit_token)
                unsplit_tokens, split_tokens = split_tokens, []
            new_tokens += unsplit_tokens
        else:
            new_tokens.append(token)

    return new_tokens


class DropReader(object):
    def __init__(self,
                 max_n_samples: int = -1,
                 include_more_numbers: bool = False,
                 max_number_of_answer: int = 8,
                 include_multi_span: bool = True,
                 logger = None
                 ) -> None:
        super().__init__()
        self._tokenizer = WordTokenizer()
        self.include_more_numbers = include_more_numbers
        self.max_number_of_answer = max_number_of_answer
        self.include_multi_span = include_multi_span
        self.logger = logger
        self.max_n_samples = max_n_samples

    def _read(self, file_path: str):
        self.logger.info('creating examples')
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        self.logger.info("Reading file at %s", file_path)
        dataset = read_file(file_path)
        examples, skip_count = [], 0
        for passage_id, passage_info in tqdm(dataset.items()): 
            passage_text = passage_info["passage"]
            for question_answer in passage_info["qa_pairs"]:
                question_id = question_answer["query_id"]
                question_text = question_answer["question"].strip()
                answer_annotations = []
                if "answer" in question_answer:
                    answer_annotations.append(question_answer["answer"])
                if "validated_answers" in question_answer:
                    answer_annotations += question_answer["validated_answers"]
                if not answer_annotations:
                    # for test set use a dummy
                    answer_annotations = [{'number':'0', 'date':{"month":'', "day":'', "year":''}, 'spans':[]}]
                example = self.text_to_example(question_text, passage_text, passage_id, question_id, answer_annotations)
                if example is not None:
                    examples.append(example)
                else:
                    skip_count += 1
            if self.max_n_samples > 0 and len(examples) >= self.max_n_samples:
                break
        self.logger.info(f"Skipped {skip_count} examples, kept {len(examples)} examples.")
        return examples

    
    def text_to_example(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         passage_id: str,
                         question_id: str,
                         answer_annotations: List[Dict] = None,
                         passage_tokens: List[Token] = None):
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
            passage_tokens = split_tokens_by_hyphen(passage_tokens)
        question_tokens = self._tokenizer.tokenize(question_text)
        question_tokens = split_tokens_by_hyphen(question_tokens)

        orig_passage_tokens = copy.deepcopy(passage_tokens)
        orig_question_tokens = copy.deepcopy(question_tokens)
        
        answer_type: str = None
        answer_texts: List[str] = []
        number_of_answer: int = None
        if answer_annotations:
            # Currently we only use the first annotated answer here, but actually this doesn't affect
            # the training, because we only have one annotation for the train set.
            answer_type, answer_texts = self.extract_answer_info_from_annotation(answer_annotations[0])
            number_of_answer = min(len(answer_texts), self.max_number_of_answer)
        
        if answer_type is None or (answer_type == 'spans' and len(answer_texts) > 1 and not self.include_multi_span): 
            return None # multi-span
        
       # Tokenize the answer text in order to find the matched span based on token
        tokenized_answer_texts = []
        for answer_text in answer_texts:
            answer_tokens = self._tokenizer.tokenize(answer_text)
            answer_tokens = split_tokens_by_hyphen(answer_tokens)
            tokenized_answer_texts.append(answer_tokens)

        # During our pre-training, we dont handle "12,000", etc and hence
        # are normalizing the numbers in passage, question and answer texts.
        def _normalize_number_tokens(tokens: List[Token]) -> List[Token]:
            # returns a new list of tokens after normalizing the numeric tokens
            new_tokens = []
            for token in tokens:
                number = self.convert_word_to_number(token.text, self.include_more_numbers)
                if number is not None:
                    new_tokens.append(Token(str(number)))
                else:
                    new_tokens.append(Token(token.text))
            return new_tokens
        passage_tokens = _normalize_number_tokens(passage_tokens)
        question_tokens = _normalize_number_tokens(question_tokens)
        tokenized_answer_texts = list(map(_normalize_number_tokens, tokenized_answer_texts))
        normalized_answer_texts = list(map(lambda l: ' '.join([t.text for t in l]), tokenized_answer_texts))
        
        # for multi-span we remove duplicates and arrange by order of occurance in passage
        unique_answer_texts = sorted(set(normalized_answer_texts), key=normalized_answer_texts.index)
        passage_text = ' '.join([token.text for token in passage_tokens])
        arranged_answer_texts = sorted(unique_answer_texts, key=passage_text.find)
        normalized_answer_texts = arranged_answer_texts
        
        numbers_in_passage = []
        number_indices = []
        for token_index, token in enumerate(passage_tokens):
            number = self.convert_word_to_number(token.text, self.include_more_numbers)
            if number is not None:
                numbers_in_passage.append(number)
                number_indices.append(token_index)

        valid_passage_spans = \
            self.find_valid_spans(passage_tokens, tokenized_answer_texts) if tokenized_answer_texts else []
        valid_question_spans = \
            self.find_valid_spans(question_tokens, tokenized_answer_texts) if tokenized_answer_texts else []
        number_of_answer = None if valid_passage_spans == [] and valid_question_spans == [] else number_of_answer

        return DropExample(
            qas_id=question_id,
            passage_id=passage_id,
            question_tokens=[token.text for token in question_tokens],
            passage_tokens=[token.text for token in passage_tokens],
            orig_question_tokens=[token.text for token in orig_question_tokens],
            orig_passage_tokens=[token.text for token in orig_passage_tokens],
            numbers_in_passage=numbers_in_passage,
            number_indices=number_indices,
            answer_type=answer_type,
            number_of_answer=number_of_answer,
            passage_spans=valid_passage_spans,
            question_spans=valid_question_spans,
            answer_annotations=answer_annotations,
            answer_texts=normalized_answer_texts)

    @staticmethod
    def extract_answer_info_from_annotation(answer_annotation: Dict[str, Any]) -> Tuple[str, List[str]]:
        answer_type = None
        if answer_annotation["spans"]:
            answer_type = "spans"
        elif answer_annotation["number"]:
            answer_type = "number"
        elif any(answer_annotation["date"].values()):
            answer_type = "date"

        answer_content = answer_annotation[answer_type] if answer_type is not None else None

        answer_texts: List[str] = []
        if answer_type is None:  # No answer
            pass
        elif answer_type == "spans":
            # answer_content is a list of string in this case
            answer_texts = answer_content
        elif answer_type == "date":
            # answer_content is a dict with "month", "day", "year" as the keys
            date_tokens = [answer_content[key]
                           for key in ["month", "day", "year"] if key in answer_content and answer_content[key]]
            answer_texts = [' '.join(date_tokens)]
        elif answer_type == "number":
            # answer_content is a string of number
            answer_texts = [answer_content]
        return answer_type, answer_texts

    @staticmethod
    def convert_word_to_number(word: str, try_to_include_more_numbers=False, normalized_tokens=None, token_index=None):
        """
        Currently we only support limited types of conversion.
        """
        if try_to_include_more_numbers:
            # strip all punctuations from the sides of the word, except for the negative sign
            punctruations = string.punctuation.replace('-', '')
            word = word.strip(punctruations)
            # some words may contain the comma as deliminator
            word = word.replace(",", "")
            # word2num will convert hundred, thousand ... to number, but we skip it.
            if word in ["hundred", "thousand", "million", "billion", "trillion"]:
                return None
            try:
                number = word_to_num(word)
            except ValueError:
                try:
                    number = int(word)
                except ValueError:
                    try:
                        number = float(word)
                    except ValueError:
                        number = None
            return number
        else:
            no_comma_word = word.replace(",", "")
            if no_comma_word in WORD_NUMBER_MAP:
                number = WORD_NUMBER_MAP[no_comma_word]
            else:
                try:
                    number = int(no_comma_word)
                except ValueError:
                    number = None
            return number

    @staticmethod
    def find_valid_spans(passage_tokens: List[Token],
                         answer_texts: List[List[Token]]) -> List[Tuple[int, int]]:
        normalized_tokens = [token.text.lower().strip(STRIPPED_CHARACTERS) for token in passage_tokens]
        word_positions: Dict[str, List[int]] = defaultdict(list)
        for i, token in enumerate(normalized_tokens):
            word_positions[token].append(i)
        spans = []
        for answer_text in answer_texts:
            answer_tokens = [token.text.lower().strip(STRIPPED_CHARACTERS) for token in answer_text]
            num_answer_tokens = len(answer_tokens)
            if answer_tokens[0] not in word_positions:
                continue
            for span_start in word_positions[answer_tokens[0]]:
                span_end = span_start  # span_end is _inclusive_
                answer_index = 1
                while answer_index < num_answer_tokens and span_end + 1 < len(normalized_tokens):
                    token = normalized_tokens[span_end + 1]
                    if answer_tokens[answer_index].strip(STRIPPED_CHARACTERS) == token:
                        answer_index += 1
                        span_end += 1
                    elif token in IGNORED_TOKENS:
                        span_end += 1
                    else:
                        break
                if num_answer_tokens == answer_index:
                    spans.append((span_start, span_end))
        return spans

def split_digits(wps: List[str]) -> List[str]:
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

def convert_answer_spans(spans, orig_to_tok_index, all_len, all_tokens):
    tok_start_positions, tok_end_positions = [], []
    for span in spans:
        start_position, end_position = span[0], span[1]
        tok_start_position = orig_to_tok_index[start_position]
        if end_position + 1 >= len(orig_to_tok_index):
            tok_end_position = all_len - 1
        else:
            tok_end_position = orig_to_tok_index[end_position + 1] - 1
        if tok_start_position < len(all_tokens) and tok_end_position < len(all_tokens):
            tok_start_positions.append(tok_start_position)
            tok_end_positions.append(tok_end_position)
    return tok_start_positions, tok_end_positions


def convert_examples_to_features(examples, tokenizer, max_seq_length, max_decoding_steps, indiv_digits=True, logger=None):
    """Loads a data file into a list of `InputBatch`s."""

    logger.info('creating features')
    unique_id = 1000000000
    skip_count, truncate_count = 0, 0
    
    tokenize = (lambda s: split_digits(tokenizer.tokenize(s))) if indiv_digits else tokenizer.tokenize
    
    features, all_qp_lengths = [], []
    for (example_index, example) in enumerate(tqdm(examples)):
        que_tok_to_orig_index = []
        que_orig_to_tok_index = []
        all_que_tokens = []
        for (i, token) in enumerate(example.question_tokens):
            que_orig_to_tok_index.append(len(all_que_tokens))
            sub_tokens = tokenize(token)
            que_tok_to_orig_index += [i]*len(sub_tokens)
            all_que_tokens += sub_tokens
            
        doc_tok_to_orig_index = []
        doc_orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.passage_tokens):
            doc_orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenize(token)
            doc_tok_to_orig_index += [i]*len(sub_tokens)
            all_doc_tokens += sub_tokens
            
        # The -3 accounts for [CLS], [SEP] and [SEP]
        # Truncate the passage according to the max sequence length
        max_tokens_for_doc = max_seq_length - len(all_que_tokens) - 3
        all_doc_len = len(all_doc_tokens)
        if all_doc_len > max_tokens_for_doc:
            all_doc_tokens = all_doc_tokens[:max_tokens_for_doc]
            truncate_count += 1
        all_qp_lengths.append(len(all_que_tokens) + all_doc_len + 3)
        
        query_tok_start_positions, query_tok_end_positions = \
            convert_answer_spans(example.question_spans, que_orig_to_tok_index, len(all_que_tokens), all_que_tokens)

        passage_tok_start_positions, passage_tok_end_positions = \
            convert_answer_spans(example.passage_spans, doc_orig_to_tok_index, all_doc_len, all_doc_tokens)

        tok_number_indices = []
        # these are the _starting_ indices of numeric wps of original number token
        for index in example.number_indices:
            if index != -1:
                tok_index = doc_orig_to_tok_index[index]
                if tok_index < len(all_doc_tokens):
                    tok_number_indices.append(tok_index)
            else:
                tok_number_indices.append(-1)

        tokens, segment_ids = [], []
        que_token_to_orig_map = {}
        doc_token_to_orig_map = {}
        tokens.append("[CLS]")
        for i in range(len(all_que_tokens)):
            que_token_to_orig_map[len(tokens)] = que_tok_to_orig_index[i]
            tokens.append(all_que_tokens[i])
        tokens.append("[SEP]")
        segment_ids += [0]*len(tokens)

        for i in range(len(all_doc_tokens)):
            doc_token_to_orig_map[len(tokens)] = doc_tok_to_orig_index[i]
            tokens.append(all_doc_tokens[i])
        tokens.append("[SEP]")
        segment_ids += [1]*(len(tokens) - len(segment_ids))
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # segment_ids are 0 for toks in [CLS] Q [SEP] and 1 for P [SEP]
        assert len(segment_ids) == len(input_ids)
        
        # we expect the generative head to output the wps of the joined answer text
        answer_text = (SPAN_SEP+' ').join(example.answer_texts).strip()
        dec_toks = [START_TOK] + tokenize(answer_text) + [END_TOK]
        # ending the number with a '\\' to signify end
        dec_toks = dec_toks[:max_decoding_steps]
        dec_ids = tokenizer.convert_tokens_to_ids(dec_toks)
        decoder_label_ids = dec_ids + [IGNORE_IDX]*(max_decoding_steps - len(dec_ids))
        # IGNORE_IDX is ignored while computing the loss
        
        number_indices = []  # again these are the starting indices of wps of an orig number token
        doc_offset = len(all_que_tokens) + 2
        que_offset = 1
        for tok_number_index in tok_number_indices:
            if tok_number_index != -1:
                number_index = tok_number_index + doc_offset
                number_indices.append(number_index)
            else:
                number_indices.append(-1)

        start_indices, end_indices, number_of_answers = [], [], []
        # Shift the answer span indices according to [CLS] Q [SEP] P [SEP] structure
        if passage_tok_start_positions != [] and passage_tok_end_positions != []:
            for tok_start_position, tok_end_position in zip(passage_tok_start_positions, 
                                                            passage_tok_end_positions):
                start_indices.append(tok_start_position + doc_offset)
                end_indices.append(tok_end_position + doc_offset)
        
        if query_tok_start_positions != [] and query_tok_end_positions != []:
            for tok_start_position, tok_end_position in zip(query_tok_start_positions, 
                                                            query_tok_end_positions):
                start_indices.append(tok_start_position + que_offset)
                end_indices.append(tok_end_position + que_offset)

        if start_indices != [] and end_indices != []:
            assert example.number_of_answer is not None
            number_of_answers.append(example.number_of_answer - 1)
            
        features.append(DropFeatures(
                unique_id=unique_id,
                example_index=example_index,
                max_seq_length=max_seq_length,
                tokens=tokens,
                que_token_to_orig_map=que_token_to_orig_map,
                doc_token_to_orig_map=doc_token_to_orig_map,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                number_indices=number_indices,
                start_indices=start_indices,
                end_indices=end_indices,
                number_of_answers=number_of_answers,
                decoder_label_ids=decoder_label_ids))
        unique_id += 1
       
    logger.info(f"Skipped {skip_count} features, truncated {truncate_count} features, kept {len(features)} features.")
    return features, all_qp_lengths

def read_file(file):
    if file.endswith('jsonl'):
        with jsonlines.open(file, 'r') as reader:
            return [d for d in reader.iter()]
    
    if file.endswith('json'):
        with open(file, encoding='utf8') as f:
            return json.load(f)

    if any([file.endswith(ext) for ext in ['pkl', 'pickle', 'pck', 'pcl']]):
        with open(file, 'rb') as f:
            return pickle.load(f)

def write_file(data, file):
    if file.endswith('jsonl'):
        with jsonlines.open(file, mode='w') as writer:
            writer.write_all(data)

    if file.endswith('json'):
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    
    if any([file.endswith(ext) for ext in ['pkl', 'pickle', 'pck', 'pcl']]):
        with open(file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    parser = argparse.ArgumentParser(description='Used to make examples, features from DROP-like dataset.')
    parser.add_argument("--drop_json", default='drop_dataset_dev.json', type=str, help="The eval .json file.")
    parser.add_argument("--split", default='eval', type=str, help="data split: train/eval.")
    parser.add_argument("--max_n_samples", default=-1, type=int, help="max num samples to process.")
    parser.add_argument("--output_dir", default='data/examples_n_features', type=str, help="Output dir.")
    parser.add_argument("--max_seq_length", default=512, type=int, help="max seq len of [cls] q [sep] p [sep]")
    parser.add_argument("--max_decoding_steps", default=20, type=int, help="max tokens to be generated by decoder")
    
    args = parser.parse_args()
    
    drop_reader = DropReader(max_n_samples=args.max_n_samples,
                             include_more_numbers=True,
                             max_number_of_answer=8,
                             include_multi_span=True,
                             logger=logger)

    examples = drop_reader._read(args.drop_json)

    features, lens = convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        max_decoding_steps=args.max_decoding_steps,
        indiv_digits=True,
        logger=logger)

    
    os.makedirs(args.output_dir, exist_ok=True)
    split = 'train' if args.split == 'train' else 'eval'
    write_file(examples, args.output_dir + '/%s_examples.pkl' % split)
    write_file(features, args.output_dir + '/%s_features.pkl' % split)

if __name__ == "__main__":
    main()

'''
# drop  (MAX_DECODING_STEPS = 20)
python create_examples_n_features.py --split train --drop_json ../../data/drop_dataset_train.json --output_dir data/examples_n_features --max_seq_length 512
python create_examples_n_features.py --split eval --drop_json ../../data/drop_dataset_dev.json --output_dir data/examples_n_features --max_seq_length 512

# texual synthetic data:  (MAX_DECODING_STEPS = 20)
python create_examples_n_features.py --split train --drop_json ../../data/synthetic_textual_mixed_min3_max6_up0.7_train_drop_format.json --output_dir data/examples_n_features_syntext --max_seq_length 160 --max_n_samples -1
python create_examples_n_features.py --split eval --drop_json ../../data/synthetic_textual_mixed_min3_max6_up0.7_dev_drop_format.json --output_dir data/examples_n_features_syntext --max_seq_length 160

# numeric data (use MAX_DECODING_STEPS = 11):
python create_examples_n_features.py --split train --drop_json ../../data/synthetic_numeric_train_drop_format.json --output_dir data/examples_n_features_numeric --max_seq_length 50 --max_decoding_steps 11 --max_n_samples -1
python create_examples_n_features.py --split eval --drop_json ../../data/synthetic_numeric_dev_drop_format.json --output_dir data/examples_n_features_numeric --max_seq_length 50 --max_decoding_steps 11

# numeric data without DT (use indiv_digits=False, MAX_DECODING_STEPS = 11):
python create_examples_n_features.py --split train --drop_json ../../data/synthetic_numeric_train_drop_format.json --output_dir data/examples_n_features_numeric_wp --max_seq_length 42 --max_decoding_steps 11 --max_n_samples -1
python create_examples_n_features.py --split eval --drop_json ../../data/synthetic_numeric_dev_drop_format.json --output_dir data/examples_n_features_numeric_wp --max_seq_length 42 --max_decoding_steps 11
'''

