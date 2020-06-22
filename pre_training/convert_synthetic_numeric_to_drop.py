import sys, os, logging, argparse, jsonlines
from tqdm import tqdm
import ujson as json

sys.path.insert(1, os.path.join(sys.path[0], 'gen_bert'))  # to import from gen_bert dir
from create_examples_n_features import read_file, write_file

# create logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_synthetic_texual_to_drop(data):
    new_data, nums_used = {}, set()
    for d in data:
        _id, expr, val = d['id'], d['expr'].strip(), str(d['val']).strip()
        sample = {'passage': expr}
        qa_pair = {'question':'', 'query_id':_id}
        is_num = True
        try:
            float(val)
        except ValueError:
            is_num = False
            assert val in expr
        a = {'number': '', 'date': {'day': '', 'month': '', 'year': ''}, 'spans': []}
        if is_num:
            a['number'] = val
        else:
            a['spans'] = [val]
        qa_pair['answer'] = a
        sample['qa_pairs'] = [qa_pair]
        new_data[_id] = sample
        
        if d['check_domain']: 
            nums_used.update(d['args'])
    return new_data, nums_used


def main():
    parser = argparse.ArgumentParser(description='For converting synthetic numeric data to Drop format.')
    parser.add_argument("--data_jsonl", default='synthetic_for_drop.jsonl', type=str, 
                        help="The synthetic numeric data .jsonl file.")
    args = parser.parse_args()

    logger.info("Reading %s" % args.data_jsonl)
    data = read_file(args.data_jsonl)
    
    train_data = [d for d in data if d['split'] == 'train']
    dev_data   = [d for d in data if d['split'] != 'train']
        
    logger.info("Converting...")
    new_train_data, train_nums = convert_synthetic_texual_to_drop(train_data)
    new_dev_data, dev_nums = convert_synthetic_texual_to_drop(dev_data)
    
    assert train_nums.isdisjoint(dev_nums)
    
    for new_data, split in zip([new_train_data, new_dev_data], ['train', 'dev']):
        save_path = args.data_jsonl.replace('.jsonl', '') + f'_{split}_drop_format.json'
        logger.info("Saving %s" % save_path)
        write_file(new_data, save_path)

if __name__ == "__main__":
    main()

'''
python convert_synthetic_numeric_to_drop.py --data_jsonl ./data/synthetic_numeric.jsonl
'''