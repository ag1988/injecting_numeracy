import os, logging, argparse
from copy import deepcopy
from tqdm import tqdm
import ujson as json

# create logger
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

answer_type_map = {'intra_entity_superlative': 'number', 'inter_entity_superlative': 'spans', 
                   'intra_entity_simple_diff': 'number', 'intra_entity_subset': 'number', 
                   'inter_entity_sum': 'number', 'inter_entity_comparison': 'spans', 'select': 'number'}

def convert_synthetic_texual_to_drop(data):
    _data = deepcopy(data)
    for sample in _data.values():
        for qa_pair in sample['qa_pairs']:
            qa_pair['query_id'] = qa_pair['question_id']
            a = {'number': '', 'date': {'day': '', 'month': '', 'year': ''}, 'spans': []}
            if answer_type_map[qa_pair['generator']] == 'number':
                a['number'] = str(qa_pair['answer'])
            else:
                a['spans'].append(str(qa_pair['answer']))
            qa_pair['answer'] = a
            for key in list(qa_pair.keys()):
                if key not in {'query_id', 'question', 'answer'}:
                    del qa_pair[key]
    return _data


def main():
    parser = argparse.ArgumentParser(description='For converting synthetic texual data to Drop format.')
    parser.add_argument("--data_json", default='synthetic_textual_mixed_min3_max6_up0.7_train.json', type=str, 
                        help="The synthetic texual data .json file.")
    args = parser.parse_args()

    logger.info("Reading %s" % args.data_json)
    with open(args.data_json, encoding='utf8') as f:
        data = json.load(f)

    logger.info("Converting...")
    new_data = convert_synthetic_texual_to_drop(data)

    save_path = args.data_json.replace('.json', '') + '_drop_format.json'
    logger.info("Saving %s" % save_path)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False)

    
if __name__ == "__main__":
    main()

'''
python convert_synthetic_texual_to_drop.py --data_json ../../data/synthetic_textual_mixed_min3_max6_up0.7_train.json
python convert_synthetic_texual_to_drop.py --data_json ../../data/synthetic_textual_mixed_min3_max6_up0.7_dev.json
'''