import uuid, random, jsonlines, logging, argparse
from datetime import datetime, date, timedelta
from dateutil import relativedelta
import ujson as json
from tqdm import tqdm
import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer
from nltk.corpus import words, wordnet

# create logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# words with <= 2 wordpieces
nltk_words = [w.lower() for w in words.words() if len(bert_tokenizer.tokenize(w)) <= 2]


superlatives = {'max':["longest", "last", "highest", "largest", "most"], 
                'min':["shortest", "first", "smallest", "lowest", "least"]}

comparatives = {'less': ["fewer", "less", "before", "earlier", "smaller", "lower", "shorter"],
                'more': ["more", "later", "bigger", "higher", "longer", "larger", "greater", "taller"]}

date_superlatives = {'max':["last", "latest", "most recent", "youngest"], 
                    'min':["first", "earliest", "oldest", 'least recent']}

date_comparatives = {'less': ["before", "earlier", "older"],
                     'more': ['after', "later", "younger"]}


# def rand_expression(args):
#     # returns arithmetic expression, val
#     if len(args) == 1:
#         return str(args[0]), args[0]
#     split = random.randint(1, len(args)-1)
#     exp1, val1 = rand_expression(args[:split])
#     exp2, val2 = rand_expression(args[split:])
#     op = random.choice(['+', '-', '*']) #if len(str(val1*val2)) < 10 else random.choice(['+', '-'])
#     val = {'+': val1 + val2, '-': val1 - val2, '*': val1 * val2}[op]
#     return '(%s %s %s)' % (exp1, op, exp2), val # infix


def rand_float(x):
    # randomly add upto 2 decimal places
    precision = np.random.choice([0, 1, 2], p=[0.2, 0.4, 0.4])
    fractional_part = {0: 0, 1: random.randint(0, 9)*0.1, 2: random.randint(0, 99)*0.01}[precision]
    return x + fractional_part


def signed_expression(args):
    # returns signed combination, val
    expr, val = '', 0
    for a in args:
        sign = random.choice(['+', '-']) 
        val += {'+': a, '-': -a}[sign]
        expr += '%s %s ' % (sign, str(a))
    expr = expr[1:] if expr[0] == '+' else expr
    return expr.strip(), round(val, 2)


def min_max_avg_expression(args):
    # returns min/max expression, val
    expr, val = '', 0
    choice = random.randint(0,2)
    val = [max(args), min(args), round(sum(args)/len(args), 2)][choice]
    expr = ', '.join(map(str, args)).strip()
    expr = '%s(%s)' % ([random.choice(superlatives['max']), random.choice(superlatives['min']), 
                        'average'][choice], expr)
    return expr.strip(), val


def arg_min_max_expression(wrds, args):
    # returns argmin/argmax expression, val
    expr = ''
    for w, a in zip(wrds, args):
        expr += '%s %s, ' % (w, str(a))
    mn, mx, expr = min(args), max(args), expr[:-2].strip()
    max_or_min = random.randint(0,1)
    val = wrds[args.index(mx)] if max_or_min else wrds[args.index(mn)]
    expr = '%s(%s)' % ('argmax' if max_or_min else 'argmin', expr)
    return expr.strip(), val


def rand_percent():
    # returns argmin/argmax expression, val
    # sample 3-5 args
    wrds = [random.choice(nltk_words)
            for _ in range(np.random.choice([3, 4, 5], p=[0.2, 0.4, 0.4]))]
    args = []
    for p in np.random.dirichlet(np.ones(len(wrds)))*100:
        p = {0:float, 1: int}[random.randint(0,1)]((round(p, random.randint(1,2))))
        args.append(p)
    args[0] = round(100 - sum(args[1:]), 2)
    context = ''
    for w, a in zip(wrds, args):
        context += '%s %s%%, ' % (w, str(a))
    context = context[:-2].strip()
    n_q_args = min(np.random.choice([1, 2, 3], p=[0.4, 0.3, 0.3]), len(args) - 1)
    q_ids_wrds = random.sample(list(enumerate(wrds)), n_q_args)
    q_args, q_wrds = [], []
    for tup in q_ids_wrds:
        q_args.append(args[tup[0]]); q_wrds.append(tup[1])
    negate = random.choice(['', 'not '])
    q = 'percent %s' % negate + ', '.join(q_wrds)
    expr = q + ' :: ' + context
    val = {'': sum(q_args), 'not ': 100 - sum(q_args)}[negate]
    return expr.strip(), context.strip(), q.strip(), round(val, 2), args


def date_min_max(n_args=3):
    # returns min/max expression, val, args
    rds = [datetime.now() - timedelta(days=2018*365) * random.random() for _ in range(n_args)]
    day_diff_range = 30 if random.randint(0,1) else 150
    diffs = random.sample(range(1, day_diff_range+1), n_args-1)
    for i in range(1, len(rds)):
        rds[i] = rds[0] + random.choice([-1,1]) * timedelta(days=diffs[i-1])
    random.shuffle(rds)
    choices = [[rd.strftime("%d %B %Y"), rd.strftime("%B %d, %Y")][random.randint(0, 1)] for rd in rds]
    expr, max_or_min = '; '.join(choices).strip(), random.randint(0,1)
    rd = [max(rds), min(rds)][max_or_min]
    val = choices[rds.index(rd)]
    expr = '%s(%s)' % ([random.choice(date_superlatives['max']), 
                        random.choice(date_superlatives['min'])][max_or_min], expr)
    return expr.strip(), val, choices


def date_diff(typ=''):
    # returns expression, val, args
    typ = typ if typ else random.choice(['years', 'months', 'days'])
    rds = [datetime.now() - timedelta(days=2018*365) * random.random() for _ in range(2)]
    if typ in ['months', 'days']:
        diff = timedelta(days=60) if random.randint(0,1) else timedelta(days=200)
        rds[1] = rds[0] + random.choice([-1,1]) * diff * random.random()
    random.shuffle(rds)
    choices = [[rd.strftime("%d %B %Y"), rd.strftime("%B %d, %Y")][random.randint(0, 1)] for rd in rds]
    # DROP: yr diff depends only on yr vals, similarly for months within an yr
    diff_years = max(rds).year - min(rds).year
    diff_months = diff_years*12 + (max(rds).month - min(rds).month)
    diff_days = (max(rds) - min(rds)).days
    val = {'years':diff_years, 'months':diff_months, 'days':diff_days}[typ]
    expr = '; '.join(choices).strip()
    expr = 'difference in %s(%s)' % (typ, expr)
    return expr.strip(), val, choices


def main():
    parser = argparse.ArgumentParser(description='For generating synthetic numeric data.')
    parser.add_argument("--num_samples", default=1e6, type=float, help="Total number of samples to generate.")
    parser.add_argument("--num_dev_samples", default=1e4, type=float, help="Num of samples to keep aside for dev set.")
    parser.add_argument("--output_jsonl", default='./data/synthetic_numeric.jsonl', type=str, 
                        help="Output synthetic numeric data .jsonl file.")
    pargs = parser.parse_args()
    
    # split the domain
    domain, train_number_range, dev_number_range = int(2e4), [], []
    for i in range(domain):
        x = train_number_range if random.random() < 0.8 else dev_number_range
        x.append(i)

    n_examples, n_dev, q_types = int(pargs.num_samples), int(pargs.num_dev_samples), 6
    discrete_ops_data, n_iters = [], n_examples // q_types
    train_args, dev_args = set(), set()

    logger.info(f"Creating {n_examples} samples...")
    for i_s in tqdm(range(n_iters)):
        # decide train/dev split
        split = 'train' if i_s < n_iters - (n_dev // q_types) else 'dev'
        rng = {'train': train_number_range, 'dev': dev_number_range}[split]
        args = [random.choice(rng) for _ in range(np.random.choice([2, 3, 4], p=[1/3]*3))]
        # with 50% prob add rand fraction
        args = list(map(rand_float, args)) if random.randint(0,1) else args
        train_args.update(args) if split == 'train' else dev_args.update(args)

        wrds = [random.choice(nltk_words) for _ in range(len(args))]

        expr, val = signed_expression(args)
        d1 = {'id': str(uuid.uuid4().hex), 'expr': expr, 'val': val, 'args': args, 
              'type': 'signed_expression', 'check_domain':True, 'split': split}

        expr, val = min_max_avg_expression(args)
        d2 = {'id': str(uuid.uuid4().hex), 'expr': expr, 'val': val, 'args': args, 
              'type': 'min_max_avg_expression', 'check_domain':True, 'split': split}

        expr, val = arg_min_max_expression(wrds, args)
        d3 = {'id': str(uuid.uuid4().hex), 'expr': expr, 'val': val, 'args': args, 
              'type': 'arg_min_max_expression', 'check_domain':True, 'split': split}

        expr, val, date_args = date_min_max(n_args=len(args))
        d4 = {'id': str(uuid.uuid4().hex), 'expr': expr, 'val': val, 'args': date_args, 
              'type': 'date_min_max', 'check_domain':False, 'split': split}

        expr, val, date_args = date_diff()
        d5 = {'id': str(uuid.uuid4().hex), 'expr': expr, 'val': val, 'args': date_args, 
              'type': 'date_diff', 'check_domain':False, 'split': split}

        expr, context, qn, val, args = rand_percent()
        d6 = {'id': str(uuid.uuid4().hex), 'expr': expr, 'val': val, 'args': args, 'ques': qn, 
              'context': context, 'type': 'percent', 'check_domain':False, 'split': split}

        discrete_ops_data += [d1, d2, d3, d4, d5, d6]

    assert train_args.isdisjoint(dev_args) # trn, dev args are disjoint

    with jsonlines.open(pargs.output_jsonl, mode='w') as writer:
        writer.write_all(discrete_ops_data)
    

if __name__ == "__main__":
    main()
    
    
'''
python gen_numeric_data.py --num_samples 1e6 --num_dev_samples 1e4 --output_jsonl ../data/synthetic_numeric.jsonl
'''