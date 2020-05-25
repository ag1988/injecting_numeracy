
import argparse
import json
import numpy as np

from tqdm import tqdm

from generators.generator import Generator


def get_generators(args):
    if args.domain in ["history", "nfl"]:
        generators = {
            args.domain: Generator(
                vocab_file=f"generators/vocabularies/{args.domain}.json",
                templates_file="generators/templates/base.json",
                min_num_sentences=args.min_num_sentences,
                max_num_sentences=args.max_num_sentences,
                used_values_prob=args.used_values_prob
            )
        }
        num_passages = {
            args.domain: args.num_passages
        }
        num_eval_examples = {
            args.domain: args.num_eval_examples,
        }

    else:   # args.domain == "mixed"
        generators = {
            domain: Generator(
                vocab_file=f"generators/vocabularies/{domain}.json",
                templates_file="generators/templates/base.json",
                min_num_sentences=args.min_num_sentences,
                max_num_sentences=args.max_num_sentences,
                used_values_prob=args.used_values_prob
            )
            for domain in ["history", "nfl"]
        }
        num_history_passages = int(args.num_passages * 0.8)
        num_history_eval_examples = int(args.num_eval_examples * 0.8)
        num_passages = {
            "history": num_history_passages,
            "nfl": args.num_passages - num_history_passages,
        }
        num_eval_examples = {
            "history": num_history_eval_examples,
            "nfl": args.num_eval_examples - num_history_eval_examples,
        }

    return generators, num_passages, num_eval_examples


def add_passage_examples(examples, domain, passage_id, new_examples):
    examples[f"{domain}_{passage_id}"] = {
        "passage": new_examples[0].passage(),
        "qa_pairs": [
            {
                "question_id": f"{domain}_{passage_id}_{question_id}",
                "domain": domain,
                "generator": example.qa_generator,
                "question": example.question,
                "answer": example.answer,
                "expression": example.expression,
            }
            for question_id, example in enumerate(new_examples)
        ]
    }


def generate_passages(domain, generator, num_passages, print_examples):
    examples = {}
    example_count = 0
    empty_count = 0
    for passage_id in tqdm(range(num_passages)):
        new_examples = generator.generate_example()
        if not new_examples:
            empty_count += 1
            continue

        if print_examples:
            if len(new_examples) > 0:
                print(new_examples[0].get_context())
            for example in new_examples:
                print(example)

        add_passage_examples(examples, domain, passage_id, new_examples)
        example_count += len(new_examples)

    print(f"number of generated {domain} questions: {example_count}")
    print(f"cases of empty {domain} examples: {empty_count}")

    return examples, example_count, empty_count


def generate_passages_limited(domain, generator, total_num_examples):
    examples = {}
    example_count = 0
    empty_count = 0

    while example_count < total_num_examples:
        new_examples = generator.generate_example()
        if not new_examples:
            empty_count += 1
            continue

        num_new_examples = len(new_examples)
        num_examples_to_add = min(total_num_examples - example_count, num_new_examples)
        examples_to_add = new_examples[:num_examples_to_add]

        passage_id = len(examples)
        add_passage_examples(examples, domain, passage_id, examples_to_add)

        example_count += len(examples_to_add)

    assert example_count == total_num_examples, f"{example_count} != {total_num_examples}"
    print(f"number of generated {domain} questions: {example_count}")
    print(f"cases of empty {domain} examples: {empty_count}")

    return examples, example_count, empty_count


def main(args):
    #
    # initialize generator(s) and seed
    #
    if args.seed >= 0:
        np.random.seed(args.seed)

    generators, num_passages, num_eval_examples = get_generators(args)

    #
    # generate examples
    #
    examples = {}
    eval_examples = {}
    total_example_count = 0
    total_empty_count = 0
    total_eval_example_count = 0
    for domain in generators:
        generator = generators[domain]
        generator_num_passages = num_passages[domain]
        generator_num_eval_examples = num_eval_examples[domain]

        # TODO(mega): probably should generate the train set with 'generate_passages_limited' as well.
        generator_examples, example_count, empty_count = generate_passages(
            domain, generator, generator_num_passages,
            args.print_examples)
        examples.update(generator_examples)
        total_example_count += example_count
        total_empty_count += empty_count

        generator_eval_examples, eval_example_count, _ = generate_passages_limited(
            domain, generator, generator_num_eval_examples)
        eval_examples.update(generator_eval_examples)
        total_eval_example_count += eval_example_count

    assert total_eval_example_count == args.num_eval_examples

    print(f"total number of generated questions: {total_example_count}")
    print(f"total cases of empty examples: {total_empty_count}")
    print(f"total number of evaluation questions: {args.num_eval_examples}")

    #
    # store output
    #
    if args.output_file_base:
        with open(args.output_file_base + '_train.json', "w") as fd:
            json.dump(examples, fd, indent=4)
        print(f"stored generated train examples in: {args.output_file_base + '_train.json'}")

        with open(args.output_file_base + '_dev.json', "w") as fd:
            json.dump(eval_examples, fd, indent=4)
        print(f"stored generated dev examples in: {args.output_file_base + '_dev.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="example: python generate_examples.py --domain history --num_passages 1 "
                    "--min_num_sentences 3 --max_num_sentences 6 --used_values_prob 0.7 "
                    "--print_examples"
    )
    parser.add_argument('--domain', type=str, choices=["nfl", "history", "mixed"],
                        help='examples domain')
    parser.add_argument('--num_passages', type=int, default=20,
                        help='number of passages to generate')
    parser.add_argument('--num_eval_examples', type=int, default=0,
                        help='number of examples to save aside for evaluation')
    parser.add_argument('--min_num_sentences', type=int, default=3,
                        help='minimum number of sentences per example')
    parser.add_argument('--max_num_sentences', type=int, default=8,
                        help='maximum number of sentences per example')
    parser.add_argument('--used_values_prob', type=float, default=0.6,
                        help='probability of sampling previously instantiated values '
                             'rather than sampling from the vocabulary')
    parser.add_argument('--output_file_base', type=str, default='',
                        help='base path (without an extension) to output file')
    parser.add_argument('--print_examples', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=-1, help='random seed')
    args = parser.parse_args()

    main(args)

