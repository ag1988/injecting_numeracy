
import json
import spacy

from generators.element_type import DependentElementTypes, IndependentElementTypes
from generators.generator_utils import *
from generators.state import State
from generators.qa_generator import \
    QuestionAnswerGeneratorSelect, \
    QuestionAnswerGeneratorIntraEntitySimpleDiff, \
    QuestionAnswerGeneratorIntraEntitySubset, \
    QuestionAnswerGeneratorInterEntityComparison, \
    QuestionAnswerGeneratorInterEntitySuperlative, \
    QuestionAnswerGeneratorIntraEntitySuperlative, \
    QuestionAnswerGeneratorInterEntitySum


class SyntheticExample(object):
    def __init__(self, qa_generator, sentences, states, question, answer, expression):
        self.qa_generator = qa_generator
        self.sentences = sentences
        self.states = states
        self.question = question
        self.answer = answer
        self.expression = expression

    def __str__(self):
        return f"generator: {self.qa_generator}\n" \
            f"question: {self.question}\n" \
            f"answer: {self.answer}\n" \
            f"expression: {self.expression}\n"

    def get_context(self):
        return f"context: {' '.join(self.sentences)}\n"

    def passage(self):
        return ' '.join(self.sentences)


class Generator(object):
    def __init__(self, vocab_file, templates_file,
                 min_num_sentences=2, max_num_sentences=6, used_values_prob=0.7,
                 debug=False):
        self._debug = debug

        self._load_vocab(vocab_file)
        self._load_templates(templates_file)
        self._container_identifier_separator = ' '

        self._min_num_sentences = min_num_sentences
        self._max_num_sentences = max_num_sentences
        self._sample_from_used_values_prob = used_values_prob

        self._qa_generators = [
            QuestionAnswerGeneratorSelect(debug),
            QuestionAnswerGeneratorIntraEntitySimpleDiff(debug),
            QuestionAnswerGeneratorIntraEntitySubset(debug),
            QuestionAnswerGeneratorInterEntityComparison(debug),
            QuestionAnswerGeneratorInterEntitySuperlative(debug),
            QuestionAnswerGeneratorIntraEntitySuperlative(debug),
            QuestionAnswerGeneratorInterEntitySum(debug)
        ]

        self._nlp = spacy.load("en")

    def _load_vocab(self, vocab_file):
        self._vocab = json.load(open(vocab_file, "r"))

        # match parent objects to parent indices
        container_types = list(self._vocab[ElementType.container.value].keys())
        for container_type in container_types:
            container_type_objs = self._vocab[ElementType.container.value][container_type]
            for container_obj in container_type_objs:
                parent_indices = [x.split("-") for x in container_obj["parent_indices"]]
                container_obj["parents"] = {container_type: [] for container_type in container_types}
                for parent_indice in parent_indices:
                    parent_obj = self._vocab[ElementType.container.value][parent_indice[0]][int(parent_indice[1])-1]
                    container_obj["parents"][parent_indice[0]].append(parent_obj)

        # TODO(mega): validate the vocab structure

    def _load_templates(self, templates_file):
        templates = json.load(open(templates_file, "r"))

        self._templates = templates
        self._templates = {
            "_name": templates["_name"],
            "_subtype_joker": templates["_subtype_joker"],
            "questions": [],
            "sentences": []
        }

        for template_type in ["questions", "sentences"]:
            for template in templates[template_type]:
                # TODO(mega): check how many tokens of each type and subtype and see that there
                #  are enough of each type to populate the template.

                tokens = template["abstracted"].split(" ")
                num_tokens = len(tokens)
                add_template = True
                for token_idx, token in enumerate(tokens):
                    token_info = get_abstracted_token_info(self._vocab, self._templates,
                                                           token, token_idx, num_tokens)
                    if token_info.type in IndependentElementTypes and \
                            (token_info.type not in self._vocab or
                             not self._vocab[token_info.type]):
                        add_template = False
                        break
                    if token_info.subtype and \
                            (token_info.subtype not in self._vocab[token_info.type] or
                             not self._vocab[token_info.type][token_info.subtype]):
                        add_template = False
                        break

                if add_template:
                    self._templates[template_type].append(template)
                    if self._debug:
                        print(f"[-] adding {template_type} template: {template['abstracted']}")
                elif self._debug:
                    print(f"[-] ignoring {template_type} template: {template['abstracted']}")

    def generate_example(self):
        # TODO(mega): weigh templates based on frequency?
        sents, instantiation, states = self._generate_passage()

        if self._debug:
            for sent, state in zip(sents, states):
                print(f"sent: {sent}")
                print(f"state:\n{state}")

        last_state = states[-1]
        qa_pairs = {}
        for qa_generator in self._qa_generators:
            if qa_generator.is_compatible(last_state, instantiation):
                qa_pairs[qa_generator.name] = qa_generator.generate(last_state, instantiation)

        examples = []
        for qa_generator_name in qa_pairs:
            examples.extend([
                SyntheticExample(qa_generator_name, sents, states,
                                 qa_pair.question.replace('  ', ' '), qa_pair.answer, qa_pair.expression)
                for qa_pair in qa_pairs[qa_generator_name]
            ])

        return examples

    def _generate_passage(self):
        num_sentences = np.random.randint(self._min_num_sentences, self._max_num_sentences+1)
        abs_sents = np.random.choice(self._templates["sentences"], num_sentences, replace=True)

        # instantiate abstracted sentences
        sents = []
        instantiation = []
        for abs_sent in abs_sents:
            sents.append(self._instantiate_abstracted_sent(abs_sent, instantiation))

        # generate states
        states = []
        last_state = State()
        for sent_idx, abs_sent in enumerate(abs_sents):
            sent_state = last_state.duplicate()
            sent_instantiation = instantiation[sent_idx]
            for container_update in abs_sent["state_updates"]:
                apply_container_update_to_state(sent_state, container_update, sent_instantiation,
                                                self._container_identifier_separator)

            last_state = sent_state
            states.append(sent_state)

        return sents, instantiation, states

    def _instantiate_abstracted_sent(self, abs_sent, instantiation):
        instant_by_type = instantiation_by_type(instantiation)

        sent = []
        sent_instantiation = {}
        last_indep_token = {}
        used_indep_values = {elem_value: set() for elem_value in ElementType.values()
                             if elem_value if IndependentElementTypes}

        tokens = abs_sent["abstracted"].split(" ")
        num_tokens = len(tokens)
        for token_idx, token in enumerate(tokens[::-1]):
            token_info = get_abstracted_token_info(self._vocab, self._templates, token, token_idx, num_tokens)

            # abstracted token already instantiated
            if token in sent_instantiation:
                update_existing_instantiation(token, token_info.reverse_idx, sent_instantiation, sent)
                if token_info.type in IndependentElementTypes:
                    last_indep_token = sent_instantiation[token]

            # attributes or numbers
            # we assume that attributes and numbers always precede an entity or a container.
            elif token_info.type in DependentElementTypes:
                select_update_dependent_element(token, token_info, last_indep_token, sent_instantiation, sent)

            # containers, entities or verbs
            elif token_info.type in IndependentElementTypes:
                # sample from previously instantiated tokens
                candidates = []
                if np.random.rand() < self._sample_from_used_values_prob:
                    candidates = instant_by_type[token_info.type]
                    if token_info.subtype:
                        candidates = [candidate for candidate in candidates
                                      if candidate["token_subtype"] == token_info.subtype]

                    candidates = [candidate["vocab_obj"] for candidate in candidates
                                  if get_instant_obj_value(candidate) not in used_indep_values[token_info.type]]

                # sample from vocabulary
                if not candidates:
                    if token_info.subtype:
                        candidates = self._vocab[token_info.type][token_info.subtype]
                    else:
                        candidates = self._vocab[token_info.type]

                    candidates = [candidate for candidate in candidates
                                  if candidate["value"] not in used_indep_values[token_info.type]]

                selected = np.random.choice(candidates)
                sent_instantiation[token] = get_token_instant_obj(token_info, selected)
                last_indep_token = sent_instantiation[token]
                used_indep_values[token_info.type].add(selected["value"])
                sent.append(selected["value"])

            # non-abstracted token
            else:
                sent.append(token)

        instantiation.append(sent_instantiation)
        sent = reversed_token_list_to_sentence(sent)

        return sent
