
import spacy

from collections import namedtuple, Counter

from generators.element_type import ContainerType
from generators.generator_utils import *
from generators.state import concat_container_entity_attributes, comparable_container_states


QuestionAnswerPair = namedtuple("QuestionAnswerPair",
                                ["question", "answer", "expression"])


class QuestionAnswerGeneratorBase(object):
    def __init__(self, debug=False):
        self.name = "base"
        self._debug = debug

        self._nlp = spacy.load("en")

        self._checked = False

    def is_compatible(self, state, instantiation):
        self._checked = True
        return self._is_compatible(state, instantiation)

    def generate(self, state, instantiation):
        assert self._checked
        self._checked = False
        return self._generate(state, instantiation)

    def _is_compatible(self, state, instantiation):
        raise NotImplementedError

    def _generate(self, state, instantiation):
        raise NotImplementedError


class QuestionAnswerGeneratorSelect(QuestionAnswerGeneratorBase):
    def __init__(self, *args, **kwargs):
        super(QuestionAnswerGeneratorSelect, self).__init__(*args, **kwargs)
        self.name = "select"

        self._verb_pos_instants = None

    def _is_compatible(self, state, instantiation):
        instant_by_type = instantiation_by_type(instantiation)
        self._verb_pos_instants = [
            instant for instant in instant_by_type[ElementType.verb.value]
            if instant["token_subtype"] == VerbType.positive.value
        ]

        return state.get_num_containers() > 0 and len(self._verb_pos_instants) > 0

    def _generate(self, state, instantiation):
        qa_pairs = []
        for container_identifier in state.get_container_identifiers():
            for container_state in state.get_container(container_identifier):
                if container_state.number > 0:
                    verb_pos_instant = np.random.choice(self._verb_pos_instants)
                    verb_pos = verb_pos_instant["vocab_obj"]["value"]
                    entity_attrs = concat_container_entity_attributes(container_state.entity_attributes)

                    # ContainerType.environment.value
                    if container_state.container_type == ContainerType.environment.value:
                        question = \
                            f"How many {entity_attrs} {container_state.entity} were " \
                            f"in {container_identifier} ?"

                    # ContainerType.agent.value
                    else:
                        question = \
                            f"How many {entity_attrs} {container_state.entity} did " \
                            f"{container_identifier} {verb_pos} ?"

                    answer = container_state.number
                    expression = container_state.records

                    qa_pairs.append(QuestionAnswerPair(
                        question=question, answer=answer, expression=expression
                    ))

        return qa_pairs


class QuestionAnswerGeneratorIntraEntitySimpleDiff(QuestionAnswerGeneratorBase):
    def __init__(self, *args, **kwargs):
        super(QuestionAnswerGeneratorIntraEntitySimpleDiff, self).__init__(*args, **kwargs)
        self.name = "intra_entity_simple_diff"

        self._container_identifiers = None

    def _is_compatible(self, state, instantiation):
        self._container_identifiers = {}
        for container_identifier in state.get_container_identifiers():
            pos_container_state_objs = [
                container_state_obj for container_state_obj in state.get_container(container_identifier)
                if container_state_obj.number > 0
            ]
            if len(pos_container_state_objs) >= 2:
                self._container_identifiers[container_identifier] = pos_container_state_objs

        return len(self._container_identifiers) > 0

    def _generate(self, state, instantiation):
        qa_pairs = []
        for container_identifier in self._container_identifiers:
            state_objs = self._container_identifiers[container_identifier]
            num_state_objs = len(state_objs)
            for i in range(num_state_objs):
                for j in range(i+1, num_state_objs):
                    if state_objs[i].number > state_objs[j].number:
                        bigger_obj = state_objs[i]
                        smaller_obj = state_objs[j]
                    elif state_objs[j].number > state_objs[i].number:
                        bigger_obj = state_objs[j]
                        smaller_obj = state_objs[i]
                    else:
                        continue

                    bigger_obj_entity_attrs = concat_container_entity_attributes(bigger_obj.entity_attributes)
                    smaller_obj_entity_attrs = concat_container_entity_attributes(smaller_obj.entity_attributes)

                    # ContainerType.environment.value
                    if bigger_obj.container_type == ContainerType.environment.value:
                        question = \
                            f"How many more {bigger_obj_entity_attrs} {bigger_obj.entity} were " \
                            f"in {container_identifier} than {smaller_obj_entity_attrs} {smaller_obj.entity} ?"

                    # ContainerType.agent.value
                    else:
                        question = \
                            f"How many more {bigger_obj_entity_attrs} {bigger_obj.entity} did " \
                            f"{container_identifier} have than {smaller_obj_entity_attrs} {smaller_obj.entity} ?"

                    answer = bigger_obj.number - smaller_obj.number
                    expression = [1 * bigger_obj.number, -1 * smaller_obj.number]

                    qa_pairs.append(QuestionAnswerPair(
                        question=question, answer=answer, expression=expression
                    ))

        return qa_pairs


class QuestionAnswerGeneratorIntraEntitySubset(QuestionAnswerGeneratorBase):
    def __init__(self, *args, **kwargs):
        super(QuestionAnswerGeneratorIntraEntitySubset, self).__init__(*args, **kwargs)
        self.name = "intra_entity_subset"

        self._container_identifiers = None

    def _is_compatible(self, state, instantiation):
        self._container_identifiers = {}
        for container_identifier in state.get_container_identifiers():
            container_state = state.get_container(container_identifier)
            entity_counts = Counter()
            entity_counts.update([
                container_state_obj.entity
                for container_state_obj in container_state
            ])

            for entity, count in entity_counts.items():
                entity_state_objs = [
                    state_obj for state_obj in container_state
                    if state_obj.entity == entity and state_obj.number > 0
                ]
                if self._valid_entity_set(entity_state_objs):
                    if container_identifier in self._container_identifiers:
                        self._container_identifiers[container_identifier].append(entity_state_objs)
                    else:
                        self._container_identifiers[container_identifier] = [entity_state_objs]

        return len(self._container_identifiers) > 0

    def _generate(self, state, instantiation):
        qa_pairs = []
        for container_identifier in self._container_identifiers:
            for entity_state_objs in self._container_identifiers[container_identifier]:

                entity_sum = sum([entity_state_obj.number for entity_state_obj in entity_state_objs])

                for entity_state_obj in entity_state_objs:
                    entity_state_obj_attrs = concat_container_entity_attributes(entity_state_obj.entity_attributes)
                    question = \
                        f"How many {entity_state_obj.entity} of {container_identifier} were " \
                        f"{entity_state_obj_attrs} {entity_state_obj.entity} ?"

                    qa_pairs.append(QuestionAnswerPair(
                        question=question,
                        answer=entity_state_obj.number,
                        expression=[entity_state_obj.number]
                    ))

                    question = \
                        f"How many {entity_state_obj.entity} of {container_identifier} were not " \
                        f"{entity_state_obj_attrs} {entity_state_obj.entity} ?"
                    answer = entity_sum - entity_state_obj.number
                    qa_pairs.append(QuestionAnswerPair(
                        question=question,
                        answer=answer,
                        expression=[1 * entity_sum, -1 * entity_state_obj.number]
                    ))

        return qa_pairs

    @staticmethod
    def _valid_entity_set(entity_state_objs):
        num_state_objs = len(entity_state_objs)
        if num_state_objs < 2:
            return False

        for i in range(num_state_objs):
            for j in range(i+1, num_state_objs):
                attr_set_i = set(entity_state_objs[i].entity_attributes)
                attr_set_j = set(entity_state_objs[j].entity_attributes)
                if attr_set_i.issubset(attr_set_j) or attr_set_j.issubset(attr_set_i):
                    return False

        return True


class QuestionAnswerGeneratorInterEntityComparison(QuestionAnswerGeneratorBase):
    def __init__(self, *args, **kwargs):
        super(QuestionAnswerGeneratorInterEntityComparison, self).__init__(*args, **kwargs)
        self.name = "inter_entity_comparison"

        self._container_identifiers = None

    def _is_compatible(self, state, instantiation):
        self._container_identifiers = {}
        state_container_identifiers = list(state.get_container_identifiers())
        num_container_identifiers = len(state_container_identifiers)

        for i in range(num_container_identifiers):
            for j in range(i+1, num_container_identifiers):
                container_i = state.get_container(state_container_identifiers[i])
                container_j = state.get_container(state_container_identifiers[j])
                for container_i_state_obj in container_i:
                    for container_j_state_obj in container_j:
                        if comparable_container_states(container_i_state_obj, container_j_state_obj) and \
                                container_i_state_obj.number > 0 and container_j_state_obj.number > 0:
                            key = (state_container_identifiers[i], state_container_identifiers[j])
                            value = (container_i_state_obj, container_j_state_obj)
                            if key in self._container_identifiers:
                                self._container_identifiers[key].append(value)
                            else:
                                self._container_identifiers[key] = [value]

        return len(self._container_identifiers) > 0

    def _generate(self, state, instantiation):
        qa_pairs = []
        for container_identifiers in self._container_identifiers:
            first_container_identifier, second_container_identifier = container_identifiers
            state_obj_pairs = self._container_identifiers[container_identifiers]
            for first_state_obj, second_state_obj in state_obj_pairs:
                if first_state_obj.number > second_state_obj.number:
                    bigger_obj = first_state_obj
                    smaller_obj = second_state_obj
                    bigger_identifier = first_container_identifier
                    smaller_identifier = second_container_identifier
                elif second_state_obj.number > first_state_obj.number:
                    bigger_obj = second_state_obj
                    smaller_obj = first_state_obj
                    bigger_identifier = second_container_identifier
                    smaller_identifier = first_container_identifier
                else:
                    continue

                entity = bigger_obj.entity
                entity_attrs = concat_container_entity_attributes(bigger_obj.entity_attributes)
                identifiers = [first_container_identifier, second_container_identifier]
                for comp_word, ans_obj in zip(["more", "less"],
                                              [(bigger_identifier, bigger_obj), (smaller_identifier, smaller_obj)]):
                    np.random.shuffle(identifiers)

                    # ContainerType.environment.value
                    if bigger_obj.container_type == ContainerType.environment.value:
                        question = \
                            f"Were there {comp_word} {entity_attrs} {entity} in " \
                            f"{identifiers[0]} or in {identifiers[1]} ?"

                    # ContainerType.agent.value
                    else:
                        question = \
                            f"Who had {comp_word} {entity_attrs} {entity}, " \
                            f"{identifiers[0]} or {identifiers[1]} ?"

                    answer = ans_obj[0]
                    expression = [ans_obj[1].number]

                    qa_pairs.append(QuestionAnswerPair(
                        question=question, answer=answer, expression=expression
                    ))

        return qa_pairs


class QuestionAnswerGeneratorInterEntitySuperlative(QuestionAnswerGeneratorBase):
    def __init__(self, *args, **kwargs):
        super(QuestionAnswerGeneratorInterEntitySuperlative, self).__init__(*args, **kwargs)
        self.name = "inter_entity_superlative"

        self.entity_attrs_to_state_objs = None

    def _is_compatible(self, state, instantiation):
        self.entity_attrs_to_state_objs = {}

        # create a mapping of entity to list of agent containers consisting them.
        entity_attrs_to_state_objs = {}
        for container_identifier in state.get_container_identifiers():
            for state_obj in state.get_container(container_identifier):
                # consider only agent containers.
                if state_obj.container_type == ContainerType.agent.value:
                    entity_attrs = concat_container_entity_attributes(sorted(state_obj.entity_attributes))
                    key = (state_obj.entity, entity_attrs)
                    value = (container_identifier, state_obj)
                    if key in entity_attrs_to_state_objs:
                        entity_attrs_to_state_objs[key].append(value)
                    else:
                        entity_attrs_to_state_objs[key] = [value]

        # going over every entity container list, and checking there are no parent-child pairs,
        # and that there's at least one container with a positive number per entity.
        for entity_attrs in entity_attrs_to_state_objs:
            # check that there are at least two containers with the given entity
            state_objs = entity_attrs_to_state_objs[entity_attrs]
            if len(state_objs) <= 1:
                continue

            # check that there's at least one container with a positive number
            container_entity_positive_numbers = [
                state_obj.number
                for container_identifier, state_obj in state_objs
                if state_obj.number > 0
            ]
            if len(container_entity_positive_numbers) == 0:
                continue

            # check that there are no parent-child container pairs
            is_comparable_container_list = True
            num_state_objs = len(state_objs)
            for i in range(num_state_objs):
                for j in range(i+1, num_state_objs):
                    if state.are_containers_related(state_objs[i][0], state_objs[j][0]):
                        is_comparable_container_list = False
                        break
                if not is_comparable_container_list:
                    break

            if not is_comparable_container_list:
                continue

            self.entity_attrs_to_state_objs[entity_attrs] = state_objs

        return len(self.entity_attrs_to_state_objs) > 0

    def _generate(self, state, instantiation):
        qa_pairs = []
        for key in self.entity_attrs_to_state_objs:
            entity, entity_attrs = key
            container_state_objs = self.entity_attrs_to_state_objs[key]
            numbers = np.array([state_obj.number for _, state_obj in container_state_objs])
            max_container_state_obj = container_state_objs[np.argmax(numbers)]
            min_container_state_obj = container_state_objs[np.argmin(numbers)]

            for super_word, (ans_container_identifier, ans_obj) in \
                    zip(["highest", "lowest"],
                        [max_container_state_obj, min_container_state_obj]):
                # generate "lowest" question only in case the minimal number is positive.
                if super_word == "lowest" and ans_obj.number < 0:
                    continue

                question = f"Who had the {super_word} number of {entity_attrs} {entity} in total ?"
                answer = ans_container_identifier
                expression = [ans_obj.number]

                qa_pairs.append(QuestionAnswerPair(
                    question=question, answer=answer, expression=expression
                ))

        return qa_pairs


class QuestionAnswerGeneratorIntraEntitySuperlative(QuestionAnswerGeneratorBase):
    def __init__(self, *args, **kwargs):
        super(QuestionAnswerGeneratorIntraEntitySuperlative, self).__init__(*args, **kwargs)
        self.name = "intra_entity_superlative"

        self.container_state_objs = None
        self._verb_pos_instants = None

    def _is_compatible(self, state, instantiation):
        instant_by_type = instantiation_by_type(instantiation)
        self._verb_pos_instants = [
            instant for instant in instant_by_type[ElementType.verb.value]
            if instant["token_subtype"] == VerbType.positive.value
        ]

        self.container_state_objs = {}

        for container_identifier in state.get_container_identifiers():
            for state_obj in state.get_container(container_identifier):
                if len(state_obj.records) > 1 and max(state_obj.records) > 0:
                    if container_identifier in self.container_state_objs:
                        self.container_state_objs[container_identifier].append(state_obj)
                    else:
                        self.container_state_objs[container_identifier] = [state_obj]

        return len(self.container_state_objs) > 0 and len(self._verb_pos_instants) > 0

    def _generate(self, state, instantiation):
        qa_pairs = []
        for container_identifier in self.container_state_objs:
            state_objs = self.container_state_objs[container_identifier]
            for state_obj in state_objs:
                verb_pos_instant = np.random.choice(self._verb_pos_instants)
                verb_pos = verb_pos_instant["vocab_obj"]["value"]

                for super_word, ans_number in \
                        zip(["highest", "lowest"],
                            [max(state_obj.records), min(state_obj.records)]):
                    # generate "lowest" question only in case the minimal number is positive.
                    if super_word == "lowest" and ans_number < 0:
                        continue

                    entity_attrs = concat_container_entity_attributes(state_obj.entity_attributes)

                    # ContainerType.environment.value
                    if state_obj.container_type == ContainerType.environment.value:
                        question = f"What was the {super_word} number of " \
                            f"{entity_attrs} {state_obj.entity} {verb_pos} in {container_identifier} ?"

                    # ContainerType.agent.value
                    else:
                        question = f"What is the {super_word} number of " \
                            f"{entity_attrs} {state_obj.entity} {container_identifier} {verb_pos} ?"

                    answer = ans_number
                    expression = [ans_number]

                    qa_pairs.append(QuestionAnswerPair(
                        question=question, answer=answer, expression=expression
                    ))

        return qa_pairs


class QuestionAnswerGeneratorInterEntitySum(QuestionAnswerGeneratorBase):
    def __init__(self, *args, **kwargs):
        super(QuestionAnswerGeneratorInterEntitySum, self).__init__(*args, **kwargs)
        self.name = "inter_entity_sum"

        self.entity_attrs_to_state_objs = None
        self.suffix = ["combined", "in total"]

    def _is_compatible(self, state, instantiation):
        self.entity_attrs_to_state_objs = {}

        # create a mapping of entity to list of containers consisting them.
        # consider only container states with a positive number.
        entity_attrs_to_state_objs = {}
        for container_identifier in state.get_container_identifiers():
            for state_obj in state.get_container(container_identifier):
                if state_obj.number > 0:
                    entity_attrs = concat_container_entity_attributes(sorted(state_obj.entity_attributes))
                    key = (state_obj.entity, entity_attrs, state_obj.container_type)
                    value = (container_identifier, state_obj)
                    if key in entity_attrs_to_state_objs:
                        entity_attrs_to_state_objs[key].append(value)
                    else:
                        entity_attrs_to_state_objs[key] = [value]

        # going over every entity container list, and checking there are no parent-child pairs.
        for key in entity_attrs_to_state_objs:
            # check that there are at least two containers with the given entity
            state_objs = entity_attrs_to_state_objs[key]
            if len(state_objs) <= 1:
                continue

            # check that there are no parent-child container pairs
            is_comparable_container_list = True
            num_state_objs = len(state_objs)
            for i in range(num_state_objs):
                for j in range(i+1, num_state_objs):
                    if state.are_containers_related(state_objs[i][0], state_objs[j][0]):
                        is_comparable_container_list = False
                        break
                if not is_comparable_container_list:
                    break

            if not is_comparable_container_list:
                continue

            self.entity_attrs_to_state_objs[key] = state_objs

        return len(self.entity_attrs_to_state_objs) > 0

    def _generate(self, state, instantiation):
        qa_pairs = []
        for key in self.entity_attrs_to_state_objs:
            entity, entity_attrs, container_type = key
            container_state_objs = self.entity_attrs_to_state_objs[key]
            container_identifiers = np.array([
                container_identifier
                for container_identifier, _ in container_state_objs
            ])
            np.random.shuffle(container_identifiers)
            container_identifiers_str = ', '.join(container_identifiers[:-1]) + \
                                        ' and ' + container_identifiers[-1]
            numbers = [
                state_obj.number
                for _, state_obj in container_state_objs
            ]
            np.random.shuffle(self.suffix)

            # ContainerType.environment.value
            if container_type == ContainerType.environment.value:
                question = \
                    f"How many {entity_attrs} {entity} were " \
                    f"in {container_identifiers_str} {self.suffix[0]} ?"

            # ContainerType.agent.value
            else:
                question = \
                    f"How many {entity_attrs} {entity} did " \
                    f"{container_identifiers_str} have {self.suffix[0]} ?"

            answer = sum(numbers)
            expression = [1 * number for number in numbers]

            qa_pairs.append(QuestionAnswerPair(
                question=question, answer=answer, expression=expression
            ))

        return qa_pairs
