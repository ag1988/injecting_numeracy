
import numpy as np

from generators.element_type import ElementType, VerbType, AbsTokenInfo
from generators.state import ContainerState, \
    get_container_identifier, get_container_parent_identifiers

from utils.py_utils import uppercase_first_char


#################################################################################
# sentence instantiation
#

def get_abstracted_token_info(vocab, templates, token, token_idx, num_tokens):
    token_parts = token.split("-")
    assert 1 <= len(token_parts) <= 3

    token_type = token_parts[0]

    if len(token_parts) > 1:
        token_type_idx = token_parts[1]
    else:
        token_type_idx = ''

    if len(token_parts) > 2:
        token_subtype = token_parts[2]
        if token_subtype == templates["_subtype_joker"]:
            token_subtype = np.random.choice(list(vocab[token_type].keys()))
    else:
        token_subtype = ''

    token_reverse_idx = num_tokens - token_idx - 1

    return AbsTokenInfo(token_type, token_type_idx, token_subtype, token_reverse_idx)


def reversed_token_list_to_sentence(tokens):
    sentence = tokens[::-1]
    sentence[0] = uppercase_first_char(sentence[0])

    return ' '.join([x for x in sentence if x])


def update_existing_instantiation(token, token_idx, sent_instantiation, sent):
    selected = sent_instantiation[token]
    sent_instantiation[token]["token_indices"].append(token_idx)
    sent.append(get_instant_obj_value(selected))


def select_update_dependent_element(token, token_info, last_indep_token, sent_instantiation, sent,
                                    use_instant_attr=False):
    #
    # token_type == ElementType.number.value
    #
    if token_info.type == ElementType.number.value:
        # TODO(mega): handle cases when the number cannot be random, for example
        #  when it's dependent on previously selected numbers.
        selected = np.random.randint(last_indep_token["vocab_obj"]["number_min"],
                                     last_indep_token["vocab_obj"]["number_max"])
        last_indep_token["numbers_abs_tokens"].append(token)
        last_indep_token["numbers_values"].append(selected)

    #
    # token_type == ElementType.attribute.value
    #
    else:
        if use_instant_attr:
            # select attributes from previously instantiated abstract token
            attributes = last_indep_token["instant_obj"]["attributes_values"]
        else:
            # select attributes from vocabulary object
            attributes = last_indep_token["vocab_obj"]["attributes"]

        if attributes:
            selected = np.random.choice(attributes)
        else:
            selected = ''

        last_indep_token["attributes_abs_tokens"].append(token)
        last_indep_token["attributes_values"].append(selected)

    sent_instantiation[token] = get_token_instant_obj(token_info, str(selected))
    sent.append(str(selected))


def get_token_instant_obj(token_info, selected_candidate, is_instant_obj=False):
    if is_instant_obj:
        vocab_obj = selected_candidate["vocab_obj"]
        instant_obj = selected_candidate
    else:
        vocab_obj = selected_candidate
        instant_obj = {}

    return {
        "token_type": token_info.type,
        "token_type_idx": token_info.type_idx,
        "token_subtype": token_info.subtype,
        "token_indices": [token_info.reverse_idx],
        "attributes_abs_tokens": [],
        "attributes_values": [],
        "numbers_abs_tokens": [],
        "numbers_values": [],
        "vocab_obj": vocab_obj,
        "instant_obj": instant_obj
    }


def get_instant_obj_value(instant):
    if type(instant["vocab_obj"]) == dict:
        return instant["vocab_obj"]["value"]
    elif type(instant["vocab_obj"]) == str:
        return instant["vocab_obj"]
    else:
        raise RuntimeError


def get_instant_attribute_values(update, instant, attr_key):
    attr_values = [
        instant["attributes_values"][instant["attributes_abs_tokens"].index(abs_attr)]
        for abs_attr in update[attr_key]
    ]
    if '' in attr_values:
        attr_values.remove('')

    return attr_values


def instantiation_by_type(sents_instantiation):
    dict = {element_type: [] for element_type in ElementType.values()}
    for sent_instantiation in sents_instantiation:
        for token in sent_instantiation:
            token_type = sent_instantiation[token]["token_type"]
            dict[token_type].append(sent_instantiation[token])

    return dict


#################################################################################
# state tracking
#


def apply_container_update_to_state(sent_state, update, sent_instantiation, separator):
    # check update values correspond to instantiation
    container_instant = sent_instantiation[update["container"]]
    assert set(update["container_attributes"]) == set(container_instant["attributes_abs_tokens"])
    entity_instant = sent_instantiation[update["entity"]]
    assert set(update["entity_attributes"]) == set(entity_instant["attributes_abs_tokens"])

    # build container state
    container_value = get_instant_obj_value(container_instant)
    entity_value = get_instant_obj_value(entity_instant)
    container_attr_values = get_instant_attribute_values(update, container_instant,
                                                         "container_attributes")
    entity_attr_values = get_instant_attribute_values(update, entity_instant,
                                                      "entity_attributes")

    entity_number_idx = entity_instant["numbers_abs_tokens"].index(update["number"])
    num_sign = get_number_sign(update, sent_instantiation)
    entity_num_value = entity_instant["numbers_values"][entity_number_idx] * num_sign

    container_identifier = get_container_identifier(container_value, container_attr_values, separator)
    container_state_update = ContainerState(
        container_instant["token_subtype"], entity_num_value,
        entity_value, entity_attr_values, [entity_num_value]
    )
    if not sent_state.exists(container_identifier):
        parent_identifiers_types = get_container_parent_identifiers(container_instant, separator)
    else:
        parent_identifiers_types = []

    sent_state.update(container_identifier, container_state_update, parent_identifiers_types)


def get_number_sign(update, sent_instantiation):
    # TODO(mega): handle cases where verb is fixed and not abstracted with another field.
    if not update["verb"]:
        verb_type = 'OBS'
    else:
        verb_instant = sent_instantiation[update["verb"]]
        verb_type = verb_instant["token_subtype"]
    container_instant = sent_instantiation[update["container"]]
    container_index = container_instant["token_type_idx"]

    # gain for container
    if verb_type in [VerbType.observation.value, VerbType.positive.value, VerbType.construct.value]:
        return 1

    # loss for container
    elif verb_type in [VerbType.negative.value, VerbType.destroy.value]:
        return -1

    # transfer from/to container - we assume there are at most two containers, and
    # that transfer verbs are from CONT-1 to CONT-2.
    elif verb_type == VerbType.positive_transfer.value:
        if container_index == '1':
            return 1
        else:
            return -1
    elif verb_type == VerbType.negative_transfer.value:
        if container_index == '1':
            return -1
        else:
            return 1

    else:
        print(f"_get_number_sign: could not parse update based on verb type and instantiation: "
              f"{verb_type}, {container_index} (returned 1)")
        return 1

