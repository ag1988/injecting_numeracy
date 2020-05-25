
from collections import namedtuple

from utils.collections import get_list_subsets


ContainerState = namedtuple("ContainerState",
                            ["container_type", "number", "entity", "entity_attributes", "records"])


class State(object):
    def __init__(self):
        self._container_states = {}
        self._children_to_parents = {}           # can contain container identifiers without a state
        self._parents_to_children = {}           # include only children with a container state
        self._identifier_to_container_type = {}  # can contain container identifiers without a state

    def __str__(self):
        container_states = "\n".join([
            f"  (container) {container_identifier}:\n\t\t" +
            "\n\t\t".join([
                str(container_state)
                for container_state in self._container_states[container_identifier]
            ])
            for container_identifier in self._container_states
        ]) + "\n"

        parents_to_children = "\n".join([
            f"  (parent-children) {parent_identifier}: {self._parents_to_children[parent_identifier]}"
            for parent_identifier in self._parents_to_children
        ]) + "\n"

        return container_states + parents_to_children + "\n"

    def update(self, container_identifier, container_state_update, parent_identifiers_types):
        self._update_container_state(container_identifier, container_state_update, parent_identifiers_types)

    def add(self, container_identifier, container_state_updates, parent_identifiers):
        for container_state_update in container_state_updates:
            self._update_container_state(container_identifier, container_state_update, parent_identifiers)

    def exists(self, container_identifier):
        return container_identifier in self._container_states

    def get_container_identifiers(self):
        return self._container_states.keys()

    def get_num_containers(self):
        return len(self._container_states)

    def get_orphan_container_identifiers(self):
        orphan_container_identifiers = []
        for container_identifier in self._container_states:
            parent_identifiers = self._children_to_parents[container_identifier]
            existing_parent_identifiers = [
                parent_identifier for parent_identifier in parent_identifiers
                if self.exists(parent_identifier)
            ]
            if len(existing_parent_identifiers) == 0:
                orphan_container_identifiers.append(container_identifier)

        return orphan_container_identifiers

    def get_container(self, container_identifier):
        assert container_identifier in self._container_states
        return self._container_states[container_identifier]

    def _update_container_state(self, container_identifier, container_state_update, parent_identifiers_types):
        #
        # update container identifier-to-type information
        #
        for parent_identifier, parent_container_type in parent_identifiers_types:
            if parent_identifier not in self._identifier_to_container_type:
                self._identifier_to_container_type[parent_identifier] = parent_container_type

        #
        # update the container state
        #
        if not self.exists(container_identifier):
            self._container_states[container_identifier] = [container_state_update]
            self._identifier_to_container_type[container_identifier] = container_state_update.container_type

            # store parents list of container_identifier, and
            # mark container_identifier as a child for all of its parents in the state.
            parent_identifiers = [parent_identifier for parent_identifier, _ in parent_identifiers_types]
            self._children_to_parents[container_identifier] = parent_identifiers
            for parent_identifier in parent_identifiers:
                if self.exists(parent_identifier):
                    self._parents_to_children[parent_identifier].append(container_identifier)

            # initialize children list for container_identifier, and
            # add all of its children in the state to it.
            # also, aggregate children updates into container_identifier.
            self._parents_to_children[container_identifier] = []
            for child_identifier in self.get_container_identifiers():
                if container_identifier in self._children_to_parents[child_identifier]:
                    self._parents_to_children[container_identifier].append(child_identifier)
                    # adjust child container updates to match container type and
                    # make a recursive call.
                    container_type = self._identifier_to_container_type[container_identifier]
                    adjusted_container_state_updates = [
                        adjust_container_type(child_state_update, container_type)
                        for child_state_update in self._container_states[child_identifier]
                    ]
                    self.add(container_identifier, adjusted_container_state_updates, [])

        else:
            container_state = self._container_states[container_identifier]

            # If the update's entity already exists in the container, update it.
            container_updated = False
            for obj_index, container_state_obj in enumerate(container_state):
                if container_state_obj.entity == container_state_update.entity and \
                        set(container_state_obj.entity_attributes) == set(container_state_update.entity_attributes):
                    self._container_states[container_identifier].append(ContainerState(
                        container_state_obj.container_type,
                        container_state_obj.number + container_state_update.number,
                        container_state_obj.entity,
                        container_state_obj.entity_attributes,
                        container_state_obj.records + [container_state_update.number]
                    ))
                    del self._container_states[container_identifier][obj_index]
                    container_updated = True
                    break

            # If we get here there is a new entity for the container.
            if not container_updated:
                self._container_states[container_identifier].append(container_state_update)

        #
        # update the state of every container's parent
        #
        for parent_identifier in self._children_to_parents[container_identifier]:
            if self.exists(parent_identifier):
                # adjust container update to match parent container type and
                # make a recursive call.
                parent_container_type = self._identifier_to_container_type[parent_identifier]
                parent_state_update = adjust_container_type(container_state_update, parent_container_type)
                self._update_container_state(parent_identifier, parent_state_update, [])

    def duplicate(self):
        new_state = State()

        # adding container states without parents information
        for container_identifier in self._container_states:
            container_state = self._container_states[container_identifier]
            new_state.add(container_identifier, container_state, [])

        # artificially adding parents information
        for child_identifier in self._children_to_parents:
            parent_identifiers = self._children_to_parents[child_identifier]
            new_state._children_to_parents[child_identifier] = [
                parent_identifier for parent_identifier in parent_identifiers
            ]
        for parent_identifier in self._parents_to_children:
            child_identifiers = self._parents_to_children[parent_identifier]
            new_state._parents_to_children[parent_identifier] = [
                child_identifier for child_identifier in child_identifiers
            ]

        return new_state

    def are_containers_related(self, first_identifier, second_identifier):
        return first_identifier in self._children_to_parents[second_identifier] or \
               second_identifier in self._children_to_parents[first_identifier]


def concat_container_entity_attributes(entity_attributes, separator=' '):
    return separator.join(entity_attributes)


def get_container_identifier(container_value, container_attr_values, separator):
    return separator.join(container_attr_values + [container_value])


def get_container_parent_identifiers(container_instant, separator):
    # for every parent, extract all possible identifiers with up to 2 attributes.
    parent_identifiers_types = []
    for container_type in container_instant["vocab_obj"]["parents"]:
        for parent_obj in container_instant["vocab_obj"]["parents"][container_type]:
            parent_value = parent_obj["value"]
            parent_attr_subsets = get_list_subsets(parent_obj["attributes"])
            for parent_attr_subset in parent_attr_subsets:
                parent_attr_values = list(parent_attr_subset)
                parent_identifier = separator.join(parent_attr_values + [parent_value])
                parent_identifiers_types.append((parent_identifier, container_type))

    return parent_identifiers_types


def comparable_container_states(first_state, second_state, check_cont_type=True):
    if check_cont_type:
        return (first_state.container_type == second_state.container_type and
                first_state.entity == second_state.entity and
                set(first_state.entity_attributes) == set(second_state.entity_attributes))
    else:
        return (first_state.entity == second_state.entity and
                set(first_state.entity_attributes) == set(second_state.entity_attributes))


def adjust_container_type(container_state_update, new_container_type):
    return ContainerState(
        new_container_type,
        container_state_update.number,
        container_state_update.entity,
        container_state_update.entity_attributes,
        container_state_update.records
    )
