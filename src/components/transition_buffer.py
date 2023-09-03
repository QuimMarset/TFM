import numpy as np
import torch as th


class TransitionsBatch:


    def __init__(self, batch_size, scheme, groups, data=None, device='cpu'):
        self.batch_size = batch_size
        self.scheme = scheme
        self.groups = groups
        self.device = device
        if data is not None:
            self.data = data
        else:
            self._create_data_arrays()

    
    def _create_data_arrays(self):
        self._update_scheme()
        self.data = {}

        for field_key, field_info in self.scheme.items():
            shape = self._get_field_shape(field_key, field_info)
            dtype = field_info.get("dtype", th.float32)
            self.data[field_key] = th.zeros((self.batch_size, 2, *shape), dtype=dtype, device=self.device)


    def _update_scheme(self):
        assert "filled" not in self.scheme, '"filled" is a reserved key for masking.'
        self.scheme.update({
            "filled": {"vshape": (1,), "dtype": th.long},
        })


    def _get_field_shape(self, field_key, field_info):
        assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
        vshape = field_info["vshape"]
        group = field_info.get("group", None)
            
        if isinstance(vshape, int):
            vshape = (vshape,)

        if group:
            assert group in self.groups, "Group {} must have its number of members defined in _groups_".format(group)
            return (self.groups[group], *vshape)
        else:
            return vshape
        

    def __len__(self):
        return self.batch_size

    
    def __getitem__(self, item):
        if isinstance(item, str):
            return self.data[item]
        else:
            item = self._process_indexation_item(item)
            return self._create_subset(item)


    def _process_indexation_item(self, indexation_item):
        processed_item = []
        if isinstance(indexation_item, (int, slice, list, np.ndarray, th.LongTensor)):
            # Add time dimension
            indexation_item = (indexation_item, slice(None))

        for item in indexation_item:
            if isinstance(item, int):
                # Keep single-element dimensions
                processed_item.append(slice(item, item + 1))
            else:
                processed_item.append(item)
        return processed_item


    def _create_subset(self, indexation_item):
        new_data = {}
        for key, value in self.data.items():
            new_data[key] = value[indexation_item]
        subset_size = self._get_subset_size(indexation_item)
        return TransitionsBatch(subset_size, self.scheme, self.groups, new_data)
    

    def _get_subset_size(self, indexation_item):
        if isinstance(indexation_item, (list, np.ndarray)):
            return len(indexation_item)
        elif isinstance(indexation_item, slice):
            indexation_range = indexation_item.indices(self.batch_size)
            return 1 + (indexation_range[1] - indexation_range[0] - 1) // indexation_range[2]
        

    def to(self, device):
        for key, value in self.data.items():
            self.data[key] = value.to(device)
        self.device = device



class TransitionsReplayBuffer(TransitionsBatch):


    def __init__(self, buffer_size, scheme, groups, device='cpu'):
        super().__init__(buffer_size, scheme, groups, device=device)
        self.buffer_size = buffer_size
        self.buffer_index = 0
        self.filled = False


    def add_transitions(self, transition_data):
        num_transitions = len(transition_data['reward'])

        for i in range(num_transitions):

            for key, values in transition_data.items():

                values_i = values[i]

                if 'next' in key:
                    time_index = 1
                    key = key[5:]
                else:
                    time_index = 0

                if isinstance(values_i, (float, int, list)):
                    values_i = np.array(values_i)

                if isinstance(values_i, np.ndarray):
                    dtype = self.scheme[key].get("dtype", th.float32)
                    values_i = th.tensor(values_i, dtype=dtype, device=self.device)

                self.data[key][self.buffer_index, time_index] = values_i.view_as(self.data[key][0, 0])

            self.data['filled'][self.buffer_index, :] = 1

            if self.buffer_index + 1 >= self.buffer_size:
                self.filled = True
            self.buffer_index = (self.buffer_index + 1) % self.buffer_size


    def can_sample(self, batch_size):
        return self.filled or self.buffer_index >= batch_size


    def sample(self, batch_size):
        indices = np.random.choice(self.buffer_index, batch_size)
        return self[indices]
