import torch
import random
import numpy as np
import collections

class DotDict(dict):
    '''
    enable to use dot to search the dict
    dict = {'name':cici}
    dotdict = DotDict(dict)
    dotdict.name
    '''
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        v = DotDict(v)
                    if isinstance(v, list):
                        self.__convert(v)
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    v = DotDict(v)
                elif isinstance(v, list):
                    self.__convert(v)
                self[k] = v

    def __convert(self, v):
        for elem in range(0, len(v)):
            if isinstance(v[elem], dict):
                v[elem] = DotDict(v[elem])
            elif isinstance(v[elem], list):
                self.__convert(v[elem])

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]


def load_pretrain(model, pre_s_dict):
    ''' Load state_dict in pre_model to model
    Solve the problem that model and pre_model have some different keys'''
    s_dict = model.state_dict()
    # remove fc weights and bias
    # use new dict to store states, record missing keys
    missing_keys = []
    new_state_dict = collections.OrderedDict()
    for key in s_dict.keys():
        if key in pre_s_dict.keys():
            new_state_dict[key] = pre_s_dict[key]
        else:
            new_state_dict[key] = s_dict[key]
            missing_keys.append(key)
    print('{} keys are not in the pretrain model:'.format(len(missing_keys)), missing_keys)
    # load new s_dict
    model.load_state_dict(new_state_dict)
    return model


class AvgMeter(object):
    '''
    from TransFuse 
    https://github.com/Rayicer/TransFuse/blob/main/utils/utils.py
    '''
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))

def fix_all_seed(seed):
    print('seed-----------all device', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
