import functools
from contextlib import contextmanager


LINE_WIDTH = 100

class DefaultDict:
    def __init__(self, default=None):
        self.default = default
        self.dict = dict()

    def get_default(self):
        if callable(self.default):
            return self.default()
        else:
            return self.default

    def __getitem__(self, key):
        if key not in self.dict:
            self.dict[key] = self.get_default()
        return self.dict[key]
    
    def __setitem__(self, key, value):
        self.dict[key] = value
    
    def has(self, key):
        return key in self.dict

    def keys(self):
        return self.dict.keys()
    
    def values(self):
        return self.dict.values()

    def items(self):
        return self.dict.items()


@contextmanager
def line_printer(title, mark="="):
    name_len = (len(title) + 8) // 2 * 2
    format_str = "{left}{title:^" + str(name_len) + "s}{right}"
    print(format_str.format(left=mark * ((LINE_WIDTH - name_len) // 2), title=title, right=mark * ((LINE_WIDTH - name_len) // 2)))
    yield
    print(mark * LINE_WIDTH + "\n")


def sort_on_values(dict_, key=None):
    if key is None:
        key = lambda x: -x[1]
    return sorted(dict_.items(), key=key)

