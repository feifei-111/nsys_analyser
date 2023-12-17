import functools


LINE_WIDTH = 100

class DefaultDict:
    def __init__(self, default=None):
        self.default = default
        self.dict = dict()

    def __getitem__(self, key):
        if key not in self.dict:
            self.dict[key] = self.default
        return self.dict[key]
    
    def __setitem__(self, key, value):
        self.dict[key] = value

    def sort_on_values(self, key=None):
        if key is None:
            key = lambda x: -x[1]
        return sorted(self.dict.items(), key=key)



def line_printer(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        name_len = (len(func.__name__) + 8) // 2 * 2
        format_str = "{left}{func_name:^" + str(name_len) + "s}{right}"
        print(format_str.format(left="=" * ((LINE_WIDTH - name_len) // 2), func_name=func.__name__, right="=" * ((LINE_WIDTH - name_len) // 2)))
        ret = func(*args, **kwargs)
        print("=" * LINE_WIDTH + "\n")
        return ret

    return inner