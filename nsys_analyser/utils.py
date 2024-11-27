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

    def __len__(self):
        return len(self.dict)

    def has(self, key):
        return key in self.dict

    def keys(self):
        return self.dict.keys()

    def values(self):
        return self.dict.values()

    def items(self):
        return self.dict.items()


class Singleton:
    def __init__(self, cls):
        self._cls = cls
        self._instance = None

    def __call__(self):
        if self._instance is None:
            self._instance = self._cls()
        return self._instance


def sort_on_values(dict_, key=None):
    if key is None:
        key = lambda x: -x[1]
    return sorted(dict_.items(), key=key)


def target_events_checker(target_events=None):
    if target_events is None:
        return lambda node: True
    elif isinstance(target_events, str):
        return lambda node: node.text == target_events
    elif isinstance(target_events, (list, tuple)):
        return lambda node: node.text in target_events
    else:
        raise RuntimeError(f"Not Support {type(target_events)} to mark a event")
