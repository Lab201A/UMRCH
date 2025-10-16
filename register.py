class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}
    
    def register(self, model):
        def add_model(key, value):
            if not callable(value):
                raise Exception(f"register object must be callable")
            if key in self._dict:
                print(f"warning: {value.__name__} has been registered before")
            self[key] = value
            return value

        if callable(model):
            return add_model(model.__name__, model)
        else:
            return lambda name : add_model(model, name)
    
    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]
    
    def __contains__(self, key):
        return key in self._dict
    
    def __str__(self):
        return str(self._dict)
    
    def keys(self):
        return self._dict.keys()
    
    def values(self):
        return self._dict.values()
    
    def items(self):
        return self._dict.items()

    def __call__(self, target):
        return self.register(target)