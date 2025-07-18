from abc import ABC, abstractmethod


class ObjectCreator(ABC):

    @abstractmethod
    def __call__(self):
        """"""

class object_factoryBase(ABC):
    def __init__(self):
        self._creators = {}

    def register_creator(self, key, creator:ObjectCreator):
        self._creators[key] = creator

    @abstractmethod
    def create(self, key, **kwargs):
        """"""

    def keys(self):
        return self._creators.keys()


class SpecificObjectFactory(object_factoryBase):

    def create(self, key, *args, **kwargs)->ObjectCreator:
        creator = self._creators.get(key)
        if not creator:
            message = "Key: {} \n Not Found. Available Keys: {}".format(key, self._creators.keys())
            raise KeyError(message)
        return creator(*args, **kwargs)


class DefaultObjectFactory(object_factoryBase):
    
    def __init__(self, default_creator:ObjectCreator):
        super().__init__()
        self._default_creator = default_creator

    def create(self, key, *args, **kwargs):
        if key not in self._creators.keys():
            creator = self._default_creator
        else:
            creator = self._creators.get(key)
        return creator(*args, **kwargs)


class IdentifierBase(ABC):

    def __init__(self, default=None):
        self._registry = {}
        self._default = None
        self.set_default(default)

    def set_default(self, default):
        self._default = default

    @abstractmethod
    def identify(self):
        """"""

    def register(self, key, value):
        self._registry[key] = value
    
    @property
    def keys(self):
        return self._registry.keys()

    def _check_key(self, key):
        if key not in self.keys and self._default is None:
            message = (f"\nCould not identify key \"{key}\" and no default provided. "+
                       f"available keys are:\n")
            for key in self.keys:
                message += f"    {key}\n"
            raise KeyError(message)


class BasicIdentifier(IdentifierBase):

    def identify(self, key=None):
        self._check_key(key)
        if self._default is not None and key not in self.keys:
            if callable(self._default):
                return self._default()
            else:
                return self._default
        return self._registry[key]


class IdentifierByTestFunction(IdentifierBase):

    def identify(self):
        value = None
        for current_test_key, current_value in self._registry.items():
            if current_test_key():
                value = current_value
                break
        if value is None:
            value = self._default
        return value
 
