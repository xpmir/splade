import sys
from types import ModuleType
from importlib.machinery import ModuleSpec
from importlib.abc import MetaPathFinder


def noop_decorator(fn=None, *args, **kwargs):
    """No-op decorator that works with or without arguments."""
    if fn is not None:
        return fn
    return lambda f: f


class _FakeClass:
    """Fake class that can be inherited from without metaclass conflicts."""
    _finder = None
    _path = ""
    
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _FakeClass()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        full_path = f"{self._path}.{name}"
        if self._finder and full_path in self._finder.decorators:
            return noop_decorator
        return self._finder._make_class(full_path, name) if self._finder else _FakeClass()
    
    @classmethod
    def __class_getitem__(cls, item):
        """Support for subscript notation like List[int] or TextEncoderBase[str, Tensor]."""
        return cls
    
    def __mro_entries__(self, bases):
        """Tell Python to use object instead of this class when used as a base.
        This avoids metaclass conflicts when inheriting from real classes."""
        return (object,)


class _FakeClassProxy:
    """Wrapper that intercepts attribute access on fake classes."""
    def __init__(self, fake_class):
        object.__setattr__(self, '_fake_class', fake_class)
    
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        fake_class = object.__getattribute__(self, '_fake_class')
        # Try to get from the fake class first
        try:
            return getattr(fake_class, name)
        except AttributeError:
            # Return noop_decorator for missing attributes
            return noop_decorator
    
    def __call__(self, *args, **kwargs):
        return _FakeClassProxy(_FakeClass())
    
    def __mro_entries__(self, bases):
        """When used as a base, return the actual fake class."""
        return (object.__getattribute__(self, '_fake_class'),)


class FakeModule(ModuleType):
    def __init__(self, name, finder):
        super().__init__(name)
        self.__path__ = []
        self._finder = finder
        self._cache = {}

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)

        full_path = f"{self.__name__}.{name}"

        if full_path in self._finder.decorators:
            return noop_decorator

        if name not in self._cache:
            self._cache[name] = self._finder._make_class(full_path, name)

        return self._cache[name]


class FakeModuleFinder(MetaPathFinder):
    def __init__(self, modules, decorators=None):
        self.modules = set(modules)
        self.decorators = set(decorators or [])
        self._class_cache = {}

    def find_spec(self, fullname, path, target=None):
        if any(fullname == m or fullname.startswith(m + '.') for m in self.modules):
            return ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        fullname = spec.name
        if fullname in sys.modules:
            return sys.modules[fullname]
        module = FakeModule(fullname, self)
        sys.modules[fullname] = module
        return module

    def exec_module(self, module):
        pass

    def _make_class(self, path, name):
        if path not in self._class_cache:
            cls = type(name, (_FakeClass,), {
                '_finder': self,
                '_path': path,
            })
            # Wrap the class so attribute access is intercepted
            self._class_cache[path] = _FakeClassProxy(cls)
        return self._class_cache[path]
    
    def __desc__(self):
        return "FakeModuleFinder({self.modules})"


# Usage
sys.meta_path.insert(0, FakeModuleFinder(
    ['torch', 'lightning_fabric', 'torchmetrics', 'pytorch_lightning', 'huggingface_hub', 'transformers'],
    decorators=[
        'torch.compile',
        'torch.jit.unused',
        'torch.jit.script',
        'torch.jit.export',
        'torch.jit.ignore',
        'torch.no_grad',
        'torch.inference_mode',
    ]
))
print("[installed fake modules for torch and co]", file=sys.stderr)