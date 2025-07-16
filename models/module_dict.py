from collections import OrderedDict, abc as container_abcs
from torch.nn import ModuleDict, Module
from torch._jit_internal import _copy_to_script_wrapper

import base64

from typing import Callable, Dict, Iterable, Iterator, Mapping, Optional, Tuple

__all__ = ["RobustModuleDict"]


def b64_keyerror(func: Callable) -> Callable:
    def error_handler(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyError as e:
            raise KeyError(RobustModuleDict.b64_dec(str(e)))

    return error_handler


class RobustModuleDict(ModuleDict):
    ## Subclass to allow
    def __init__(self, modules: Optional[Mapping[str, Module]] = None) -> None:
        super().__init__()
        if modules is not None:
            self.update(modules)

    @staticmethod
    def b64_enc(x: str) -> str:
        encoding = base64.b64encode(x.encode("ascii"))
        return encoding.decode("ascii")

    @staticmethod
    def b64_dec(x: str) -> str:
        encoding = base64.b64decode(x.encode("ascii"))
        return encoding.decode("ascii")

    @property
    def modules_b64(self) -> Dict[str, Module]:
        return {self.b64_dec(k): v for k, v in self._modules.items()}

    @_copy_to_script_wrapper
    @b64_keyerror
    def __getitem__(self, key: str) -> Module:
        return self._modules[self.b64_enc(key)]

    def __setitem__(self, key: str, module: Module) -> None:
        self.add_module(self.b64_enc(key), module)

    @b64_keyerror
    def __delitem__(self, key: str) -> None:
        del self._modules[self.b64_enc(key)]

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[str]:
        return iter(self.modules_b64)

    @_copy_to_script_wrapper
    def __contains__(self, key: str) -> bool:
        return self.b64_enc(key) in self._modules

    def clear(self) -> None:
        """Remove all items from the ModuleDict."""
        self._modules.clear()

    @b64_keyerror
    def pop(self, key: str) -> Module:
        r"""Remove key from the ModuleDict and return its module.

        Args:
            key (str): key to pop from the ModuleDict
        """
        v = self[key]
        del self[key]
        return v

    @_copy_to_script_wrapper
    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ModuleDict keys."""
        return self.modules_b64.keys()

    @_copy_to_script_wrapper
    def items(self) -> Iterable[Tuple[str, Module]]:
        r"""Return an iterable of the ModuleDict key/value pairs."""
        return self.modules_b64.items()

    @_copy_to_script_wrapper
    def values(self) -> Iterable[Module]:
        r"""Return an iterable of the ModuleDict values."""
        return self.modules_b64.values()

    def update(self, modules: Mapping[str, Module]) -> None:
        r"""Update the :class:`~torch.nn.ModuleDict` with key-value pairs from a mapping, overwriting existing keys.

        .. note::
            If :attr:`modules` is an ``OrderedDict``, a :class:`~torch.nn.ModuleDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Args:
            modules (iterable): a mapping (dictionary) from string to :class:`~torch.nn.Module`,
                or an iterable of key-value pairs of type (string, :class:`~torch.nn.Module`)
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError(
                "ModuleDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(modules).__name__
            )

        if isinstance(modules, (OrderedDict, ModuleDict, container_abcs.Mapping)):
            for key, module in modules.items():
                self[key] = module
        else:
            # modules here can be a list with two items
            for j, m in enumerate(modules):
                if not isinstance(m, container_abcs.Iterable):
                    raise TypeError(
                        "ModuleDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(m).__name__
                    )
                if not len(m) == 2:
                    raise ValueError(
                        "ModuleDict update sequence element "
                        "#" + str(j) + " has length " + str(len(m)) + "; 2 is required"
                    )
                # modules can be Mapping (what it's typed at), or a list: [(name1, module1), (name2, module2)]
                # that's too cumbersome to type correctly with overloads, so we add an ignore here
                self[m[0]] = m[1]  # type: ignore[assignment]

    # remove forward alltogether to fallback on Module's _forward_unimplemented
