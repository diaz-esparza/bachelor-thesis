from copy import deepcopy
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

import itertools
import torch

from typing import Any, Callable, Generic, Hashable, Sequence, TypeVar

from ..module_dict import RobustModuleDict

__all__ = [
    "GRL",
    "Lambda",
    "GRL_MODULE",
    "DAClassifier",
    "DAWrapper",
]

_T = TypeVar("_T", bound=Hashable)


class GRL(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, *grad_outputs):
        assert len(grad_outputs) == 1
        return grad_outputs[0].neg()


class Lambda(nn.Module):
    def __init__(self, f: Callable) -> None:
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


GRL_MODULE = Lambda(GRL.apply)


class DAClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_domains: int,
        layers: Sequence[int] = (1024, 1024),
        act: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.model = nn.Sequential(
            nn.Linear(in_features, layers[0]),
            act(),
            *tuple(
                nn.Sequential(nn.Linear(layers[i - 1], x), act())
                for i, x in enumerate(layers[1:])
            ),
            nn.Linear(layers[-1], n_domains),
            # nn.Softmax(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.handle_pooling(x, self.in_features)
        return self.model(x)

    @staticmethod
    def handle_pooling(x: torch.Tensor, n_features: int) -> torch.Tensor:
        r"""
        ASSUMPTIONS:
            1. The first dimension of the Tensor is always the batch size.
            2. There is one dimention which refers to the number of features.
            3. There is only one dimension where the size equals the number of features.
            4. The tensor has a maximum of 4. It can work for 5 but it WOULD NOT happen.
        """

        dims = tuple(x.shape)
        assert len(dims) in range(2, 5)

        idx_features: int | None = None
        for idx, size in enumerate(dims[1:], 1):
            if size == n_features:
                assert idx_features is None
                idx_features = idx
        assert idx_features is not None

        if idx_features != 1:
            x = x.movedim(idx_features, 1)

        if len(dims) == 2:
            return x
        elif len(dims) == 3:
            return F.avg_pool1d(x, x.shape[2:]).squeeze(-1)
        elif len(dims) == 4:
            return F.avg_pool2d(x, x.shape[2:]).squeeze((-2, -1))
        else:
            assert False


class DAWrapper(nn.Module):
    def __init__(
        self,
        core: nn.Module,
        n_domains: int,
        layers: Sequence[int] = (1024, 1024),
        act: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.core = core
        self.grl = GRL_MODULE

        if n_domains < 2:
            raise ValueError(f"{n_domains=} (expected >=2).")
        elif n_domains == 2:
            # SIGMOID
            n_domains = 1

        self.da_classifier = DAClassifier(
            ## Timm attribute
            in_features=self.core.num_features,
            n_domains=n_domains,
            layers=layers,
            act=act,
        )
        self.__da_value = None

    ## Attribute setting/getting rate must be 1:1
    @property
    def da_value(self) -> torch.Tensor:
        output = self.__da_value
        self.__da_value = None
        assert output is not None
        return output

    @da_value.setter
    def da_value(self, x: torch.Tensor):
        ## Relaxing constraint because of inference
        # assert self.__da_value is None
        self.__da_value = x

    def chain(self, grl_features: torch.Tensor) -> None:
        self.da_value = self.da_classifier(grl_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ## Common timm method
        ## Extract features
        x = self.core.forward_features(x)
        self.chain(self.grl(x))
        ## Standard timm head
        return self.core.forward_head(x)


class MultiDAWrapper(nn.Module):
    def __init__(
        self,
        core: nn.Module,
        n_domains_per_da: dict[str, int],
        args_per_da: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()

        args_per_da = args_per_da or {}
        if not set(args_per_da.keys()).issubset(n_domains_per_da.keys()):
            raise ValueError(
                f"The keys of the optional DA args '{args_per_da.keys()}' "
                "do not match mandatory the keys of the mandatory DA args "
                f"'{n_domains_per_da.keys()}'."
            )

        self.core = core
        self.grl = GRL_MODULE
        ## Strat: store wrappers sequentially and keep an id
        self.__da_subwrappers = RobustModuleDict()
        for k in n_domains_per_da.keys():
            self.__da_subwrappers[k] = DAWrapper(
                self.core,
                n_domains=n_domains_per_da[k],
                **(args_per_da.get(k, {})),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ## Feed it to each one, sequentially.
        ## We also offer support for empty MultiDAWrapper(s)

        ## Common timm method
        ## Extract features
        x = self.core.forward_features(x)

        x_grl = self.grl(x)
        for wrapper in self.__da_subwrappers.values():
            wrapper.chain(x_grl)

        ## Standard timm head
        return self.core.forward_head(x)

    @property
    def da_values(self) -> dict[str, torch.Tensor]:
        return {k: v.da_value for k, v in self.__da_subwrappers.items()}
