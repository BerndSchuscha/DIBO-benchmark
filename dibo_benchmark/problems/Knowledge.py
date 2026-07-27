from enum import Enum
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List

from pibob.problems.ExclusiveBank import ExclusiveBank


from pibob.models.GPanalticalmean import GPAnalyticalMean

from pibob.models.pyro_model import PyroAnalyticalMean


class KnowledgeKind(str, Enum):
    KERNEL = "kernel"
    DESCRIPTOR = "descriptor"
    CUSTOM_MEAN = "custom_mean"
    PYRO = "pyro"
    DELTA = "delta"


def kind_to_bank(kind: KnowledgeKind) -> str:
    kind = KnowledgeKind(kind)
    if kind == KnowledgeKind.KERNEL:
        return "kernel"
    if kind == KnowledgeKind.DESCRIPTOR:
        return "descriptor"
    # all remaining kinds live in eq bank
    return "eq"


class Knowledge(nn.Module):
    """
    3 banks (descriptor/kernel/eq) but 5 kinds.
    Only one kind is active at a time.
    """

    def __init__(
        self,
        descriptor_bank: Optional[ExclusiveBank] = None,
        kernel_bank: Optional[ExclusiveBank] = None,
        eq_bank: Optional[ExclusiveBank] = None,
        active_kind: Optional[KnowledgeKind] = None,
    ):
        super().__init__()
        self.descriptor_bank = descriptor_bank or ExclusiveBank([])
        self.kernel_bank = kernel_bank or ExclusiveBank([])
        self.eq_bank = eq_bank or ExclusiveBank([])
        self.active_kind = (
            KnowledgeKind(active_kind) if active_kind is not None else None
        )

    def _bank(self, kind: KnowledgeKind) -> ExclusiveBank:
        b = kind_to_bank(kind)
        if b == "descriptor":
            return self.descriptor_bank
        if b == "kernel":
            return self.kernel_bank
        if b == "eq":
            return self.eq_bank
        raise RuntimeError(b)

    def set_active_kind(self, kind: KnowledgeKind):
        kind = KnowledgeKind(kind)
        bank = self._bank(kind)
        if bank.is_empty():
            raise RuntimeError(f"Cannot activate kind={kind}: its bank is empty.")
        self.active_kind = kind

    def get_fns(
        self,
        active_index: Optional[int] = None,
    ) -> Tuple[Optional[KnowledgeKind], Optional[Union[nn.Module, List[nn.Module]]]]:
        """
        Returns (kind, knowledge_fn_or_list).

        - If bank option is a single module: returns a single callable module.
        - If bank option is a list/tuple of modules (one per objective): returns a list
          of callable modules (one per objective), wrapped according to active_kind.

        active_index overrides bank selection without mutating state.
        """
        if self.active_kind is None:
            return None, None

        bank = self._bank(self.active_kind)
        obj = bank.get_module(
            active_index
        )  # may be None, module, or list/tuple[module]
        if obj is None:
            return None, None

        kind = self.active_kind  # enum
        kind_str = kind.value  # "pyro", "custom_mean", ...

        def wrap_core(core: nn.Module) -> nn.Module:
            # You decide which wrapper corresponds to which kind.
            # Typical mapping:
            # - CUSTOM_MEAN -> GPAnalyticalMean(core)
            # - PYRO        -> PyroAnalyticalMean(core)
            # - DELTA       -> (your Delta wrapper, if you have one)
            if kind == KnowledgeKind.CUSTOM_MEAN:
                return GPAnalyticalMean(core)
            if kind == KnowledgeKind.PYRO:
                return core
            if kind == KnowledgeKind.DELTA:
                # TODO: replace with your delta wrapper
                # e.g., return DeltaLearningWrapper(core)
                return core  # placeholder: no wrapping
            # kernel/descriptor: return as-is
            return core

        # --- handle list-of-cores (multi-objective) ---
        if isinstance(obj, (list, tuple)):
            wrapped = [wrap_core(core) for core in obj]
            return kind, wrapped

        # --- single core/module ---
        wrapped = wrap_core(obj)
        return kind, wrapped

    @property
    def active_name(self) -> Optional[str]:
        if self.active_kind is None:
            return None
        m = self._bank(self.active_kind).get_module()
        return None if m is None else m.__class__.__name__

    @property
    def active_index(self) -> Optional[int]:
        if self.active_kind is None:
            return None
        bank = self._bank(self.active_kind)
        return None if bank.is_empty() else bank.active


