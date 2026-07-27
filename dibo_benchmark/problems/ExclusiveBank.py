import torch.nn as nn
from typing import List, Optional, Sequence

class ExclusiveBank(nn.Module):
    """
    Each option is a list of modules, one per objective.
    """
    def __init__(
        self,
        options: Optional[Sequence[Sequence[nn.Module]]] = None,
        active: int = 0,
    ):
        super().__init__()
        options = [] if options is None else list(options)

        # sanity: all options must have same length
        lengths = {len(opt) for opt in options}
        if len(lengths) > 1:
            raise ValueError("All options must have same number of objectives")

        # register everything
        self.options = nn.ModuleList(
            [nn.ModuleList(opt) for opt in options]
        )
        self.active = int(active)

    def is_empty(self) -> bool:
        return len(self.options) == 0

    @property
    def n_objectives(self) -> int:
        return 0 if self.is_empty() else len(self.options[0])

    def set_active(self, idx: int):
        if self.is_empty():
            raise RuntimeError("Bank is empty")
        self.active = int(idx)

    def get_module(
        self,
        active_index: Optional[int] = None,
    ) -> Optional[List[nn.Module]]:
        """
        Returns list of modules (one per objective) or None if empty.
        """
        if self.is_empty():
            return None

        idx = self.active if active_index is None else active_index
        return list(self.options[idx])
