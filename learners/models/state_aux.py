"""RLModule mixin for a training-only state-classification head."""

from __future__ import annotations

from torch import nn

from ray.rllib.core.columns import Columns

NAMESPACE = "state_aux"
FWD_KEY = f"{NAMESPACE}/logits"


class StateAuxHead:
    """Add state logits for supervised/probe experiments."""

    def setup(self):
        super().setup()
        config = self.model_config.get(NAMESPACE, {})
        if "num_classes" not in config:
            raise ValueError(
                f"model_config[{NAMESPACE!r}] must define 'num_classes'"
            )
        num_classes = int(config["num_classes"])
        if num_classes <= 0:
            raise ValueError(f"{NAMESPACE}/num_classes must be positive")
        self.state_aux_head = nn.Linear(self._embedding_dim, num_classes)

    def _forward_train(self, batch, **kwargs):
        outputs = super()._forward_train(batch, **kwargs)
        outputs[FWD_KEY] = self.state_aux_head(outputs[Columns.EMBEDDINGS])
        return outputs
