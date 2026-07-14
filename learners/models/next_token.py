"""RLModule mixin for a training-only next-token classification head."""

from __future__ import annotations

from torch import nn

from ray.rllib.core.columns import Columns

NAMESPACE = "next_token_aux"
FWD_KEY = f"{NAMESPACE}/logits"


class NextTokenAuxHead:
    """Add next-token logits without coupling the encoder to an RL algorithm."""

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
        self.next_token_aux_head = nn.Linear(
            self._embedding_dim, num_classes
        )

    def _forward_train(self, batch, **kwargs):
        outputs = super()._forward_train(batch, **kwargs)
        outputs[FWD_KEY] = self.next_token_aux_head(
            outputs[Columns.EMBEDDINGS]
        )
        return outputs
