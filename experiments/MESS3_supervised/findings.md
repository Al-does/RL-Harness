# MESS3 supervised replication

This experiment targets the core MESS3 simplex result from Shai et al.,
“Transformers Represent Belief State Geometry in their Residual Stream”
([arXiv:2405.15943](https://arxiv.org/abs/2405.15943)).

The paper and archived training configuration disagree about the nominal
training budget. This recipe targets 629,209,600 token-position losses
(983,140 SGD updates), matching the checkpoint used for the published final
simplex rather than the unfinished nominal 10-million-update configuration.

Each completed run writes its measured findings, figures, probe metrics, and
stage timings beneath `results/<run-id>/`. Full checkpoints remain under the
ignored `artifacts/<run-id>/` tree.
