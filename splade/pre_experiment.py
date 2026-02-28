import sys
from experimaestro.experiments import mock_modules

# Usage
mock_modules(
    ['torch', 'lightning', 'lightning_fabric', 'torchmetrics', 'pytorch_lightning', 'huggingface_hub', 'transformers'],
)

print("[installed fake modules for torch and co]", file=sys.stderr)