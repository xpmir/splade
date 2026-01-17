import sys

from experimaestro.experiments import FakeModuleFinder

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