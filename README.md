# DRIFT — Diffusion Reverse Inversion Fingerprint Tracking

DRIFT detects deepfakes and attributes generated images to their source generator by mapping images through a DDIM inversion process (x₀ → x_T) and analyzing the resulting noise-space trajectories as discriminative fingerprints.

---

## Installation

```bash
git clone <this-repo>
cd DRIFT
pip install -r requirements.txt
```

The ADM backbone relies on the guided-diffusion code vendored inside `AIGCDetectBenchmark-main/preprocessing_model/`. No extra install step is needed — `ADMBackbone` adds the path at import time.

Download the ADM checkpoint (256×256 unconditional):
```
https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
```
Place it at the path specified by `configs/drift_default.yaml → model.adm_model_path`.

---

## Quick Start

```
Step A  — Prepare data (directory layout: root/0_real/ and root/1_fake/)
Phase 1 — Implement feature extractors F1/F2 (inherit FeatureExtractor)
Phase 2 — Implement feature extractors F3/F4
Phase 3 — Train binary detection head with DRIFTTrainer
Phase 4 — Train attribution head (GMM → linear alignment)
```

### Minimal binary-detection run

```python
from DRIFT.data.dataloader import DRIFTDataLoader
from DRIFT.models.backbone.adm_wrapper import ADMBackbone
from DRIFT.evaluation.evaluator import DRIFTEvaluator

loader = DRIFTDataLoader(
    root="/data/AIGCBenchmark",
    mode="binary_mode",
    split="test",
)
backbone = ADMBackbone(
    model_path="/checkpoints/256x256_diffusion_uncond.pt",
    device="cuda",
    ddim_steps=20,
)
# ... attach feature extractor and head, then:
evaluator = DRIFTEvaluator()
results = evaluator.run_full_evaluation(model, {"test": loader})
print(results)
```

---

## Directory Structure

```
DRIFT/
├── configs/
│   └── drift_default.yaml       Default hyper-parameters and paths
├── data/
│   ├── dataloader.py            DRIFTDataLoader — binary & attribution modes
│   └── transforms.py            Train/test augmentation pipelines
├── models/
│   ├── backbone/
│   │   └── adm_wrapper.py       ADMBackbone — wraps DDIM inversion
│   ├── features/
│   │   └── base.py              FeatureExtractor abstract base class
│   └── heads/
│       ├── binary.py            BinaryDetectionHead
│       └── attribution.py       GeneratorAttributionHead (GMM + linear)
├── training/
│   ├── trainer.py               DRIFTTrainer
│   └── losses.py                Loss functions
├── evaluation/
│   ├── metrics.py               AUC, AP, attribution accuracy, timing
│   └── evaluator.py             DRIFTEvaluator — paper-style tables
└── utils/
    ├── logger.py                 Structured logging
    ├── visualization.py          t-SNE, PSD, Wasserstein heatmaps
    └── checkpointing.py          Save / load checkpoints
```

---

## Phase Roadmap

| Phase | Task | Key files |
|-------|------|-----------|
| Infra | This scaffold | all of the above |
| 1 | Implement F1 (x_T statistics) and F2 (trajectory statistics) | `models/features/` |
| 2 | Implement F3 (PSD-based) and F4 (Wasserstein) | `models/features/` |
| 3 | Binary detection training & cross-generator eval | `training/`, `evaluation/` |
| 4 | GMM attribution + linear alignment | `models/heads/attribution.py` |

---

## Citation

If you use this code please cite the works that made it possible:

**DIRE** (ADM backbone + DDIM inversion baseline):
```bibtex
@inproceedings{wang2023dire,
  title     = {DIRE for Diffusion-Generated Image Detection},
  author    = {Wang, Zhendong and Bao, Jianmin and Zhou, Wengang and Wang, Weilun
               and Hu, Hezhen and Chen, Hong and Li, Houqiang},
  booktitle = {ICCV},
  year      = {2023},
}
```

**SDAIE** (GMM attribution code):
```bibtex
@article{sdaie2024,
  title  = {SDAIE: Single-Domain Attribution via Inversion Embeddings},
  author = {-- see SDAIE-main/ for full citation --},
  year   = {2024},
}
```

**DFFreq** (frequency-domain detection reference):
```bibtex
@article{dffreq2023,
  title  = {DFFreq: DeepFake Detection in the Frequency Domain},
  author = {-- see DFFreq-main/ for full citation --},
  year   = {2023},
}
```
