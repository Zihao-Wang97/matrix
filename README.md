# HAWP-LAQ

Holistic Attention Weight Pruning with Learned Adaptive Quantization

## Quick Start

```bash
pip install -e .
```

## Run

Local dev:
```bash
python -m hawp_laq.offline.pipeline configs/dev_local.yaml
```

Server:
```bash
python -m hawp_laq.runtime.server configs/run_server.yaml
```
