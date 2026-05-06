# slowfast

Lightweight violence-detection pipeline for:

1. whole-frame sliding clip generation
2. fast clip scoring with a 3D video model
3. uncertainty-based routing
4. selective VLM refinement
5. event-level fight export

The first implementation favors a simple, debuggable pipeline over maximum model complexity.

## Baseline

The stabilized CCTV baseline is documented in [docs/baseline.md](/home/deepgu/slowfast/docs/baseline.md:1).

## Layout

- `configs/`: pipeline and training config
- `data/manifests/`: CSV/JSON metadata
- `models/fast/`: fast 3D model dataset, training, inference helpers
- `models/vlm/`: prompt, parser, and VLM interface
- `pipeline/`: orchestration and scoring stages
- `scripts/`: entrypoints for manifest creation, training, inference, evaluation
- `utils/`: shared helpers

## First Run

1. Create or collect metadata in `data/manifests/`
2. Update `configs/base.yaml`
3. Train or point to a fast-model checkpoint
4. Run:

```bash
python scripts/run_inference.py \
  --config configs/base.yaml \
  --video /abs/path/video.mp4 \
  --run-name demo_run
```

## Notes

- The fast stage supports a simple heuristic fallback when no trained checkpoint is available.
- The VLM stage is implemented behind a provider interface with both `mock` and `internvl` providers.
- The current baseline config is set to real local `provider: internvl`; see [docs/baseline.md](/home/deepgu/slowfast/docs/baseline.md:1) for the evaluated routing and fusion policy.
