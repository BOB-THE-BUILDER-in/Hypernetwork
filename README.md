## Pretrained models and data

Trained checkpoints and processed data are hosted separately due to size.

- **Hugging Face dataset**: [bobthebuilderinternational/hypernet-checkpoints](https://huggingface.co/datasets/bobthebuilderinternational/hypernet-checkpoints) (private)
- Includes: anchors, autoencoder, mappers, watertight meshes, SDF samples, shape-SIRENs, image-SIRENs, hypernets

### Restore on a fresh machine

​```bash
git clone https://github.com/BOB-THE-BUILDER-in/Hypernetwork.git /workspace/hypernet
cd /workspace/hypernet
pip install -U "huggingface_hub[cli]"
hf auth login

hf download bobthebuilderinternational/hypernet-checkpoints \
    --repo-type dataset --local-dir /tmp/restore

tar xzf /tmp/restore/tier_essential.tar.gz
tar xzf /tmp/restore/tier_data.tar.gz
tar xzf /tmp/restore/tier_hypernets.tar.gz
​```



# hypernet → shape pipeline

Image-to-3D experiments via per-shape SIRENs and learned weight-space latent codes.

## Pipeline overview

```
24 images → 24 image-SIRENs → 1 hypernet (per object)

hypernet weights → tiny mapper → 128-dim latent → AE decoder → shape-SIREN weights → SDF → mesh
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                  weight-space autoencoder rescues mesh quality
                                  vs predicting 264K weights directly
```

## Layout

- `configs/` — pipeline config (CFG.data, CFG.shape_siren, etc.)
- `src/` — core modules (siren, hypernet, render, watertight)
- `scripts/` — numbered pipeline stages plus experiment scripts:
  - `01_watertight.py` … `80_train_shape_sirens.py` — main pipeline
  - `hypernet_to_shape_mapper.py` — direct weight-prediction mapper (baseline)
  - `autoencoder_pipeline_n100_mlp.py` — latent autoencoder + tiny mapper (current)
  - `scale_to_n100.py` — orchestrator for scaling experiment
  - `ood_test_full.py` — OOD generalization test on a held-out shape

## Reproducing

1. Run `00_download_objaverse.py` … `80_train_shape_sirens.py` in order.
   Set `CFG.data.num_objects = 100` first if you want N=100.
2. `python scripts/autoencoder_pipeline_n100_mlp.py` to train the
   weight-space autoencoder and tiny latent mapper.
3. `python scripts/ood_test_full.py` to test generalization on a new
   Objaverse shape.

## Key findings (April 2026)

- **Direct weight prediction** at N=10 plateaus at MSE 0.143; at N=100
  reaches best-case 0.139 but interpolation between training points
  collapses, and OOD novel shapes produce floating fragments.
- **Weight-space autoencoder** (128-dim latent, MLP architecture) achieves
  reconstruction MSE 0.0026, near-perfect tiny-mapper (MSE 1.4e-5,
  scrambled-cond ratio 9M×), and OOD prediction produces coherent geometry
  rather than fragments.
- Conclusion: at small/medium N, direct weight prediction is too brittle
  due to SIREN sensitivity; learned latent compression rescues OOD
  generalization.
