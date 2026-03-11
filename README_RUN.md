Run and collect example output (WSL / Ubuntu)
===========================================

Use this when you want to run the example locally in your Ubuntu/WSL session and collect the full log for debugging.

Steps (run inside WSL / Ubuntu):

1. Make the helper script executable:

```bash
chmod +x /mnt/c/PROJETOS/BIONIX-ML/scripts/run_and_collect.sh
```

2. Run it (this will save a log in `/tmp` and print head/tail):

```bash
/mnt/c/PROJETOS/BIONIX-ML/scripts/run_and_collect.sh
```

3. Copy the printed output (head/tail) or the full log path shown and paste it here. I will analyze and provide fixes.

If `pixi` is not found, ensure it's installed and in your PATH in the WSL session. You can verify with:

```bash
which pixi
pixi --version
```

If `pixi` is missing and you want installation steps, tell me your Ubuntu version and I will provide commands.

Kaggle GPU execution
--------------------

For remote GPU validation (no local GPU), use:

- `README_KAGGLE_GPU.md`
- `scripts/run_cuda_smoke_kaggle.sh`

This path executes:

1. CUDA smoke test (`std.gpu`) in `exemplos/e000005_reconhecimento_digitos_cuda/smoke_test_cuda.mojo`
2. Full CUDA example entrypoint `exemplos/e000005_reconhecimento_digitos_cuda/principal.mojo`
