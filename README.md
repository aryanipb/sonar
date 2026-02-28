# AdaPoinTr Inference Guide (Beginner Friendly)

This guide is written for first-time users.
If you can copy-paste commands, you can run it.

## Goal
You will do 3 things:
1. Set up the environment.
2. Run AdaPoinTr inference.
3. Open an interactive 3D page that shows:
   - actual input object point cloud
   - predicted completed object point cloud

---

## 1) Open Terminal in Project Folder

Project path used in this guide:

```bash
cd /home/aryan/work/sonar/PoinTr
```

If this command works, you are in the correct folder.

---

## 2) Put the Model Checkpoint File in Place

You already have:

`/home/aryan/Downloads/AdaPoinTr_PCN.pth`

Check it exists:

```bash
ls -lh /home/aryan/Downloads/AdaPoinTr_PCN.pth
```

If this prints a file size, you are good.

---

## 3) Setup Environment (One Command)

Run:

```bash
bash scripts/setup_adapointr_inference.sh --smoke-test --config cfgs/PCN_models/AdaPoinTr.yaml
```

What this does:
- installs Python requirements (if needed)
- checks your environment
- builds and runs a tiny model smoke test

You should see a line similar to:
- `smoke_output: [(1, 512, 3), (1, 16384, 3)]`

That means pipeline is healthy.

---

## 4) Run Inference + Create 3D View (Single File)

Run this exact command:

```bash
bash scripts/run_adapointr_inference.sh \
  --config cfgs/PCN_models/AdaPoinTr.yaml \
  --ckpt /home/aryan/Downloads/AdaPoinTr_PCN.pth \
  --pc demo/airplane.pcd \
  --out inference_result/adapointr_single \
  --save-vis \
  --make-3d-view
```

What this creates:
- predicted point cloud (`fine.npy`)
- input and predicted JPG previews
- interactive 3D comparison HTML file

Output folder:

`inference_result/adapointr_single/airplane/`

Expected files:
- `fine.npy`
- `input.jpg`
- `fine.jpg`
- `actual_vs_pred_3d.html`

---

## 5) Run Inference + 3D View (Whole Demo Folder)

Run:

```bash
bash scripts/run_adapointr_inference.sh \
  --config cfgs/PCN_models/AdaPoinTr.yaml \
  --ckpt /home/aryan/Downloads/AdaPoinTr_PCN.pth \
  --pc_root demo/ \
  --out inference_result/adapointr_demo \
  --save-vis \
  --make-3d-view
```

This processes all point clouds inside `demo/`.

---

## 6) Open the Interactive 3D View

Open one HTML result in browser (example):

```bash
xdg-open /home/aryan/work/sonar/PoinTr/inference_result/adapointr_single/airplane/actual_vs_pred_3d.html
```

If `xdg-open` is unavailable, open the file manually in your file manager/browser.

---

## 7) Understand the Result Quickly

In each 3D HTML:
- Left panel = actual input point cloud (incomplete object)
- Right panel = predicted completed point cloud (model output)

You can:
- drag to rotate
- scroll to zoom
- hold right click to pan

---

## 8) Important Notes About Your System

- Your current runtime can run this pipeline without strict old CUDA pinning.
- If CUDA is unavailable at runtime, the script auto-falls back to CPU.
- CUDA extensions like `chamfer` are optional for inference in this pipeline.

---

## 9) Useful Extra Commands

### Check environment quickly

```bash
./env5/bin/python tools/check_inference_env.py --config cfgs/PCN_models/AdaPoinTr.yaml
```

### Build only 3D compare HTML from existing input + output

```bash
./env5/bin/python tools/visualize_actual_vs_pred.py \
  --input_pc demo/airplane.pcd \
  --pred_pc inference_result/adapointr_single/airplane/fine.npy \
  --output_html inference_result/adapointr_single/airplane/actual_vs_pred_3d.html
```

---

## 10) Troubleshooting

### A) `No checkpoint file from path ...`
Cause: wrong checkpoint path.
Fix: verify file exists:

```bash
ls -lh /home/aryan/Downloads/AdaPoinTr_PCN.pth
```

### B) `CUDA was requested but is unavailable. Falling back to CPU.`
This is safe. Inference will still run on CPU.

### C) 3D HTML not generated
Check whether `fine.npy` exists in that sample’s output folder.
If yes, run the standalone visualize command in section 9.

### D) `plotly is required`
Install it:

```bash
./env5/bin/python -m pip install plotly
```

---

## 11) Full One-Shot Command (Copy-Paste)

Single sample with everything:

```bash
cd /home/aryan/work/sonar/PoinTr && \
bash scripts/setup_adapointr_inference.sh --smoke-test --config cfgs/PCN_models/AdaPoinTr.yaml && \
bash scripts/run_adapointr_inference.sh \
  --config cfgs/PCN_models/AdaPoinTr.yaml \
  --ckpt /home/aryan/Downloads/AdaPoinTr_PCN.pth \
  --pc demo/airplane.pcd \
  --out inference_result/adapointr_single \
  --save-vis \
  --make-3d-view
```

You can now open:

`/home/aryan/work/sonar/PoinTr/inference_result/adapointr_single/airplane/actual_vs_pred_3d.html`

