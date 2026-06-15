# Plan: 1 notebook báo cáo 4DGS — Setup + 3 dataset (D-NeRF, HyperNeRF, NeRF-DS)

## Context

Tái cấu trúc `4DGaussians.ipynb` thành **MỘT notebook** trình bày khoa học, chạy **THỦ CÔNG từng cell / từng dataset**. Clone từ fork đã chuẩn bị `https://github.com/MinhThang1009/4D-Gaussian` (env-compat: vendored submodules + mmcv→mmengine + bỏ pin torch/argparse + font DejaVu + env-var `HYPER_RATIO`) → Setup **clone-only** (không vá runtime).

**Khung bài tập (thầy):** "paper có code → chạy thử, **áp dụng dataset KHÁC**, **so sánh**."
- *Chạy thử* = 4DGS trên dataset paper: **D-NeRF (synthetic) + HyperNeRF (real)**.
- *Dataset khác* = **NeRF-DS** (real specular, ngoài paper).
- *So sánh* = 2 trục: (1) của-tôi vs số CÔNG BỐ (validate tái hiện); (2) **NeRF-DS cạnh HyperNeRF/D-NeRF** (mục tiêu chính). Chi tiết PHẦN 5.

**Scope (user chốt):** chỉ **3 dataset** D-NeRF + HyperNeRF + NeRF-DS. **BỎ Neu3D/DyNeRF** (data Plenoptic gated + GB video, bắt buộc COLMAP nặng → dễ fail Colab). 3 dataset này đều dùng-ngay/bỏ-COLMAP nên khả thi.

**Kiến trúc:** PHẦN 1 Setup (1 lần) + PHẦN 2 D-NeRF + PHẦN 3 HyperNeRF + PHẦN 4 NeRF-DS + PHẦN 5 Tổng hợp (partial-friendly). 1 runtime dùng chung Setup; resume-skip + Drive giữ tiến độ.

**Deliverable:** GHI ĐÈ `4D-Gaussians.ipynb` tại chỗ (user chốt giữ tên gạch nối). Build script đọc chính `4D-Gaussians.ipynb` làm nguồn → **backup `4D-Gaussians.ipynb.bak`** → dựng cấu trúc mới → ghi đè `4D-Gaussians.ipynb`.

## Quy chuẩn đặt tên (đồng nhất toàn notebook)
**`scene id = tên file config 4DGS`** (lowercase, không prefix/hậu tố). folder data = scene id; `--expname "<ds>/<scene>"`; output = `output/<ds>/<scene>`; checkpoint = `output/<ds>/<scene>/point_cloud/iteration_<IT>`.
- D-NeRF: bouncingballs, hellwarrior, hook, jumpingjacks, lego, mutant, standup, trex
- HyperNeRF: 3dprinter, broom2, chicken, banana
- NeRF-DS: as, basin, bell, cup, plate, press, sieve

## ✅ Verify thông số + data (ĐÃ KIỂM TRỰC TIẾP)
Codebase `4DGS-codebase/` (file:line):
- iterations: D-NeRF **20000** (`dnerf/dnerf_default.py:11`), HyperNeRF/NeRF-DS **14000** (`hypernerf/default.py:18`); coarse 3000. Scene config đều `_base_` → kế thừa iter đúng (verify `dnerf/bouncingballs.py`, `hypernerf/chicken.py`, `broom2.py`).
- `train.py:711` save_iterations + `:720 append(args.iterations)` → checkpoint iter cuối chắc chắn.
- `metrics.py:88-93,106,108` → `results.json` (6 metric: SSIM/PSNR/LPIPS-vgg/LPIPS-alex/MS-SSIM/D-SSIM) + `per_view.json`.
- `render.py:57-90` test→`model_path/test/ours_<IT>/{renders,gt}`, video→`model_path/video/ours_<IT>/video_rgb.mp4`; **`render.py:83` in `FPS:`** → `tee` parse được.
- `dataset_readers.py:373-378` (fork) ratio qua `HYPER_RATIO` env (default 0.5 → rgb/2x). `hyper_loader.py:96` map `rgb/{int(1/ratio)}x`.

Data thật (range-peek/gdown zip):
- **NeRF-DS** release v0.1 (1 zip 2.1GB, 7 scene `*_novel_view`): `rgb/1x`=**480×270** ✓ → `HYPER_RATIO=1.0`; `dataset.json`/`points.npy`/`scene.json`/`metadata.json` đủ.
- **HyperNeRF** vrig: `rgb/2x` tồn tại (ratio 0.5 ✓); `dataset.json`/`points.npy` đủ. 4 asset + tên folder (xem recipe Phần 3).
- **hypernerf-pcd.zip** (Drive, 37.9MB): top `hypernerf/{interp,virg}/`, mỗi scene 1 `points3D_downsample2.ply` (~2-5MB). Folder virg dùng convention RIÊNG → cần mapping dict.

## Nguyên tắc SỐ LIỆU (fail-loud, chính xác 100%)
`!python` KHÔNG raise khi lỗi → chặn:
1. **Assert sau mỗi `!python`:** train→ply+deformation.pth ở iter cuối; render→đủ `renders/*.png`; metrics→results.json đủ 6 metric finite. Thiếu→`raise`.
2. **Khớp #frame:** `#renders==#gt==#test kỳ vọng` (D-NeRF: `transforms_test.json`; HyperNeRF/NeRF-DS: `dataset.json` val_ids). Lệch→`raise`.
3. **Resume-skip CHỈ khi xác minh hoàn chỉnh** (=điều kiện assert).
4. **Báo cáo hard-fail:** đủ scene + khớp frame mới ghi summary/zip; thiếu→`raise`. Loại scene tường minh.
5. Bảng có provenance (iter+#frame); TRUNG BÌNH chỉ in khi đủ scene.
6. Không `except: pass`/`continue` nuốt lỗi trong cell mới.

## PHẦN 1 — Setup (chạy 1 lần)
- **1.1 Môi trường:** auto-detect Colab/Kaggle→`ROOT`; `download_file`; `TF_CPP_MIN_LOG_LEVEL=2`; mount Drive (Colab). `DRIVE_ROOT=MyDrive/4DGaussians`, `DRIVE_OUT={DRIVE_ROOT}/output` (đổi tên `checkpoints`→`output` cho đúng nghĩa: chứa cả model+renders+video+metrics, khớp tên dir 4DGS gốc), `DRIVE_REPORTS=.../reports`, `DRIVE_FIGS=.../figures`. Migration `4dgs_output`→`output` (giữ scene đã train). Kaggle: `DRIVE_*=None`, guard `if DRIVE_OUT:`.
- **1.2 Cài & build (CLONE-ONLY):**
  ```
  [ -d 4DGaussians/.git ] || git clone https://github.com/MinhThang1009/4D-Gaussian 4DGaussians
  pip install -r requirements.txt        # sạch: mmengine, KHÔNG torch/argparse/mmcv
  pip install gdown                       # tải PLY HyperNeRF từ Drive
  apt-get install -y libglm-dev
  pip install -e submodules/depth-diff-gaussian-rasterization
  pip install -e submodules/simple-knn
  ```
  Bỏ: submodule init, lọc argparse, sed float.h, sed font, patch mmcv (đã xử lý trong fork).
  - **⚠️ CAVEAT:** torch 2.x ≠ torch 1.13.1+cu116 (paper) → số có thể lệch nhẹ vs công bố. Ghi rõ fork chỉ env-compat.
- **1.3 Symlink:** `output/`→`DRIVE_OUT`. Auto-sync mức SCENE. Resume-skip mức scene. Kaggle: bỏ symlink.
- **1.4 Helper:** `display_video` + `_iter_of`, `mark`, `check_scene`, `assert_scene_complete(base,it,n_frames)`, `summary_table(ds,scenes,it,highlight,paper_ref)`, `make_report(ds,scenes,it)`.

## Đa nền tảng Colab + Kaggle
Train/render/metrics/config/iter/assert/summary giống hệt, chỉ khác nơi lưu. Colab: Drive+symlink+copy reports/figures. Kaggle: `DRIVE_*=None` guard, persist qua Settings→Persistence, báo cáo tab Output; cần Internet=On + GPU.

## PHẦN 2..4 — mỗi dataset KHUNG 5 bước (a→e)
a. Chuẩn bị data · b. Train+Render+Video từng scene (3 cell/scene) · c. Metrics (loop+skip+assert) · d. Báo cáo (`make_report`+bảng đối sánh+zip+Drive) · e. Trạng thái (`check_scene`).

| Phần | DS | Scene | Data nguồn | IT | config | Metric nhấn + Paper |
|---|---|---|---|---|---|---|
| 2 | dnerf | 8 | HF `camenduru/4DGaussians` data.zip → `{ROOT}/test/data/<S>` | 20000 | `arguments/dnerf/<S>.py` | PSNR/SSIM/LPIPS ← `table_dnerf` (34.05/0.98/0.02) |
| 3 | hypernerf | 4 | google/hypernerf v0.1 vrig zip + PLY pregenerated → `data/hypernerf/virg/<S>` | 14000 | `arguments/hypernerf/<S>.py` | PSNR/MS-SSIM ← `tab:hypernerf` (25.2/0.845) |
| 4 | nerfds | 7 | JokerYan/NeRF-DS v0.1 zip → `data/nerfds/<S>`; npy→ply; **`HYPER_RATIO=1.0`** | 14000 | `arguments/hypernerf/default.py` | PSNR/SSIM/LPIPS-vgg ← SpectroMotion (22.79/0.8235/0.2115) |

**Recipe HyperNeRF (Phần 3) — ĐÃ VERIFY ĐẦY ĐỦ:**
- Tải 4 scene: `https://github.com/google/hypernerf/releases/download/v0.1/<asset>` rồi unzip + rename về scene id chuẩn. **Mapping tường minh (bắt buộc — 3 nguồn 3 convention):**

  | scene id | asset zip | folder trong zip → rename | folder PLY trong `hypernerf/virg/` |
  |---|---|---|---|
  | 3dprinter | vrig_3dprinter.zip | `vrig-3dprinter` | `virg-3dprinter` |
  | broom2 | vrig_broom.zip | `broom2` | `broom2` |
  | chicken | vrig_chicken.zip | `vrig-chicken` | `virg-chickchicken` |
  | banana | vrig_peel-banana.zip | `vrig-peel-banana` | `peel-banana` |

- PLY pregenerated: `gdown 1fUHiSgimVjVQZ2OOzTFtz02E9EqCoWr5` → `hypernerf-pcd.zip` → copy `hypernerf/virg/<plyfolder>/points3D_downsample2.ply` vào `data/hypernerf/virg/<scene id>/points3D_downsample2.ply`. Assert fail-loud nếu thiếu ply scene nào → KHÔNG chạy COLMAP.
- ratio 0.5 (rgb/2x) = default, KHÔNG set env. iter 14000.

**Pattern cell từng-scene:**
```python
import os, time
%cd {ROOT}/4DGaussians
_ckpt=f"{ROOT}/4DGaussians/output/<DS>/<S>/point_cloud/iteration_<IT>"
if all(os.path.exists(f"{_ckpt}/{f}") for f in ["point_cloud.ply","deformation.pth"]):
    print("⏭️ <S>: đã train xong — bỏ qua")
else:
    _t=time.time()
    !python train.py -s <DATA> --port 6017 --expname "<DS>/<S>" --configs <CFG>
    assert all(os.path.exists(f"{_ckpt}/{f}") for f in ["point_cloud.ply","deformation.pth"]), "❌ <S>: train KHÔNG ra checkpoint — DỪNG."
    open(f"{_ckpt}/train_time_sec.txt","w").write(str(time.time()-_t))
```
Render: `!python render.py --model_path "output/<DS>/<S>/" --skip_train --configs <CFG> 2>&1 | tee {base}/render_log.txt` + assert đủ frame. Video: `display_video output/<DS>/<S>/video/ours_*/video_rgb.mp4`.
**NeRF-DS Phần 4:** `os.environ["HYPER_RATIO"]="1.0"` ở cell Python TRƯỚC mọi `!python` của phần này. HyperNeRF Phần 3 KHÔNG set → 0.5.

**Mốc NeRF-DS = hàng 4DGS trong SpectroMotion** (nguồn DUY NHẤT có 4DGS-on-NeRF-DS; chỉ TRÍCH DẪN). Mean 22.79/0.8235/0.2115 + per-scene (as 24.85/basin 19.26/bell 22.86/cup 23.82/plate 18.77/press 24.82/sieve 25.16). `SpectroMotion-paper/` đã xóa → tải lại PDF cho citation (số đã trích, không chặn build). Cross-method NeRF-DS paper: HyperNeRF 22.7/.849/.192; NeRF-DS(Ours) 23.4/.890/.135.

## PHẦN 5 — Tổng hợp & ĐỐI SÁNH (cốt lõi, partial-friendly)
Đọc `DRIVE_REPORTS/<ds>/summary.csv` mọi dataset đã có; bỏ qua dataset chưa chạy (in "chưa có", không raise).
- **Trục 1 — Per-dataset (validate tái hiện):** D-NeRF↔table_dnerf, HyperNeRF↔tab:hypernerf, NeRF-DS↔SpectroMotion.
- **Trục 2 — Cross-dataset (mục tiêu chính):** 1 bảng 4DGS trên 3 dataset, cột chung PSNR/SSIM/LPIPS-vgg + FPS/TrainTime/Storage/#Gaussians. **NeRF-DS (real specular) so CHÍNH với HyperNeRF (real)**; D-NeRF là mốc SYNTHETIC (đọc tách biệt). Nhận xét: 4DGS kém hơn trên NeRF-DS → giới hạn với vật phản chiếu động. Caveat: khác scene/độ phân giải → đọc xu hướng, không xếp hạng tuyệt đối.
- **Biểu đồ** (matplotlib, dpi≥150, title+nhãn trục+đơn vị, `ax.bar_label`, grid alpha 0.3, palette tab10/Set2, `tight_layout`, `bbox_inches='tight'`): `bar_repro_vs_paper.png` (grouped bar Trục 1), `bar_cross_dataset.png` (grouped bar Trục 2), `scatter_fps_gaussians.png` (scatter 1 điểm/scene+nhãn). Montage render|GT mỗi dataset. Lưu `DRIVE_FIGS/` (snake_case) + zip + inline.
- **Lưu BẢNG dạng file** (persist số liệu, không chỉ chart): ghi `DRIVE_REPORTS/comparison_repro_vs_paper.csv` (Trục 1) + `DRIVE_REPORTS/comparison_cross_dataset.csv` (Trục 2).
- **`DRIVE_REPORTS/RUN_SUMMARY.md`**: tổng run (ngày, GPU `nvidia-smi`, commit fork, dataset/scene đã xong, vài số headline mỗi dataset). Partial-friendly: chỉ ghi dataset đã có.

## Số liệu (mọi dataset cùng bộ cột)
`results.json` (6 metric) + `per_view.json` + #Gaussians (vertex `point_cloud.ply`) + Storage(MB) (`iteration_<IT>/*.{ply,pth}`) + FPS (parse `render_log.txt` `FPS:`) + TrainTime (`train_time_sec.txt`) + provenance (GPU/iter/#frame/commit/ngày → `run_info.json`). FPS/TrainTime thiếu → "N/A" (không bịa).

## Cấu trúc thư mục (3 vùng)
**A. Runtime `{ROOT}`** (ephemeral): `4DGaussians/` (clone) + `data/{hypernerf/virg/<s>, nerfds/<s>}` + `{ROOT}/test/data/<s>` (D-NeRF) + `output/`→symlink Drive; `*_report.zip` staging.
**B. Drive `MyDrive/4DGaussians/`** (bền):
- `output/<ds>/<scene>/{point_cloud/iteration_<IT>/{point_cloud.ply,deformation.pth,train_time_sec.txt}, test/ours_<IT>/{renders,gt}, video/ours_<IT>/video_rgb.mp4, results.json, per_view.json, render_log.txt, cfg_args}`
- `reports/RUN_SUMMARY.md` (tổng run: ngày/GPU/commit/dataset đã xong/số headline) · `reports/comparison_repro_vs_paper.csv` (Trục 1) · `reports/comparison_cross_dataset.csv` (Trục 2) · `reports/<ds>/{<ds>_report.zip, summary.csv, summary.txt, run_info.json}`
- `figures/{bar_repro_vs_paper.png, bar_cross_dataset.png, scatter_fps_gaussians.png, montage_<ds>.png}`
**C. Tải về:** `<ds>_report.zip` ×3 (Colab download / Kaggle tab Output).

## Format / quy ước
nbformat 4, kernel Python 3 (giữ metadata gốc). Markdown + comment tiếng Việt; identifier/lệnh tiếng Anh. Heading: `#` notebook · `##` PHẦN (1–5) · `###` bước/scene. Mục lục đầu file = bảng. Emoji mốc. Comment WHY.

## Tham khảo (cell cuối)
Papers: 4DGS (CVPR2024), 3DGS (SIGGRAPH2023), D-NeRF (CVPR2021), HyperNeRF (SIGGRAPH Asia 2021), NeRF-DS (CVPR2023), SpectroMotion. Code: gốc `hustvl/4DGaussians` + fork env-compat `MinhThang1009/4D-Gaussian` (ghi rõ chỉ sửa tương thích môi trường), JokerYan/NeRF-DS, google/hypernerf, data D-NeRF HF camenduru. WebFetch verify URL/paper trước khi viết.

## Cân nhắc bổ sung (chốt khi audit — áp dụng lúc build)
1. **Disk Colab:** mỗi cell data-prep **XÓA zip sau khi extract+verify** (gate resume = data đã giải nén, không phải zip) → tiết kiệm ~7GB. *(tùy chọn, flag `PRUNE_RGB=False` mặc định)* xóa rgb variant thừa: HyperNeRF chỉ dùng `rgb/2x`, NeRF-DS chỉ `rgb/1x` — zip có cả 1x/2x/4x/8x/16x.
2. **Verify số paper TRƯỚC khi nhúng PHẦN 5:** đọc `4DGS-paper/` (local) cho D-NeRF + HyperNeRF (+ xác nhận mean hypernerf đúng 4 scene 3dprinter/broom2/chicken/banana); SpectroMotion (đã xóa local) → tải lại PDF verify số NeRF-DS, hoặc ghi rõ "trích từ SpectroMotion, chưa re-verify". KHÔNG bịa số.
3. **gdown fail-loud:** PLY pregenerated qua gdown có thể lỗi quota Drive → assert + in URL/file-id (`1fUHiSgimVjVQZ2OOzTFtz02E9EqCoWr5`) để tải tay, KHÔNG nuốt lỗi.
4. **Trục 1 hiển thị DELTA** (của-tôi − paper) cạnh số thô → trực quan độ sát tái hiện.
5. **Display TOGGLE được — cờ `SHOW_MEDIA` (mặc định False, `.ipynb` nhẹ):**
   - Render (tạo video+renders): bắt buộc — metrics+report cần; file lưu Drive.
   - **`SHOW_MEDIA = False`** global (cell 1.1) điều khiển toàn bộ display. Giữ helper `display_video`.
   - **Mỗi scene = md + train + render + video(guarded):** cell video `if SHOW_MEDIA: display(display_video(v))` else in path. Mặc định không nhúng → nhẹ.
   - PHẦN 5: charts + montage **luôn `savefig` vào `figures/`** (deliverable Drive) + `plt.show() if SHOW_MEDIA else plt.close()`.
   - Default off → không base64, `.ipynb` nhẹ, commit/agent đọc được. Đổi `SHOW_MEDIA=True` (hoặc sửa lẻ từng cell) để xem inline khi cần.
   - Visual để nộp = Drive `figures/` + `output/<ds>/<scene>/video/` + `<ds>_report.zip`. Verify = số (results.json/CSV) + PIL/numpy.

## Cách thực hiện
1. Script Python đọc `4D-Gaussians.ipynb` → backup `4D-Gaussians.ipynb.bak` → dựng cấu trúc mới → **ghi đè `4D-Gaussians.ipynb`**. Giữ metadata/nbformat; tái dùng cell D-NeRF + NeRF-DS (đã chạy OK) + author cell mới (Setup clone-only, helper 1.4, Phần 3 HyperNeRF, Phần 5 tổng hợp).
2. #frame test đọc runtime (transforms_test.json / dataset.json val_ids).

## Verification
- `nbformat.validate`; id duy nhất; metadata.kernelspec còn; magic `%cd`/`!python` còn; Setup 1 lần.
- Unit-test mock-fs: `summary_table` (6 metric+#Gaussians+Storage+FPS parse+TrainTime, trung bình đúng, thiếu→RAISE, FPS thiếu→"N/A"); `assert_scene_complete` (thiếu ckpt/frame→RAISE); `check_scene` không NameError; migration idempotent; Phần 5 thiếu dataset→bỏ qua không raise; HyperNeRF mapping dict đủ 4 scene.
- Đối chiếu lệnh 3 dataset == repo gốc (config/expname/model_path) — đã verify.
- **Giới hạn:** không chạy train/render/metrics thật ở đây (cần GPU/Colab); chỉ test logic mock. End-to-end user chạy Colab.
