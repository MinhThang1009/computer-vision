# -*- coding: utf-8 -*-
"""Dựng 4D-Gaussians.ipynb — báo cáo 4DGS trên D-NeRF + HyperNeRF + NeRF-DS.
Đọc 4D-Gaussians.ipynb làm nguồn (lấy metadata) -> backup .bak -> ghi đè cấu trúc mới."""

import json, shutil, os

SRC = "4D-Gaussians.ipynb"
nb_src = json.load(open(SRC, encoding="utf-8"))
shutil.copy(SRC, SRC + ".bak")

cells = []


def _to_source(text):
    text = text.strip("\n")
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]] if len(lines) > 1 else [text]


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _to_source(text)})


def code(text):
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": _to_source(text),
        }
    )


def sub(num, icon, title):
    md(f"### {num} {icon} {title}")


def subsub(num, icon, title):
    md(f"#### {num} {icon} {title}")


# ============================================================ TITLE + TOC
md(r"""# 🚀 4D Gaussian Splatting — Báo cáo thực nghiệm & đối sánh

<div align="center">

**Phương pháp:** 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering (CVPR 2024)
<br>
**Code:** [MinhThang1009/4D-Gaussian](https://github.com/MinhThang1009/4D-Gaussian) (dựa trên [hustvl/4DGaussians](https://github.com/hustvl/4DGaussians))
<br>
**Datasets:** D-NeRF (8 · synthetic) · HyperNeRF (4 · real) · NeRF-DS (7 · real specular)
<br>
**Nền tảng:** Google Colab / Kaggle (tự phát hiện, GPU)

</div>

---

## 🎯 Mục tiêu (theo đề bài)

Chạy thử code 4DGS trên **dataset của paper** (D-NeRF, HyperNeRF) → **áp dụng dataset KHÁC** (NeRF-DS, ngoài paper) → **so sánh** theo 2 trục:
1. **Của tôi vs số CÔNG BỐ** — validate tái hiện đúng code.
2. **NeRF-DS (cảnh phản chiếu động) cạnh HyperNeRF/D-NeRF** — mục tiêu chính, đặc tả hành vi 4DGS.

---

## 📋 Mục lục

| # | Phần | Nội dung | Scene | iter | Ước tính |
|:---:|---|---|:---:|:---:|:---:|
| **1** | 🔧 [Setup](#phan-1) | Clone + build + Drive + helper | — | — | ~5 phút |
| **2** | 🧊 [D-NeRF](#phan-2) | Synthetic | 8 | 20000 | ~2 giờ |
| **3** | 🎥 [HyperNeRF](#phan-3) | Real | 4 | 14000 | ~4 giờ |
| **4** | ✨ [NeRF-DS](#phan-4) | Real specular (dataset KHÁC) | 7 | 14000 | ~4 giờ |
| **5** | 📊 [Tổng hợp](#phan-5) | 2 trục đối sánh + biểu đồ | — | — | ~2 phút |
| **6** | 📚 [Tham khảo](#phan-6) | Papers + code + data | — | — | — |

---

> ⚠️ **Quan trọng:** chạy **THỦ CÔNG từng cell / từng dataset** (KHÔNG "Run all"). Mỗi phần tự đủ báo cáo; tự bỏ qua phần đã chạy xong + Drive giữ tiến độ qua phiên.
>
> 🔵 **Kaggle:** bật **Settings → Internet=On** (verify SĐT) + **Accelerator=GPU**; báo cáo ở tab **Output**.
>
> 🟡 **Ảnh/video inline:** mặc định **tắt** (`SHOW_MEDIA=False`) cho `.ipynb` nhẹ — đặt `True` ở cell 1.1 khi cần xem. File đầy đủ ở Drive `figures/` + `output/.../video/` + `*_report.zip`.
>
> 🟢 **Popup "restart runtime" (Colab):** nếu hiện sau Setup → bấm **Cancel**, bỏ qua an toàn.
""")

# ============================================================ PHẦN 1 — SETUP
md(r"""---

<a name="phan-1"></a>
## 🔧 PHẦN 1 — Setup (chạy 1 lần mỗi phiên)

Clone code 4DGS → cài thư viện → nối Google Drive (lưu kết quả bền, resume qua phiên) → định nghĩa các hàm dùng chung.

> ⚠️ Môi trường Colab (torch 2.x / CUDA 12) khác bản gốc của paper (torch 1.13.1) → số liệu có thể lệch nhẹ so với công bố — bình thường khi tái hiện trên môi trường mới.""")

sub("1.1", "🌐", "Môi trường & Drive")

code(r'''# === 1.1 Môi trường: phát hiện Colab/Kaggle + Drive + đường dẫn ===
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # giảm log INFO của TF (kéo theo bởi tensorboard)

SHOW_MEDIA = False  # True = hiện ảnh/video/biểu đồ inline (notebook NẶNG); False = chỉ lưu Drive (nhẹ)

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "KAGGLE_URL_BASE" in os.environ
IS_COLAB = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ
if IS_KAGGLE:   ROOT = "/kaggle/working"
elif IS_COLAB:  ROOT = "/content"
else: raise RuntimeError("Notebook cần chạy trên Colab hoặc Kaggle.")

def download_file(path):
    """Colab: tải file về máy. Kaggle: file nằm ở tab Output."""
    if IS_KAGGLE:
        print(f"📁 Kaggle: file tại {path} — tải ở tab Output (Save Version để giữ lâu dài).")
    else:
        from google.colab import files; files.download(path)

# Drive: lưu bền + resume. Kaggle (không Drive) -> DRIVE_*=None, mọi thao tác Drive được guard.
DRIVE_ROOT = DRIVE_OUT = DRIVE_REPORTS = DRIVE_FIGS = None
if IS_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        DRIVE_ROOT = "/content/drive/MyDrive/4DGaussians"
        DRIVE_OUT = f"{DRIVE_ROOT}/output"
        DRIVE_REPORTS = f"{DRIVE_ROOT}/reports"
        DRIVE_FIGS = f"{DRIVE_ROOT}/figures"
        for d in (DRIVE_OUT, DRIVE_REPORTS, DRIVE_FIGS):
            os.makedirs(d, exist_ok=True)
        # Migration cấu trúc cũ: 4dgs_output -> output (giữ scene đã train phiên trước, merge mức scene)
        _old = "/content/drive/MyDrive/4dgs_output"
        if os.path.isdir(_old):
            import shutil
            for _ds in os.listdir(_old):
                _sd = f"{_old}/{_ds}"
                if not os.path.isdir(_sd): continue
                os.makedirs(f"{DRIVE_OUT}/{_ds}", exist_ok=True)
                for _sc in os.listdir(_sd):
                    _dst = f"{DRIVE_OUT}/{_ds}/{_sc}"
                    if not os.path.exists(_dst):
                        shutil.move(f"{_sd}/{_sc}", _dst)
            print("🔄 Migration 4dgs_output -> output xong")
        print(f"✅ Drive: {DRIVE_ROOT}")
    except Exception as e:
        DRIVE_ROOT = DRIVE_OUT = DRIVE_REPORTS = DRIVE_FIGS = None
        print(f"⚠️ Không mount được Drive ({type(e).__name__}) — chạy local, KHÔNG resume qua phiên.")
elif IS_KAGGLE:
    print("ℹ️ Kaggle: bật Settings → Persistence=Files để giữ output; cần Internet=On + GPU.")

print(f"Nền tảng: {'Kaggle' if IS_KAGGLE else 'Colab'} | ROOT={ROOT} | DRIVE_OUT={DRIVE_OUT} | SHOW_MEDIA={SHOW_MEDIA}")''')

sub("1.2", "📦", "Cài đặt & build")
code(r"""# === 1.2 Cài đặt & build ===
%cd {ROOT}
# chỉ clone khi chưa có
![ -d 4DGaussians/.git ] || git clone https://github.com/MinhThang1009/4D-Gaussian 4DGaussians
%cd {ROOT}/4DGaussians
!git pull --ff-only                    # repo đã clone -> lấy commit env-compat mới nhất (render stream-write, HYPER_RATIO)
!pip install -q -r requirements.txt
!pip install -q gdown                  # để tải point cloud HyperNeRF từ Google Drive
!apt-get install -y -q libglm-dev
# build 2 CUDA submodule cho rasterizer
!pip install -q -e submodules/depth-diff-gaussian-rasterization
!pip install -q -e submodules/simple-knn
print("✅ Setup xong. Nếu Colab hiện popup 'restart runtime' → bấm Cancel, bỏ qua an toàn.")""")

sub("1.3", "🔗", "Nối output → Drive")
code(r"""# === 1.3 Symlink output/ -> Drive (resume qua phiên) ===
import os, shutil
LOCAL_OUT = f"{ROOT}/4DGaussians/output"
if DRIVE_OUT:
    # nếu lỡ train ra local trước khi nối -> đẩy sang Drive (merge mức scene)
    if os.path.isdir(LOCAL_OUT) and not os.path.islink(LOCAL_OUT):
        for ds in os.listdir(LOCAL_OUT):
            _s = f"{LOCAL_OUT}/{ds}"
            if not os.path.isdir(_s): continue
            os.makedirs(f"{DRIVE_OUT}/{ds}", exist_ok=True)
            for sc in os.listdir(_s):
                _dst = f"{DRIVE_OUT}/{ds}/{sc}"
                if not os.path.exists(_dst): shutil.move(f"{_s}/{sc}", _dst)
        shutil.rmtree(LOCAL_OUT, ignore_errors=True)
    if os.path.islink(LOCAL_OUT) and os.path.realpath(LOCAL_OUT) != os.path.realpath(DRIVE_OUT):
        os.unlink(LOCAL_OUT)
    if not os.path.islink(LOCAL_OUT):
        os.symlink(DRIVE_OUT, LOCAL_OUT)
    print(f"✅ output -> {os.path.realpath(LOCAL_OUT)} (ghi thẳng Drive, resume mức scene)")
else:
    os.makedirs(LOCAL_OUT, exist_ok=True)
    print(f"➖ Không Drive — output local {LOCAL_OUT} (mất khi hết phiên).")""")

sub("1.4", "🧰", "Hàm dùng chung")
code(r'''# === 1.4 Helper dùng chung (chạy 1 lần) ===
import os, glob, re, json, math, shutil, subprocess
from base64 import b64encode
from IPython.display import HTML, display

METRIC_KEYS = ["PSNR", "SSIM", "MS-SSIM", "D-SSIM", "LPIPS-vgg", "LPIPS-alex"]

# Số paper để đối sánh (Trục 1) — đã verify từ PDF gốc:
#   D-NeRF: 4DGS table_dnerf (PSNR 34.05 / SSIM 0.98 / LPIPS 0.02).
#   HyperNeRF: 4DGS tab:hypernerf (PSNR 25.2 / MS-SSIM 0.845).
#   NeRF-DS: SpectroMotion tab:whole_scene_tab hàng 4DGS (PSNR 22.79 / SSIM 0.8235 / LPIPS-vgg 0.2115).
PAPER_REF = {
    "dnerf":     {"PSNR": 34.05, "SSIM": 0.98,   "LPIPS-vgg": 0.02},    # 4DGS, mean 8 scene
    "hypernerf": {"PSNR": 25.2,  "MS-SSIM": 0.845},                      # 4DGS, mean vrig
    "nerfds":    {"PSNR": 22.79, "SSIM": 0.8235, "LPIPS-vgg": 0.2115},  # SpectroMotion (hàng 4DGS, mean)
}
HIGHLIGHT = {  # metric nhấn theo convention từng dataset
    "dnerf":     ["PSNR", "SSIM", "LPIPS-vgg"],
    "hypernerf": ["PSNR", "MS-SSIM"],
    "nerfds":    ["PSNR", "SSIM", "LPIPS-vgg"],
}
# Hiệu năng paper (verify 4DGS-paper/): Train(giây), FPS, Storage(MB). NeRF-DS: SpectroMotion không báo cáo -> bỏ.
PAPER_EFF = {
    "dnerf":     {"train_time_sec": 1200, "fps": 82, "storage_mb": 18},  # 4DGS table_dnerf: 20 phút
    "hypernerf": {"train_time_sec": 3600, "fps": 34, "storage_mb": 61},  # 4DGS tab:hypernerf: 1 giờ
}
EFF_LABEL = {"train_time_sec": ("Train", "s"), "fps": ("FPS", ""), "storage_mb": ("Storage", "MB")}

def display_video(video_path):
    """Nhúng video MP4 inline (base64). CHỈ gọi khi SHOW_MEDIA=True (làm .ipynb nặng)."""
    with open(video_path, "rb") as f:
        mp4 = f.read()
    url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML('<video width=900 controls><source src="%s" type="video/mp4"></video>' % url)

def data_dir_of(ds, scene):
    if ds == "dnerf":     return f"{ROOT}/test/data/{scene}"
    if ds == "hypernerf": return f"{ROOT}/4DGaussians/data/hypernerf/virg/{scene}"
    if ds == "nerfds":    return f"{ROOT}/4DGaussians/data/nerfds/{scene}"
    raise ValueError(f"dataset lạ: {ds}")

def expected_test_frames(ds, scene):
    """#frame test kỳ vọng — khớp loader 4DGS. dnerf: transforms_test.json; hypernerf/nerfds: dataset.json val_ids."""
    d = data_dir_of(ds, scene)
    if ds == "dnerf":
        with open(f"{d}/transforms_test.json", encoding="utf-8") as f:
            return len(json.load(f)["frames"])
    with open(f"{d}/dataset.json", encoding="utf-8") as f:
        dj = json.load(f)
    ids, val = dj["ids"], dj["val_ids"]
    if len(val) == 0:  # hyper_loader: i%4==0 -> +2, bỏ cuối
        idx = [i for i in range(len(ids)) if i % 4 == 0]
        return len([i + 2 for i in idx][:-1])
    sval = set(val)
    return sum(1 for i in ids if i in sval)

def assert_scene_complete(ds, scene, it):
    """Fail-loud: ply+deformation + #renders==#gt==#frame kỳ vọng + results.json đủ 6 metric finite. Trả dict metric."""
    base = f"{ROOT}/4DGaussians/output/{ds}/{scene}"
    ck = f"{base}/point_cloud/iteration_{it}"
    for fn in ("point_cloud.ply", "deformation.pth"):
        assert os.path.exists(f"{ck}/{fn}"), f"❌ {ds}/{scene}: thiếu {fn} (chưa train xong?)"
    nr = len(glob.glob(f"{base}/test/ours_{it}/renders/*.png"))
    ng = len(glob.glob(f"{base}/test/ours_{it}/gt/*.png"))
    exp = expected_test_frames(ds, scene)
    assert nr == ng == exp, f"❌ {ds}/{scene}: #frame lệch (renders={nr}, gt={ng}, kỳ vọng={exp}) — mean sẽ SAI, DỪNG."
    with open(f"{base}/results.json", encoding="utf-8") as f:
        m = (json.load(f).get(f"ours_{it}") or {})
    for k in METRIC_KEYS:
        v = m.get(k)
        assert isinstance(v, (int, float)) and math.isfinite(v), f"❌ {ds}/{scene}: results.json thiếu/không hợp lệ '{k}'"
    return {k: float(m[k]) for k in METRIC_KEYS}

def count_gaussians(ply_path):
    """Đọc 'element vertex N' từ header PLY (không load cả file)."""
    if not os.path.exists(ply_path): return None
    with open(ply_path, "rb") as f:
        for line in f:
            if line.startswith(b"element vertex"): return int(line.split()[-1])
            if line.strip() == b"end_header": break
    return None

def parse_fps(log_path):
    """FPS từ render_log.txt (render.py in 'FPS: x'). Thiếu -> None (không bịa)."""
    if not os.path.exists(log_path): return None
    txt = open(log_path, encoding="utf-8", errors="replace").read()
    m = re.findall(r"FPS:\s*([\d.]+)", txt)
    return round(float(m[-1]), 2) if m else None

def storage_mb(ckpt_dir):
    tot = sum(os.path.getsize(p) for p in glob.glob(f"{ckpt_dir}/*") if os.path.isfile(p))
    return round(tot / 1e6, 2)

def _gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return out.splitlines()[0] if out else "unknown"
    except Exception:
        return "unknown"

def _commit():
    try:
        return subprocess.run(["git", "-C", f"{ROOT}/4DGaussians", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip() or "unknown"
    except Exception:
        return "unknown"

def summary_table(ds, rows, it):
    """Bảng per-scene + MEAN + đối sánh paper (metric nhấn + delta). Trả (mean_dict, text)."""
    out = []
    P = out.append
    P(f"=== {ds.upper()} — {len(rows)} scene (iter {it}) ===")
    head = f"{'scene':14}" + "".join(f"{k:>11}" for k in METRIC_KEYS) + f"{'#Gauss':>10}{'MB':>8}{'FPS':>7}{'Train(s)':>9}{'#fr':>5}"
    P(head); P("-" * len(head))
    for r in rows:
        line = f"{r['scene']:14}" + "".join(f"{r[k]:>11.4f}" for k in METRIC_KEYS)
        line += f"{(r['n_gaussians'] or 0):>10}{(r['storage_mb'] or 0):>8.1f}"
        line += f"{(str(r['fps']) if r['fps'] is not None else 'N/A'):>7}"
        line += f"{(str(r['train_time_sec']) if r['train_time_sec'] is not None else 'N/A'):>9}{r['n_frames']:>5}"
        P(line)
    P("-" * len(head))
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in METRIC_KEYS}
    P(f"{'MEAN':14}" + "".join(f"{mean[k]:>11.4f}" for k in METRIC_KEYS))
    P("")
    P(f"Đối sánh paper (metric nhấn {HIGHLIGHT.get(ds, [])}):")
    ref = PAPER_REF.get(ds, {})
    for k in HIGHLIGHT.get(ds, []):
        if k in ref:
            P(f"  {k:10}: của tôi {mean[k]:.4f} | paper {ref[k]:.4f} | Δ {mean[k]-ref[k]:+.4f}")
    eff = PAPER_EFF.get(ds, {})
    if eff:
        P("")
        P("Đối sánh hiệu năng paper (FPS/Train tuỳ GPU -> đọc xu hướng; Storage so trực tiếp):")
        for k in ("train_time_sec", "fps", "storage_mb"):
            vals = [r[k] for r in rows if r.get(k) is not None]
            if k in eff and vals:
                m = sum(vals) / len(vals); lbl, unit = EFF_LABEL[k]
                P(f"  {lbl:10}: của tôi {m:.1f}{unit} | paper {eff[k]}{unit}")
    text = "\n".join(out)
    print(text)
    return mean, text

def make_report(ds, scenes, it):
    """Gom báo cáo 1 dataset: assert từng scene -> copy artifact + summary.csv/txt + run_info.json + zip + Drive. Fail-loud."""
    import csv, datetime
    report_dir = f"{ROOT}/report_{ds}"
    if os.path.exists(report_dir): shutil.rmtree(report_dir)
    os.makedirs(report_dir, exist_ok=True)
    rows, done = [], []
    for scene in scenes:
        base = f"{ROOT}/4DGaussians/output/{ds}/{scene}"
        if not os.path.exists(f"{base}/results.json"):
            print(f"➖ {ds}/{scene}: chưa có results.json — chưa chạy, bỏ qua"); continue
        m = assert_scene_complete(ds, scene, it)  # fail-loud nếu dở dang
        ck = f"{base}/point_cloud/iteration_{it}"
        tt = f"{ck}/train_time_sec.txt"
        rows.append({"scene": scene, **m,
                     "n_gaussians": count_gaussians(f"{ck}/point_cloud.ply"),
                     "storage_mb": storage_mb(ck),
                     "fps": parse_fps(f"{base}/render_log.txt"),
                     "train_time_sec": round(float(open(tt).read()), 1) if os.path.exists(tt) else None,
                     "n_frames": expected_test_frames(ds, scene)})
        done.append(scene)
        sd = f"{report_dir}/{scene}"
        os.makedirs(f"{sd}/renders", exist_ok=True); os.makedirs(f"{sd}/gt", exist_ok=True)
        for f in sorted(glob.glob(f"{base}/test/ours_{it}/renders/*.png"))[:5]: shutil.copy(f, f"{sd}/renders")
        for f in sorted(glob.glob(f"{base}/test/ours_{it}/gt/*.png"))[:5]: shutil.copy(f, f"{sd}/gt")
        for f in (f"{base}/results.json", f"{base}/video/ours_{it}/video_rgb.mp4"):
            if os.path.exists(f): shutil.copy(f, sd)
    assert rows, f"❌ {ds}: 0 scene hợp lệ — chưa chạy scene nào, không xuất báo cáo."
    cols = ["scene"] + METRIC_KEYS + ["n_gaussians", "storage_mb", "fps", "train_time_sec", "n_frames"]
    with open(f"{report_dir}/summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    mean, text = summary_table(ds, rows, it)
    open(f"{report_dir}/summary.txt", "w", encoding="utf-8").write(text)
    info = {"dataset": ds, "iter": it, "scenes_done": done, "n_scenes": len(done),
            "gpu": _gpu_name(), "commit": _commit(),
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "frames_per_scene": {r["scene"]: r["n_frames"] for r in rows},
            "note": "torch 2.x/CUDA12 (Colab) — có thể lệch nhẹ so với paper (torch 1.13.1)"}
    json.dump(info, open(f"{report_dir}/run_info.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    shutil.make_archive(f"{ROOT}/{ds}_report", "zip", report_dir)
    download_file(f"{ROOT}/{ds}_report.zip")
    if DRIVE_REPORTS:
        dst = f"{DRIVE_REPORTS}/{ds}"; os.makedirs(dst, exist_ok=True)
        for fn in ("summary.csv", "summary.txt", "run_info.json"): shutil.copy(f"{report_dir}/{fn}", dst)
        shutil.copy(f"{ROOT}/{ds}_report.zip", dst)
        print(f"✅ {ds}: report -> {dst}")
    print(f"✅ {ds}: {len(done)}/{len(scenes)} scene vào báo cáo: {done}")
    return rows

def check_scene(ds, scenes, it):
    print(f"=== Trạng thái {ds} (iter {it}) ===")
    print(f"{'scene':14}{'train':>7}{'render':>8}{'metrics':>9}{'PSNR':>9}")
    for s in scenes:
        base = f"{ROOT}/4DGaussians/output/{ds}/{s}"
        ck = f"{base}/point_cloud/iteration_{it}"
        tr = all(os.path.exists(f"{ck}/{f}") for f in ("point_cloud.ply", "deformation.pth"))
        rd = len(glob.glob(f"{base}/test/ours_{it}/renders/*.png")) > 0
        psnr = None
        try: psnr = (json.load(open(f"{base}/results.json", encoding="utf-8")).get(f"ours_{it}") or {}).get("PSNR")
        except Exception: pass
        mt = isinstance(psnr, (int, float)) and math.isfinite(psnr)
        print(f"{s:14}{('✅' if tr else '❌'):>7}{('✅' if rd else '❌'):>8}{('✅' if mt else '❌'):>9}{(f'{psnr:.2f}' if mt else '-'):>9}")

print("✅ Helper sẵn sàng: assert_scene_complete, make_report, summary_table, check_scene, display_video")''')

# ============================================================ TEMPLATES (per-scene)
TRAIN_TMPL = r"""# 🏋️ Train <DS>/<S> (iter <IT>)
import os, time
%cd {ROOT}/4DGaussians
<ENV>_ck = f"{ROOT}/4DGaussians/output/<DS>/<S>/point_cloud/iteration_<IT>"
if all(os.path.exists(f"{_ck}/{_f}") for _f in ("point_cloud.ply", "deformation.pth")):
    print("⏭️ <S>: đã train xong (iteration_<IT>) — bỏ qua")
else:
    _t = time.time()
    !python train.py -s <DATA> --port 6017 --expname "<DS>/<S>" --configs <CFG><TRAINFLAGS>
    assert all(os.path.exists(f"{_ck}/{_f}") for _f in ("point_cloud.ply", "deformation.pth")), "❌ <S>: train KHÔNG ra checkpoint iteration_<IT> — xem log, DỪNG."
    open(f"{_ck}/train_time_sec.txt", "w").write(str(time.time() - _t))
    print(f"✅ <S>: train xong {time.time()-_t:.0f}s")"""

RENDER_TMPL = r"""# 🎬 Render <DS>/<S> (test) -> Drive; log FPS
import os, glob
%cd {ROOT}/4DGaussians
<ENV>_base = f"{ROOT}/4DGaussians/output/<DS>/<S>"
!python render.py --model_path "output/<DS>/<S>/" --skip_train --skip_video --configs <CFG> 2>&1 | tee {_base}/render_log.txt
_nr = len(glob.glob(f"{_base}/test/ours_<IT>/renders/*.png"))
_ng = len(glob.glob(f"{_base}/test/ours_<IT>/gt/*.png"))
assert _nr > 0 and _nr == _ng, f"❌ <S>: render lỗi (renders={_nr}, gt={_ng}) — DỪNG."
print(f"✅ <S>: render {_nr} frame")"""

VIDEO_TMPL = r"""# 📺 Video <DS>/<S> (chỉ hiện khi SHOW_MEDIA=True)
import glob, re
from IPython.display import display
_v = sorted(glob.glob(f"{ROOT}/4DGaussians/output/<DS>/<S>/video/ours_*/video_rgb.mp4"),
            key=lambda p: int(re.search(r"ours_(\d+)", p).group(1)))
if not _v:
    print("⚠️ <S>: chưa có video")
elif SHOW_MEDIA:
    display(display_video(_v[-1]))
else:
    print(f"📺 <S>: video tại {_v[-1]} (đặt SHOW_MEDIA=True để xem inline)")"""

METRICS_TMPL = r"""# 📊 Metrics + Báo cáo <DS>
import os, glob, json, math
%cd {ROOT}/4DGaussians
<ENV>_scenes = <SCENES>
for _s in _scenes:
    _b = f"{ROOT}/4DGaussians/output/<DS>/{_s}"
    if not glob.glob(f"{_b}/test/ours_*"):
        print(f"⚠️ {_s}: chưa render — bỏ qua metrics"); continue
    _done = False; _rp = f"{_b}/results.json"
    if os.path.exists(_rp):
        try: _done = math.isfinite((json.load(open(_rp, encoding="utf-8")).get("ours_<IT>") or {}).get("PSNR", float("nan")))
        except Exception: _done = False
    if _done:
        print(f"⏭️ {_s}: metrics xong")
    else:
        print(f"⏳ {_s}: đang tính metrics (có thể mất vài phút) ...")
        !python metrics.py --model_paths output/<DS>/{_s}/
make_report("<DS>", _scenes, <IT>)"""

STATUS_TMPL = r"""# 🔍 Trạng thái <DS>
check_scene("<DS>", <SCENES>, <IT>)"""

# ============================================================ DATA-PREP cells
DATAPREP_DNERF = r"""# 📦 Chuẩn bị data D-NeRF (8 scene, HuggingFace mirror)
import os, glob
_scenes = ["bouncingballs","hellwarrior","hook","jumpingjacks","lego","mutant","standup","trex"]
if all(os.path.isdir(f"{ROOT}/test/data/{s}") for s in _scenes):
    print("⏭️ D-NeRF: data đã có — bỏ qua tải")
else:
    os.makedirs(f"{ROOT}/test", exist_ok=True)
    %cd {ROOT}/test
    !wget -nc https://huggingface.co/camenduru/4DGaussians/resolve/main/data/data.zip
    !unzip -o -q data.zip
    _missing = [s for s in _scenes if not os.path.isdir(f"{ROOT}/test/data/{s}")]
    assert not _missing, f"❌ D-NeRF thiếu scene: {_missing}"
    if os.path.exists(f"{ROOT}/test/data.zip"): os.remove(f"{ROOT}/test/data.zip")  # tiết kiệm disk
    print(f"✅ D-NeRF: đủ 8 scene tại {ROOT}/test/data")
%cd {ROOT}/4DGaussians"""

DATAPREP_HYPER = r"""# 📦 Chuẩn bị data HyperNeRF (4 scene vrig) — zip + rename chuẩn + PLY pregenerated (bỏ COLMAP)
import os, shutil, glob, zipfile
%cd {ROOT}/4DGaussians
os.makedirs("data/hypernerf/virg", exist_ok=True)
PRUNE_RGB = False  # True = xóa rgb variant thừa (chỉ giữ 2x) nếu hết disk
# scene id chuẩn -> (asset zip, folder trong zip, folder PLY trong hypernerf-pcd.zip/virg)  [3 nguồn 3 convention]
HYPERNERF = {
    "3dprinter": ("vrig_3dprinter.zip",   "vrig-3dprinter",   "virg-3dprinter"),
    "broom2":    ("vrig_broom.zip",       "broom2",           "broom2"),
    "chicken":   ("vrig_chicken.zip",     "vrig-chicken",     "virg-chickchicken"),
    "banana":    ("vrig_peel-banana.zip", "vrig-peel-banana", "peel-banana"),
}
BASE = "https://github.com/google/hypernerf/releases/download/v0.1"
for sid, (asset, infolder, _) in HYPERNERF.items():
    dst = f"data/hypernerf/virg/{sid}"
    if os.path.isdir(dst) and glob.glob(f"{dst}/rgb/2x/*.png"):
        print(f"⏭️ {sid}: data đã có — bỏ qua"); continue
    !wget -nc -q {BASE}/{asset}
    assert os.path.exists(asset), f"❌ tải {asset} thất bại"
    with zipfile.ZipFile(asset) as z: z.extractall("data/hypernerf/virg")
    src = f"data/hypernerf/virg/{infolder}"
    assert os.path.isdir(src), f"❌ {asset}: không thấy folder {infolder} sau giải nén"
    if src != dst:
        if os.path.exists(dst): shutil.rmtree(dst)
        os.rename(src, dst)
    if PRUNE_RGB:
        for v in glob.glob(f"{dst}/rgb/*"):
            if os.path.basename(v) != "2x": shutil.rmtree(v, ignore_errors=True)
    os.remove(asset)  # xóa zip tiết kiệm disk
    print(f"✅ {sid}: {len(glob.glob(f'{dst}/rgb/2x/*.png'))} ảnh rgb/2x")
# PLY pregenerated (COLMAP) — bỏ chạy COLMAP
import gdown
_PCD_ID = "1fUHiSgimVjVQZ2OOzTFtz02E9EqCoWr5"
if not all(os.path.exists(f"data/hypernerf/virg/{sid}/points3D_downsample2.ply") for sid in HYPERNERF):
    if not os.path.exists("hypernerf-pcd.zip"):
        try:
            gdown.download(id=_PCD_ID, output="hypernerf-pcd.zip", quiet=False)
        except Exception as e:
            raise RuntimeError(f"❌ Tải PLY pregenerated lỗi ({e}). Tải tay: https://drive.google.com/file/d/{_PCD_ID}/view -> đặt 'hypernerf-pcd.zip' tại {ROOT}/4DGaussians/")
    assert os.path.exists("hypernerf-pcd.zip"), f"❌ thiếu hypernerf-pcd.zip — tải tay https://drive.google.com/file/d/{_PCD_ID}/view"
    with zipfile.ZipFile("hypernerf-pcd.zip") as z: z.extractall("_pcd_tmp")
    for sid, (_, _, plyfolder) in HYPERNERF.items():
        srcply = f"_pcd_tmp/hypernerf/virg/{plyfolder}/points3D_downsample2.ply"
        assert os.path.exists(srcply), f"❌ thiếu PLY pregenerated cho {sid}: {srcply}"
        shutil.copy(srcply, f"data/hypernerf/virg/{sid}/points3D_downsample2.ply")
    shutil.rmtree("_pcd_tmp", ignore_errors=True)
print("✅ HyperNeRF: 4 scene + points3D_downsample2.ply sẵn sàng (ratio 0.5 / rgb/2x mặc định)")"""

DATAPREP_NERFDS = r'''# 📦 Chuẩn bị data NeRF-DS (7 scene) — zip + rename + npy->ply. Ratio 1.0 set qua HYPER_RATIO (KHÔNG sed).
import os, shutil, glob
%cd {ROOT}/4DGaussians
os.makedirs("data/nerfds", exist_ok=True)
_expected = ["as","basin","bell","cup","plate","press","sieve"]
if all(os.path.isdir(f"data/nerfds/{s}") and os.path.exists(f"data/nerfds/{s}/points3D_downsample2.ply") for s in _expected):
    print("⏭️ NeRF-DS: data + ply đã có — bỏ qua tải"); _scenes = _expected
else:
    %cd {ROOT}/4DGaussians/data/nerfds
    !wget -nc -q https://github.com/JokerYan/NeRF-DS/releases/download/v0.1-pre-release/NeRF-DS.dataset.zip
    assert os.path.exists("NeRF-DS.dataset.zip"), "❌ tải NeRF-DS.dataset.zip thất bại"
    !unzip -n -q NeRF-DS.dataset.zip
    for d in list(os.listdir(".")):
        if os.path.isdir(d) and d.endswith("_novel_view"):
            short = d[:-len("_novel_view")]
            if os.path.exists(short): shutil.rmtree(short)
            os.rename(d, short)
    _scenes = sorted([d for d in os.listdir(".") if os.path.isdir(d)])
    print(f"NeRF-DS scene ({len(_scenes)}): {_scenes}")
    import numpy as np
    from plyfile import PlyData, PlyElement
    def _npy_to_ply(scene_dir):
        pts = np.load(f"{scene_dir}/points.npy").astype(np.float32); n = pts.shape[0]
        data = np.zeros(n, dtype=[("x","f4"),("y","f4"),("z","f4"),("nx","f4"),("ny","f4"),("nz","f4"),("red","u1"),("green","u1"),("blue","u1")])
        data["x"], data["y"], data["z"] = pts[:,0], pts[:,1], pts[:,2]
        data["red"] = data["green"] = data["blue"] = 128
        PlyData([PlyElement.describe(data, "vertex")]).write(f"{scene_dir}/points3D_downsample2.ply")
    for s in _scenes:
        if os.path.exists(f"{s}/points3D_downsample2.ply"): print(f"⏭️ {s}: ply đã có")
        elif os.path.exists(f"{s}/points.npy"): _npy_to_ply(s); print(f"✅ {s}: npy->ply")
        else: raise FileNotFoundError(f"❌ {s}: thiếu points.npy")
    if os.path.exists("NeRF-DS.dataset.zip"): os.remove("NeRF-DS.dataset.zip")  # tiết kiệm ~2.1GB
    %cd {ROOT}/4DGaussians
# verify rgb/1x = 480x270 (xác nhận ratio 1.0)
from PIL import Image
_chk = glob.glob(f"data/nerfds/{_scenes[0]}/rgb/1x/*.png")
assert _chk, f"❌ NeRF-DS {_scenes[0]}: thiếu rgb/1x"
_w, _h = Image.open(_chk[0]).size
print(f"✅ NeRF-DS rgb/1x = {_w}x{_h} — dùng HYPER_RATIO=1.0")
assert (_w, _h) == (480, 270), f"⚠️ resolution {_w}x{_h} != 480x270 — kiểm lại ratio!"'''

ENV_NERFDS = 'os.environ["HYPER_RATIO"] = "1.0"  # NeRF-DS native 480x270 -> rgb/1x\n'

DNERF = [
    "bouncingballs",
    "hellwarrior",
    "hook",
    "jumpingjacks",
    "lego",
    "mutant",
    "standup",
    "trex",
]
HYPER = ["3dprinter", "broom2", "chicken", "banana"]
NERFDS = ["as", "basin", "bell", "cup", "plate", "press", "sieve"]

# Hiển thị: scene id -> (emoji, tên đẹp)
DISPLAY = {
    "bouncingballs": ("⚽", "Bouncing Balls"),
    "hellwarrior": ("⚔️", "Hell Warrior"),
    "hook": ("🎣", "Hook"),
    "jumpingjacks": ("🤸", "Jumping Jacks"),
    "lego": ("🧱", "Lego"),
    "mutant": ("🧬", "Mutant"),
    "standup": ("🧍", "Standup"),
    "trex": ("🦖", "T-Rex"),
    "3dprinter": ("🖨️", "3D Printer"),
    "broom2": ("🧹", "Broom"),
    "chicken": ("🐔", "Chicken"),
    "banana": ("🍌", "Peel Banana"),
    "as": ("✨", "AS"),
    "basin": ("🥣", "Basin"),
    "bell": ("🔔", "Bell"),
    "cup": ("☕", "Cup"),
    "plate": ("🍽️", "Plate"),
    "press": ("🗜️", "Press"),
    "sieve": ("🧺", "Sieve"),
}

PARTS = [
    {
        "ds": "dnerf",
        "scenes": DNERF,
        "it": 20000,
        "data": "{ROOT}/test/data/<S>",
        "cfg": "arguments/dnerf/<S>.py",
        "env": "",
        "train_flags": "",  # D-NeRF synthetic nhẹ -> dùng dnerf-branch (deepcopy stack) như tác giả
        "prep": DATAPREP_DNERF,
        "phan": 2,
        "icon": "🧊",
        "name": "D-NeRF (synthetic, 8 scene, iter 20000)",
        "intro": "Dataset **synthetic** của D-NeRF (Pumarola, CVPR 2021) — paper 4DGS dùng để đánh giá. **Metric nhấn:** PSNR / SSIM / LPIPS. **Số công bố (4DGS, trung bình):** PSNR 34.05 · SSIM 0.98 · LPIPS 0.02.",
    },
    {
        "ds": "hypernerf",
        "scenes": HYPER,
        "it": 14000,
        "data": "data/hypernerf/virg/<S>",
        "cfg": "arguments/hypernerf/<S>.py",
        "env": "",
        "train_flags": " --dataloader",  # real, nhiều ảnh -> DataLoader lazy tránh OOM host RAM (train.py:93-96 deepcopy cả stack khi tắt)
        "prep": DATAPREP_HYPER,
        "phan": 3,
        "icon": "🎥",
        "name": "HyperNeRF (real, 4 scene, iter 14000)",
        "intro": "Dataset **thực tế** của HyperNeRF (Park, 2021) — paper 4DGS dùng. 4 scene quay đa góc (vrig). **Metric nhấn:** PSNR / MS-SSIM. **Số công bố (4DGS, trung bình):** PSNR 25.2 · MS-SSIM 0.845.",
    },
    {
        "ds": "nerfds",
        "scenes": NERFDS,
        "it": 14000,
        "data": "data/nerfds/<S>",
        "cfg": "arguments/hypernerf/default.py",
        "env": ENV_NERFDS,
        "train_flags": " --dataloader",  # NeRF-DS cũng dùng hypernerf/default.py + nhiều ảnh real -> cùng lý do HyperNeRF
        "prep": DATAPREP_NERFDS,
        "phan": 4,
        "icon": "✨",
        "name": "NeRF-DS (real, specular — dataset KHÁC, 7 scene, iter 14000)",
        "intro": "Dataset **cảnh phản chiếu động** (NeRF-DS, Yan, CVPR 2023) — **ngoài paper 4DGS**, đây là phần *áp dụng dataset khác*. Độ phân giải 480×270, 7 scene. **Metric nhấn:** PSNR / SSIM / LPIPS. **Số đối chiếu (4DGS trên NeRF-DS, trích từ SpectroMotion):** PSNR 22.79 · SSIM 0.8235 · LPIPS-vgg 0.2115.",
    },
]


def apply(tmpl, ds, scene, it, data, cfg, env, scenes=None, train_flags=""):
    t = (
        tmpl.replace("<ENV>", env)
        .replace("<DS>", ds)
        .replace("<S>", scene)
        .replace("<IT>", str(it))
        .replace("<DATA>", data)
        .replace("<CFG>", cfg)
        .replace(
            "<TRAINFLAGS>", train_flags
        )  # chỉ TRAIN_TMPL có placeholder; template khác no-op
    )
    if scenes is not None:
        t = t.replace("<SCENES>", repr(scenes))
    return t


for part in PARTS:
    ds, it, env, P = part["ds"], part["it"], part["env"], part["phan"]
    _rows = "\n".join(
        f"| {DISPLAY[s][0]} | `{s}` | {DISPLAY[s][1]} |" for s in part["scenes"]
    )
    md(
        f'---\n\n<a name="phan-{P}"></a>\n## {part["icon"]} PHẦN {P} — {part["name"]}\n\n'
        f"{part['intro']}\n\n| | scene | tên |\n|:---:|---|---|\n{_rows}"
    )
    sub(f"{P}.1", "📦", "Chuẩn bị dữ liệu")
    code(part["prep"])
    sub(f"{P}.2", "🏋️", "Huấn luyện · Render · Video")
    for i, scene in enumerate(part["scenes"], 1):
        emoji, name = DISPLAY[scene]
        data = part["data"].replace("<S>", scene)
        cfg = part["cfg"].replace("<S>", scene)
        subsub(f"{P}.2.{i}", emoji, f"{name} — `{scene}`")
        code(
            apply(
                TRAIN_TMPL,
                ds,
                scene,
                it,
                data,
                cfg,
                env,
                train_flags=part["train_flags"],
            )
        )
        code(apply(RENDER_TMPL, ds, scene, it, data, cfg, env))
        code(apply(VIDEO_TMPL, ds, scene, it, data, cfg, env))
    sub(f"{P}.3", "📊", "Metrics + Báo cáo")
    code(apply(METRICS_TMPL, ds, "", it, "", "", env, scenes=part["scenes"]))
    sub(f"{P}.4", "🔍", "Trạng thái")
    code(apply(STATUS_TMPL, ds, "", it, "", "", "", scenes=part["scenes"]))

# ============================================================ PHẦN 5 — TỔNG HỢP
md(r"""---

<a name="phan-5"></a>
## 📊 PHẦN 5 — Tổng hợp & đối sánh

Đọc `summary.csv` mọi dataset đã chạy (bỏ qua dataset chưa có, không lỗi). Sinh **2 trục đối sánh** + lưu CSV + biểu đồ vào Drive `figures/`.

> **Lưu ý khi đọc Trục 2:** trị tuyệt đối giữa các dataset khác scene / khác độ phân giải không phải là xếp hạng công bằng — nên đọc theo **xu hướng**; FPS / dung lượng / số Gaussian so sánh trực tiếp được hơn.""")

sub("5.1", "📥", "Đọc kết quả mọi dataset")
code(r'''# === 5.1 Đọc summary mọi dataset ===
import os, csv
DATASETS = ["dnerf", "hypernerf", "nerfds"]
def _summary_path(ds):
    return f"{DRIVE_REPORTS}/{ds}/summary.csv" if DRIVE_REPORTS else f"{ROOT}/report_{ds}/summary.csv"
LOADED = {}
for ds in DATASETS:
    p = _summary_path(ds)
    if not os.path.exists(p):
        print(f"➖ {ds}: chưa có summary — bỏ qua"); continue
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    for r in rows:
        for k, v in list(r.items()):
            if k != "scene":
                r[k] = float(v) if v not in ("", "None", "N/A") else None
    LOADED[ds] = rows
    print(f"✅ Đọc {ds}: {len(rows)} scene")
assert LOADED, "❌ Chưa dataset nào có summary.csv — chạy ít nhất 1 dataset (PHẦN 2/3/4) trước."

def _mean(rows, k):
    vals = [r[k] for r in rows if r.get(k) is not None]
    return sum(vals) / len(vals) if vals else None
def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "N/A"''')

sub("5.2", "🎯", "Trục 1 — Tái hiện vs Công bố")
code(r"""# === 5.2 Trục 1 — Của tôi vs CÔNG BỐ (validate tái hiện) ===
import csv
out = []
print(f"{'dataset':12}{'metric':11}{'của tôi':>11}{'paper':>11}{'Δ':>11}")
print("-" * 56)
for ds, rows in LOADED.items():
    for k in HIGHLIGHT[ds]:  # quality: so trực tiếp
        mine = _mean(rows, k); ref = PAPER_REF[ds].get(k)
        d = (mine - ref) if (mine is not None and ref is not None) else None
        print(f"{ds:12}{k:11}{_fmt(mine):>11}{_fmt(ref):>11}{(f'{d:+.4f}' if d is not None else 'N/A'):>11}")
        out.append({"dataset": ds, "metric": k,
                    "mine": round(mine, 4) if mine is not None else None,
                    "paper": ref, "delta": round(d, 4) if d is not None else None})
    for k in PAPER_EFF.get(ds, {}):  # hiệu năng: Train/FPS/Storage (FPS/Train tuỳ GPU)
        mine = _mean(rows, k); ref = PAPER_EFF[ds].get(k)
        d = (mine - ref) if (mine is not None and ref is not None) else None
        print(f"{ds:12}{k:11}{_fmt(mine,1):>11}{_fmt(ref,1):>11}{(f'{d:+.1f}' if d is not None else 'N/A'):>11}")
        out.append({"dataset": ds, "metric": k,
                    "mine": round(mine, 1) if mine is not None else None,
                    "paper": ref, "delta": round(d, 1) if d is not None else None})
_p = f"{DRIVE_REPORTS}/comparison_repro_vs_paper.csv" if DRIVE_REPORTS else f"{ROOT}/comparison_repro_vs_paper.csv"
with open(_p, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["dataset", "metric", "mine", "paper", "delta"]); w.writeheader(); w.writerows(out)
print(f"\n✅ Lưu {_p}")
print("Lưu ý: Train/FPS tuỳ GPU (Colab ≠ paper) -> đọc xu hướng; Storage so trực tiếp. NeRF-DS không có hiệu năng paper.")""")

sub("5.3", "🔀", "Trục 2 — So sánh giữa các dataset")
code(r"""# === 5.3 Trục 2 — Cross-dataset (NeRF-DS vs HyperNeRF/D-NeRF) ===
import csv
COMMON = ["PSNR", "SSIM", "LPIPS-vgg"]
EXTRA = ["fps", "storage_mb", "train_time_sec", "n_gaussians"]
TYPE = {"dnerf": "synthetic", "hypernerf": "real", "nerfds": "real-specular"}
out = []
hdr = f"{'dataset':12}" + "".join(f"{k:>11}" for k in COMMON) + f"{'FPS':>8}{'MB':>9}{'Train(s)':>10}{'#Gauss':>11}  type"
print(hdr); print("-" * len(hdr))
for ds, rows in LOADED.items():
    rec = {"dataset": ds, "type": TYPE[ds]}
    line = f"{ds:12}" + "".join(f"{_fmt(_mean(rows, k)):>11}" for k in COMMON)
    for k in COMMON: rec[k] = round(_mean(rows, k), 4) if _mean(rows, k) is not None else None
    for k in EXTRA: rec[k] = round(_mean(rows, k), 2) if _mean(rows, k) is not None else None
    line += f"{_fmt(rec['fps'],2):>8}{_fmt(rec['storage_mb'],1):>9}{_fmt(rec['train_time_sec'],1):>10}{(str(int(rec['n_gaussians'])) if rec['n_gaussians'] else 'N/A'):>11}  {rec['type']}"
    print(line)
    out.append(rec)
_p = f"{DRIVE_REPORTS}/comparison_cross_dataset.csv" if DRIVE_REPORTS else f"{ROOT}/comparison_cross_dataset.csv"
with open(_p, "w", newline="", encoding="utf-8") as f:
    cols = ["dataset", "type"] + COMMON + EXTRA
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(out)
print(f"\n✅ Lưu {_p}")
print("\nNhận xét: NeRF-DS (real-specular) thường PSNR/SSIM thấp hơn + LPIPS cao hơn so HyperNeRF (real) -> 4DGS hạn chế với vật phản chiếu động. D-NeRF (synthetic) đọc tách biệt (mốc dễ).")""")

sub("5.4", "📈", "Biểu đồ")
code(r"""# === 5.4 Biểu đồ (lưu figures/, hiện inline nếu SHOW_MEDIA) ===
import matplotlib
if not SHOW_MEDIA: matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os, glob
FIG = DRIVE_FIGS if DRIVE_FIGS else f"{ROOT}/figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.family": "DejaVu Sans"})

def _save(fig, name):
    p = f"{FIG}/{name}"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")  # bản vector cho LaTeX
    print(f"✅ Lưu {p[:-4]} (.png + .pdf)")
    plt.show() if SHOW_MEDIA else plt.close(fig)

# --- Biểu đồ 1: bar_repro_vs_paper (subplot mỗi metric, mine vs paper) ---
metrics1 = ["PSNR", "SSIM", "MS-SSIM", "LPIPS-vgg"]
avail = [m for m in metrics1 if any(m in HIGHLIGHT[ds] and m in PAPER_REF[ds] for ds in LOADED)]
if avail:
    fig, axes = plt.subplots(1, len(avail), figsize=(4.5*len(avail), 4))
    if len(avail) == 1: axes = [axes]
    for ax, m in zip(axes, avail):
        dss = [ds for ds in LOADED if m in HIGHLIGHT[ds] and m in PAPER_REF[ds]]
        mine = [_mean(LOADED[ds], m) for ds in dss]
        paper = [PAPER_REF[ds][m] for ds in dss]
        x = np.arange(len(dss)); w = 0.38
        b1 = ax.bar(x - w/2, mine, w, label="Nhóm", color="#1f77b4")
        b2 = ax.bar(x + w/2, paper, w, label="Bài báo", color="#ff7f0e")
        ax.bar_label(b1, fmt="%.2f", fontsize=8); ax.bar_label(b2, fmt="%.2f", fontsize=8)
        ax.set_title(m); ax.set_xticks(x); ax.set_xticklabels(dss, rotation=15); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Trục 1 — Tái hiện vs Công bố"); fig.tight_layout()
    _save(fig, "bar_repro_vs_paper.png")

# --- Biểu đồ 2: bar_cross_dataset (subplot mỗi metric, các dataset cạnh nhau) ---
metrics2 = ["PSNR", "SSIM", "LPIPS-vgg"]
fig, axes = plt.subplots(1, len(metrics2), figsize=(4.5*len(metrics2), 4))
COLORS = {"dnerf": "#9467bd", "hypernerf": "#2ca02c", "nerfds": "#d62728"}
for ax, m in zip(axes, metrics2):
    dss = list(LOADED.keys()); vals = [_mean(LOADED[ds], m) for ds in dss]
    b = ax.bar(dss, [v if v is not None else 0 for v in vals], color=[COLORS[d] for d in dss])
    ax.bar_label(b, fmt="%.3f", fontsize=8)
    ax.set_title(m); ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=0.3)
fig.suptitle("Trục 2 — 4DGS cross-dataset (NeRF-DS vs HyperNeRF/D-NeRF)"); fig.tight_layout()
_save(fig, "bar_cross_dataset.png")

# --- Biểu đồ 3: scatter FPS vs #Gaussians (mỗi điểm 1 scene) ---
fig, ax = plt.subplots(figsize=(7, 5))
for ds, rows in LOADED.items():
    xs = [r["n_gaussians"] for r in rows if r.get("n_gaussians") and r.get("fps")]
    ys = [r["fps"] for r in rows if r.get("n_gaussians") and r.get("fps")]
    ns = [r["scene"] for r in rows if r.get("n_gaussians") and r.get("fps")]
    ax.scatter(xs, ys, label=ds, color=COLORS[ds], s=60)
    for x, y, n in zip(xs, ys, ns): ax.annotate(n, (x, y), fontsize=7, alpha=0.7)
ax.set_xlabel("#Gaussians"); ax.set_ylabel("FPS"); ax.set_title("FPS vs #Gaussians (mỗi điểm = 1 scene)")
ax.grid(alpha=0.3); ax.legend()
_save(fig, "scatter_fps_gaussians.png")""")

sub("5.5", "📉", "Đường hội tụ PSNR")
code(r'''# === 5.5 Đường hội tụ PSNR theo iteration (đọc tensorboard events) ===
import matplotlib
if not SHOW_MEDIA: matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob, os
FIG = DRIVE_FIGS if DRIVE_FIGS else f"{ROOT}/figures"
os.makedirs(FIG, exist_ok=True)
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as e:
    EventAccumulator = None
    print(f"➖ Bỏ qua hội tụ PSNR: thiếu tensorboard ({e})")

def _psnr_curve(scene_dir):
    """[(iter, psnr)] của fine/test từ tensorboard events; [] nếu không có."""
    if not glob.glob(f"{scene_dir}/events.out.tfevents.*"): return []
    acc = EventAccumulator(scene_dir, size_guidance={"scalars": 0}); acc.Reload()
    tag = next((t for t in acc.Tags().get("scalars", []) if t.startswith("fine/test") and "psnr" in t.lower()), None)
    return [(s.step, s.value) for s in acc.Scalars(tag)] if tag else []

if EventAccumulator:
    for ds, rows in LOADED.items():
        fig, ax = plt.subplots(figsize=(7, 5)); ok = False
        for r in rows:
            cur = _psnr_curve(f"{ROOT}/4DGaussians/output/{ds}/{r['scene']}")
            if cur:
                xs, ys = zip(*cur); ax.plot(xs, ys, marker="o", label=r["scene"]); ok = True
        if not ok:
            plt.close(fig); print(f"➖ {ds}: chưa có scalar PSNR trong tensorboard"); continue
        ax.set_xlabel("Iteration (fine)"); ax.set_ylabel("Test PSNR (dB)")
        ax.set_title(f"Hội tụ PSNR — {ds}"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        # Ép tick đúng 3 mốc eval (3000 = chuyển coarse sang fine, rồi 7000/14000) cho khớp
        # caption; tránh matplotlib tự đặt tick tròn 2000 khiến điểm 3000 trông như ở 4000.
        ax.set_xticks([3000, 7000, 14000])
        p = f"{FIG}/convergence_{ds}.png"; fig.savefig(p, bbox_inches="tight"); fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight"); print(f"✅ Lưu {p[:-4]} (.png + .pdf)")
        plt.show() if SHOW_MEDIA else plt.close(fig)
print("ℹ️ Đường hội tụ chỉ có điểm ở iter eval (3000/7000/14000) theo test_iterations.")''')

sub("5.6", "🔬", "Số bổ trợ cho báo cáo (coarse/train PSNR, init points)")
code(r"""# === 5.6.1 PSNR hellwarrior theo giai đoạn (coarse/fine) từ tensorboard -> reports/aux_stats.json ===
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, json, os
REP = DRIVE_REPORTS if DRIVE_REPORTS else f"{ROOT}/reports"
os.makedirs(REP, exist_ok=True)
_auxp = f"{REP}/aux_stats.json"
AUX = json.load(open(_auxp, encoding="utf-8")) if os.path.exists(_auxp) else {}
evs = glob.glob(f"{ROOT}/4DGaussians/output/dnerf/hellwarrior/**/events.out.tfevents.*", recursive=True)
assert evs, "❌ không thấy tfevents hellwarrior (cần Drive output còn)"
ea = EventAccumulator(evs[0]); ea.Reload()
hw = {}
for tag in ea.Tags()["scalars"]:
    if "psnr" in tag.lower():
        pts = [(p.step, round(p.value, 2)) for p in ea.Scalars(tag)]
        print(tag, "->", pts); hw[tag] = pts
AUX["hellwarrior_psnr"] = hw
json.dump(AUX, open(_auxp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"✅ Lưu {_auxp}")""")
code(r"""# === 5.6.2 Train PSNR + số điểm khởi tạo sieve/plate -> reports/aux_stats.json ===
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, json, os
from plyfile import PlyData
REP = DRIVE_REPORTS if DRIVE_REPORTS else f"{ROOT}/reports"
os.makedirs(REP, exist_ok=True)
_auxp = f"{REP}/aux_stats.json"
AUX = json.load(open(_auxp, encoding="utf-8")) if os.path.exists(_auxp) else {}
for s in ["sieve", "plate"]:
    rec = {}
    ev = glob.glob(f"{ROOT}/4DGaussians/output/nerfds/{s}/**/events.out.tfevents.*", recursive=True)
    if ev:
        ea = EventAccumulator(ev[0]); ea.Reload()
        for tag in ea.Tags()["scalars"]:
            if "train" in tag.lower() and "psnr" in tag.lower():
                pts = [(p.step, round(p.value, 2)) for p in ea.Scalars(tag)]
                print(s, "TRAIN", pts); rec["train_psnr"] = pts
    ply = glob.glob(f"{ROOT}/4DGaussians/data/nerfds/{s}/points3D_downsample2.ply")
    rec["init_points"] = len(PlyData.read(ply[0])["vertex"]) if ply else None
    print(s, "init points:", rec["init_points"])
    AUX.setdefault("nerfds_aux", {})[s] = rec
json.dump(AUX, open(_auxp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"✅ Lưu {_auxp}")""")

sub("5.7", "🖼️", "Ảnh render vs GT")
code(r"""# === 5.7 Montage render|GT (lưu figures/montage_<ds>.png) ===
import matplotlib
if not SHOW_MEDIA: matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import glob, os
FIG = DRIVE_FIGS if DRIVE_FIGS else f"{ROOT}/figures"
ITER = {"dnerf": 20000, "hypernerf": 14000, "nerfds": 14000}
for ds, rows in LOADED.items():
    it = ITER[ds]; scenes = [r["scene"] for r in rows]   # tất cả scene của dataset
    pairs = []
    for s in scenes:
        rg = sorted(glob.glob(f"{ROOT}/4DGaussians/output/{ds}/{s}/test/ours_{it}/renders/*.png"))
        gt = sorted(glob.glob(f"{ROOT}/4DGaussians/output/{ds}/{s}/test/ours_{it}/gt/*.png"))
        if rg and gt: pairs.append((s, rg[0], gt[0]))
    if not pairs:
        print(f"➖ {ds}: chưa có ảnh render/gt cho montage"); continue
    # layout ngang: render hàng trên, GT hàng dưới, mỗi scene 1 cột -> vừa khổ trang
    n = len(pairs)
    fig, axes = plt.subplots(2, n, figsize=(2.4*n, 5.2))
    if n == 1: axes = axes.reshape(2, 1)
    for i, (s, rp, gp) in enumerate(pairs):
        axes[0, i].imshow(Image.open(rp)); axes[0, i].set_title(s, fontsize=9)
        axes[1, i].imshow(Image.open(gp))
        for r in (0, 1):
            axes[r, i].set_xticks([]); axes[r, i].set_yticks([])
            for sp in axes[r, i].spines.values(): sp.set_visible(False)
    axes[0, 0].set_ylabel("render", fontsize=11); axes[1, 0].set_ylabel("GT", fontsize=11)
    fig.suptitle(f"Montage {ds} (render vs GT)"); fig.tight_layout()
    p = f"{FIG}/montage_{ds}.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight"); print(f"✅ Lưu {p[:-4]} (.png + .pdf)")
    plt.show() if SHOW_MEDIA else plt.close(fig)""")

sub("5.8", "🖼️", "Ảnh render/GT cảnh riêng cho báo cáo")
code(r"""# === 5.8 Trích ảnh render/GT cảnh riêng (mutant/lego/sieve/plate) cho báo cáo ===
from PIL import Image
import glob
FIG = DRIVE_FIGS if DRIVE_FIGS else f"{ROOT}/figures"
WANT = {"dnerf": (20000, ["mutant", "lego"]), "nerfds": (14000, ["sieve", "plate"])}
for ds, (it, scenes) in WANT.items():
    for s in scenes:
        for kind, tag in (("renders", "render"), ("gt", "gt")):
            src = sorted(glob.glob(f"{ROOT}/4DGaussians/output/{ds}/{s}/test/ours_{it}/{kind}/*.png"))
            assert src, f"❌ thiếu {ds}/{s}/{kind}"
            img = Image.open(src[len(src) // 2]).convert("RGB")   # frame giữa, đại diện
            img.save(f"{FIG}/{s}_{tag}.png")
            img.save(f"{FIG}/{s}_{tag}.pdf", "PDF", resolution=150.0)
            print(f"✅ Lưu {FIG}/{s}_{tag} (.png + .pdf)")""")

sub("5.9", "📝", "Tổng kết run")
code(r"""# === 5.9 RUN_SUMMARY.md (tổng run, lưu reports/) ===
import os, json, datetime, subprocess
REP = DRIVE_REPORTS if DRIVE_REPORTS else f"{ROOT}"
lines = ["# RUN SUMMARY — 4DGS report", ""]
lines.append(f"- Ngày: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
try: _gpu = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
except Exception: _gpu = "unknown"
try: _cm = subprocess.run(["git","-C",f"{ROOT}/4DGaussians","rev-parse","--short","HEAD"], capture_output=True, text=True, timeout=20).stdout.strip()
except Exception: _cm = "unknown"
lines += [f"- GPU: {_gpu}", f"- Fork commit: {_cm}", f"- Datasets đã chạy: {list(LOADED.keys())}", ""]
lines.append("## Headline (mean)")
lines.append("| dataset | #scene | PSNR | SSIM | MS-SSIM | LPIPS-vgg |")
lines.append("|---|---|---|---|---|---|")
for ds, rows in LOADED.items():
    lines.append(f"| {ds} | {len(rows)} | {_fmt(_mean(rows,'PSNR'),2)} | {_fmt(_mean(rows,'SSIM'),3)} | {_fmt(_mean(rows,'MS-SSIM'),3)} | {_fmt(_mean(rows,'LPIPS-vgg'),3)} |")
lines += ["", "_Lưu ý: chạy trên torch 2.x / CUDA 12 (Colab) — số có thể lệch nhẹ so với paper (torch 1.13.1)._"]
os.makedirs(REP, exist_ok=True)
open(f"{REP}/RUN_SUMMARY.md", "w", encoding="utf-8").write("\n".join(lines))
print(f"✅ {REP}/RUN_SUMMARY.md")
print("\n".join(lines))""")

# ============================================================ PHẦN 6 — THAM KHẢO
md(r"""---

<a name="phan-6"></a>
## 📚 PHẦN 6 — Tham khảo

**Papers**
- Wu et al. *4D Gaussian Splatting for Real-Time Dynamic Scene Rendering.* CVPR 2024.
- Kerbl et al. *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* SIGGRAPH 2023.
- Pumarola et al. *D-NeRF: Neural Radiance Fields for Dynamic Scenes.* CVPR 2021.
- Park et al. *HyperNeRF: A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields.* SIGGRAPH Asia 2021.
- Yan et al. *NeRF-DS: Neural Radiance Fields for Dynamic Specular Objects.* CVPR 2023.
- *SpectroMotion: Dynamic 3D Reconstruction of Specular Scenes* (nguồn trích số 4DGS-on-NeRF-DS).

**Code & data**
- Code 4DGS: [hustvl/4DGaussians](https://github.com/hustvl/4DGaussians) (bản gốc) · [MinhThang1009/4D-Gaussian](https://github.com/MinhThang1009/4D-Gaussian) (bản dùng trong báo cáo, chỉnh cho chạy được trên Colab).
- D-NeRF data: HuggingFace mirror `camenduru/4DGaussians`. HyperNeRF data: [google/hypernerf v0.1](https://github.com/google/hypernerf/releases/tag/v0.1) + PLY pregenerated (4DGS). NeRF-DS data: [JokerYan/NeRF-DS v0.1](https://github.com/JokerYan/NeRF-DS/releases).

**Code-map 4DGS:** `scene/deformation.py` (HexPlane), `scene/hexplane.py`, `scene/gaussian_model.py`, `submodules/depth-diff-gaussian-rasterization` (rasterizer), `train.py` / `render.py` / `metrics.py`.""")

# ============================================================ ASSEMBLE + WRITE
nb_out = {
    "cells": cells,
    "metadata": nb_src.get(
        "metadata", {"kernelspec": {"display_name": "Python 3", "name": "python3"}}
    ),
    "nbformat": 4,
    "nbformat_minor": nb_src.get("nbformat_minor", 0),
}
with open(SRC, "w", encoding="utf-8") as f:
    json.dump(nb_out, f, ensure_ascii=False, indent=1)

print(f"✅ Ghi {SRC}: {len(cells)} cell (backup: {SRC}.bak)")
n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"   markdown={n_md}, code={n_code}")
