# Danh so lai cac Buoc cho nhat quan: 1b->2, 2->3, 3->4, 4->5, 5->6, 6->7
import ast
import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
nb = json.load(open("4DGaussians.ipynb", encoding="utf-8"))


def set_src(i, s):
    lines = s.split("\n")
    nb["cells"][i]["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]


def get(i):
    return "".join(nb["cells"][i]["source"])


BUMP = {6: 7, 5: 6, 4: 5, 3: 4, 2: 3}  # 0,1 giu nguyen


def bump_steps(src):
    """Doi 'Bước N' va 'buoc-N' theo BUMP (quet 1 lan, khong chong)."""
    src = re.sub(
        r"Bước (\d+)",
        lambda m: f"Bước {BUMP.get(int(m.group(1)), int(m.group(1)))}",
        src,
    )
    src = re.sub(
        r"buoc-(\d+)",
        lambda m: f"buoc-{BUMP.get(int(m.group(1)), int(m.group(1)))}",
        src,
    )
    return src


def bump_scene(src, old_n, new_n):
    """Doi scene 'old_n.M' -> 'new_n.M' (vd 3.1->4.1), tranh dung so thap phan khac."""
    return re.sub(rf"(?<![\d.]){old_n}\.(\d)", rf"{new_n}.\1", src)


# ============ CELL 0: viet lai muc luc ============
cell0 = get(0)
old_toc = """| # | Bước | Nội dung | Thời gian ước tính |
|:---:|------|---------|:---:|
| **0** | 🌐 Phát hiện môi trường | Đặt `ROOT` + `download_file()` theo Colab/Kaggle | ~1 giây |
| **1** | [🔧 Cài đặt môi trường](#buoc-1) | Clone repo, cài thư viện, vá lỗi build, build CUDA submodules | ~5 phút |
| **2** | [📦 Tải dataset D-NeRF](#buoc-2) | Tải 8 scene từ HuggingFace | ~3 phút |
| **3** | [🏋️ Huấn luyện & Render D-NeRF](#buoc-3) | Train + render 8 scene (bouncingballs → trex) | ~2 giờ |
| **4** | [📊 Tính metrics & xuất báo cáo](#buoc-4) | PSNR, SSIM, LPIPS — gom zip tải về | ~5 phút |
| **5** | [🔬 NeRF-DS](#buoc-5) | Vá reader, tải dataset, train/render 7 scene | ~4 giờ |
| **6** | [✅ Kiểm tra trạng thái](#buoc-6) | Bảng tổng kết: scene nào fail ở bước nào | ~5 giây |"""
new_toc = """| # | Bước | Nội dung | Thời gian ước tính |
|:---:|------|---------|:---:|
| **0** | 🌐 Phát hiện môi trường | Đặt `ROOT` + `download_file()` theo Colab/Kaggle | ~1 giây |
| **1** | [🔧 Cài đặt môi trường](#buoc-1) | Clone repo, cài thư viện, vá lỗi build, build CUDA submodules | ~5 phút |
| **2** | [🔗 Nối output vào Drive](#buoc-2) | Symlink `output/` → Drive để lưu bền qua các phiên | ~10 giây |
| **3** | [📦 Tải dataset D-NeRF](#buoc-3) | Tải 8 scene từ HuggingFace | ~3 phút |
| **4** | [🏋️ Huấn luyện & Render D-NeRF](#buoc-4) | Train + render 8 scene (bouncingballs → trex) | ~2 giờ |
| **5** | [📊 Tính metrics & xuất báo cáo](#buoc-5) | PSNR, SSIM, LPIPS — gom zip tải về | ~5 phút |
| **6** | [🔬 NeRF-DS](#buoc-6) | Vá reader, tải dataset, train/render 7 scene | ~4 giờ |
| **7** | [✅ Kiểm tra trạng thái](#buoc-7) | Bảng tổng kết: scene nào fail ở bước nào | ~5 giây |"""
assert cell0.count(old_toc) == 1, "TOC khong khop"
cell0 = cell0.replace(old_toc, new_toc)
cell0 = cell0.replace(
    "Chạy đúng thứ tự từ Bước 0 → 6", "Chạy đúng thứ tự từ Bước 0 → 7"
)
set_src(0, cell0)

# ============ CELL 3: Buoc 1b -> Buoc 2 + anchor + ref ============
cell3 = get(3)
assert cell3.count("Bước 1b") == 1
cell3 = cell3.replace(
    "### 🔗 Bước 1b — Nối", '<a name="buoc-2"></a>\n\n### 🔗 Bước 2 — Nối'
)
cell3 = cell3.replace(
    "chạy Bước 0+1 rồi cell này → 5.4 tự resume",
    "chạy Bước 0+1 rồi cell này → 6.4 tự resume",
)
set_src(3, cell3)

# ============ Cells header don + NeRF-DS header: bump Buoc/buoc ============
for i in [5, 7, 41, 43, 52]:
    set_src(i, bump_steps(get(i)))

# ============ Cells scene D-NeRF (3.x -> 4.x) ============
for i in [9, 13, 17, 21, 25, 29, 33, 37]:
    set_src(i, bump_scene(get(i), 3, 4))

# ============ Cells scene NeRF-DS (5.x -> 6.x) ============
for i in [44, 46, 48, 50]:
    set_src(i, bump_scene(get(i), 5, 6))

# ============ CELL 53: health-check — thay tuong minh ============
c = get(53)


def rep(old, new, n=1):
    global c
    assert c.count(old) == n, f"cell53: '{old[:45]}' count={c.count(old)} (mong {n})"
    c = c.replace(old, new)


# Cum liet ke (lam truoc vi chua so tran)
rep("(Bước 0, 1, 2, 5.1–5.3)", "(Bước 0, 1, 3, 6.1–6.3)")
rep("(Bước 3, 4, 5.4)", "(Bước 4, 5, 6.4)")
rep("(Bước 4 + 5.4)", "(Bước 5 + 6.4)")
rep('"Bước 4/5.4"', '"Bước 5/6.4"')
rep("# Bước 5.2 + 5.3:", "# Bước 6.2 + 6.3:")
# Token Buoc 5.x -> 6.x (nhieu lan)
rep("Bước 5.1", "Bước 6.1", n=c.count("Bước 5.1"))
rep("Bước 5.2", "Bước 6.2", n=c.count("Bước 5.2"))
rep("Bước 5.3", "Bước 6.3", n=c.count("Bước 5.3"))
rep("Bước 5.4", "Bước 6.4", n=c.count("Bước 5.4"))
# Bước 2 (data D-NeRF) -> 3
rep("Bước 2", "Bước 3", n=c.count("Bước 2"))
# cac ref "cell N.M" / "chay N.M" tran
rep("chạy cell 5.1", "chạy cell 6.1")
rep("chạy 5.4 (cell 5.4", "chạy 6.4 (cell 6.4")
rep("chưa chạy 5.3", "chưa chạy 6.3")
rep("chạy lại cell 5.3", "chạy lại cell 6.3")
rep("chạy lại 5.3 không fix", "chạy lại 6.3 không fix")
# Bước 4 (zip D-NeRF) -> 5, target cu the de khong dung "(Bước 4,5,6.4)"
rep("xóa rồi chạy lại Bước 4", "xóa rồi chạy lại Bước 5")
rep("hoặc hỏng (chạy Bước 4)", "hoặc hỏng (chạy Bước 5)")
set_src(53, c)

json.dump(
    nb, open("4DGaussians.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1
)

# ============ VERIFY ============
nb2 = json.load(open("4DGaussians.ipynb", encoding="utf-8"))
# Syntax 2 code cell co the bi dung
for i in [3, 53]:
    src = "".join(nb2["cells"][i]["source"])
    conv = "\n".join(
        (l[: len(l) - len(l.lstrip())] + "pass")
        if l.lstrip().startswith(("%", "!"))
        else l
        for l in src.split("\n")
    )
    ast.parse(conv)
print("Syntax cell 3 + 53 OK\n")
print("=== Mọi tham chiếu Bước/buoc/scene sau khi đổi ===")
import re as _re

for i, cc in enumerate(nb2["cells"]):
    s = "".join(cc["source"])
    hits = sorted(set(_re.findall(r"Bước \d[\d.]*|buoc-\d|### [^\n]*?\d\.\d", s)))
    if hits:
        print(f"[{i}] {hits}")
