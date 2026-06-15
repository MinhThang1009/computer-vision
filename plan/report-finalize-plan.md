# Plan hoàn tất báo cáo `report/main.tex` sau khi chạy xong Colab

> Mục tiêu: thay mọi `\TODO{?}` + số cũ bằng SỐ THẬT từ `reports/*/summary.csv`, copy ảnh, điền bìa, rồi compile. Số liệu phải khớp đúng run thực tế.

## ⚠️ Phương pháp đảm bảo ĐỦ (đọc TRƯỚC — quan trọng nhất)
Liệt kê tay KHÔNG bao giờ chắc đủ (đã chứng minh: mỗi lần rà đều tìm thêm số sót — ±0,25 dB, hellwarrior 15,2, densification 47k-62k, "34--44 FPS", "39,7", "≈18 MB"...). Cách DUY NHẤT chắc chắn:
1. Sau khi điền, liệt kê **MỌI token số** (KHÔNG lọc theo đơn vị — lọc theo đơn vị là lỗi cũ làm sót SSIM `0,9787`, số ĐẦU của cặp "X vs Y", và số đầu dải "X--Y"):
   ```
   grep -noE '[0-9][0-9.,]*' report/main.tex | sort -u
   ```
   Regex này bắt mọi số: `0,9787` (SSIM), `34` (đầu dải `34--44`), `22,79` (đầu cặp), `2.000`. Duyệt TỪNG số.
2. Mỗi số "nhóm" (mình đo) phải == giá trị trong `summary.csv` của dataset tương ứng. Số "paper"/"SpectroMotion" giữ nguyên (đã verify 2 lớp).
3. `grep -n 'TODO' report/main.tex` → chỉ còn dòng macro (73).
4. Sau compile: PDF không còn chữ đỏ `[?]` và không có `??` (ref hỏng).

→ **Danh sách A/B dưới chỉ là VÍ DỤ đã biết, KHÔNG đảm bảo đủ.** Bắt buộc làm bước grep trên.

## Trạng thái dữ liệu
- **D-NeRF: ✅ XONG 8/8** — số có sẵn ở Drive `reports/dnerf/summary.csv` (liệt kê sẵn ở mục A1 dưới).
- **HyperNeRF: ⏳ chưa chạy** (PHẦN 3 notebook) → khi xong đọc `reports/hypernerf/summary.csv`.
- **NeRF-DS: ⏳ chưa chạy** (PHẦN 4) → `reports/nerfds/summary.csv`.
- **PHẦN 5 tổng hợp: ⏳** chạy sau khi có ≥1 dataset → sinh `comparison_*.csv` + `figures/*.png` + `RUN_SUMMARY.md`.

---

## A. Điền số "nhóm" vào các bảng (nguồn = `reports/<ds>/summary.csv`)

### A1. `tab:reproduce` (D-NeRF) — ĐÃ CÓ SỐ, thay số run cũ ngay
Cột: PSNR nhóm | SSIM nhóm | FPS. (run mới, T4)

| scene | PSNR | SSIM | FPS |
|---|---|---|---|
| bouncingballs | 40,71 | 0,9943 | 38,2 |
| hellwarrior | 28,79 | 0,9735 | 33,4 |
| hook | 32,82 | 0,9758 | 38,5 |
| jumpingjacks | 35,33 | 0,9856 | 43,9 |
| lego | 25,04 | 0,9376 | 30,5 |
| mutant | 37,47 | 0,9879 | 42,1 |
| standup | 38,11 | 0,9898 | 41,9 |
| trex | 34,20 | 0,9849 | 38,2 |
| **Trung bình** | **34,06** | **0,9787** | **38,3** |

(Cột "Bài báo" giữ nguyên — đã verify đúng. Mean PSNR nhóm 34,06 ≈ paper 34,05.)

### A2. `tab:sota` (hàng "4D-GS (nhóm, Colab T4)") — ĐÃ CÓ SỐ
PSNR **34,06** · SSIM **0,98** · LPIPS **0,0262** · Time **~13,5 phút** · FPS **38,3** · Storage **~24,8 MB**.

### A3. `tab:hypernerf` (4 scene) — sau khi chạy PHẦN 3
Đọc `reports/hypernerf/summary.csv`, điền PSNR/MS-SSIM/FPS cho 3dprinter, broom2, chicken, banana + dòng "Trung bình (nhóm)". So mean với paper 25,2 / 0,845.

### A4. `tab:nerfds` (7 scene) — sau khi chạy PHẦN 4
Đọc `reports/nerfds/summary.csv`, điền **PSNR nhóm / SSIM nhóm / LPIPS nhóm / FPS** cho as, basin, bell, cup, plate, press, sieve + Trung bình. (Cột "PSNR SpectroMotion" đã điền sẵn + verify.)
> ⚠️ **plate** (18,33 / 0,7459 / FPS 45,9) và **sieve** (25,64 / 0,8608 / FPS 44,6) hiện là số RUN CŨ → **THAY bằng số run mới** (như D-NeRF), KHÔNG giữ. Đồng bộ luôn số văn xuôi trích từ 2 cảnh này (§3.5 + Ch4: ~dòng 771, 772, 779, 819-820, 922).
> *(File `bb_render.png`/`bb_gt.png` có trong `images/` nhưng không dùng — bỏ qua hoặc xoá.)*

### A5. `tab:cross` (đối sánh chéo 3 dataset) — sau PHẦN 5
Đọc `reports/comparison_cross_dataset.csv`, điền mean mỗi dataset: PSNR/SSIM/LPIPS-vgg/FPS/MB/#Gauss.
**D-NeRF đã có:** PSNR 34,06 · SSIM 0,9787 · LPIPS-vgg 0,0262 · FPS 38,3 · MB 24,8 · #Gauss ~44.900.

---

### A6. Viết 3 đoạn NHẬN XÉT (prose `\TODO` — cần VIẾT CÂU, không chỉ dán số)
Báo cáo có 3 ô `\TODO` văn xuôi, sau khi có số phải thay bằng nhận xét phân tích:
- **dòng ~621** (HyperNeRF, sau `tab:hypernerf`): đối chiếu PSNR/MS-SSIM mean nhóm với paper 25,2 / 0,845 → 1-2 câu "sát/lệch bao nhiêu; dữ liệu thật nhiễu hơn synthetic nên khó khớp tuyệt đối hơn D-NeRF".
- **dòng ~766** (NeRF-DS, đầu phần kết quả): nêu mức khớp per-scene của nhóm so cột SpectroMotion (sai số trung bình ~? dB) → xác nhận quy trình áp dụng đúng.
- **dòng ~868** (cross-dataset "Đọc kết quả"): điền số cụ thể PSNR D-NeRF > HyperNeRF > NeRF-DS (LPIPS ngược lại) → chốt xu hướng tổng-hợp → thật → phản-chiếu.

> Tổng số `\TODO` trong báo cáo: ~63 (gồm ô bảng A1-A5 + 3 ô văn xuôi A6). Sau khi điền hết, grep `\TODO` phải = 0 (trừ dòng 73 là định nghĩa macro).

## B. Sửa số "nhóm" NHÚNG TRONG VĂN XUÔI (dễ sót nhất — KHÔNG phải `\TODO`, không phải bảng)
- `\caption` `fig:render_vs_gt` (~dòng 581-583): "mutant (37,71 dB)" → **37,47**; "lego (25,06 dB)" → **25,04**.
- §3.2 đối sánh (~dòng 529): "từng cảnh chênh **±0,25 dB**" → run mới max |Δ|≈**0,12** (mutant 37,47 vs 37,59) → sửa "±0,12 dB" (hoặc "dưới 0,15 dB"). KHÔNG để 0,25.
- §3.2 hội tụ (~dòng 541-544): "lego chững ~25 dB" (≈ khớp 25,04 — OK); "hellwarrior coarse **15,2 dB**" → run mới **15,13** (log eval ITER 3000) → sửa.
- §3.5 (~dòng 719-722): "32.143 điểm sieve / 9.714 plate" = số `points.npy` của dataset (ổn định → chỉ verify); "densification **47.000--62.000 điểm**" → thay bằng #Gauss thật sieve/plate run mới (cột #Gauss trong `reports/nerfds/summary.csv`).
- §3.5 (~dòng 730): anecdote "26,4 dB lần chạy đầu" → giữ (minh hoạ bug ratio) hoặc refresh theo run mới.
- §3.5 + Ch4 (NeRF-DS): sieve/plate PSNR + train--test gap → số NeRF-DS run mới khi có.
- **FPS range "34--44 FPS"** (văn xuôi ~dòng 532, 891, 957, 1000): run mới min = lego **30,5** → sửa "**~30--44 FPS**" (cả 4 chỗ).
- `tab:reproduce` + `tab:sota` cột FPS mean "**39,7**" → **38,3**; `tab:sota` Storage "**≈18 MB**" → **~24,8 MB** (run mới gần gấp đôi, nhiều Gaussian hơn); Time "~14 phút" → **~13,5 phút**.
- Mọi chỗ PSNR mean D-NeRF: giữ **34,06** (đã đúng).

(Để chắc đủ: chạy lệnh grep ở mục "⚠️ Phương pháp" đầu file, không dựa danh sách này.)

## C. Copy ảnh từ Drive `figures/` → `report/images/`
1. **`bar_repro_vs_paper.png`** (figure `fig:repro_bar`) — bắt buộc.
2. **`bar_cross_dataset.png`** (`fig:cross`) — bắt buộc.
3. **`scatter_fps_gaussians.png`** (`fig:scatter`) — bắt buộc.
4. **`convergence.png`**: thay bằng `convergence_dnerf.png` của run mới (đổi tên hoặc sửa `\includegraphics{convergence}`).
5. *(tùy)* render/gt mới cho mutant/lego/sieve/plate từ `output/<ds>/<scene>/test/.../` nếu muốn ảnh run mới.

## D. Placeholder cố định (điền tay)
- Bìa (dòng ~131-135): `<giảng viên>`, `<mã lớp học phần>`, `<họ tên + MSSV>` ×2.
- Bảng đóng góp thành viên: `<Họ tên 1/2>`.
- Verify link: source code repo + Colab (đã có sẵn, kiểm còn đúng).

## E. Compile
1. Bảo đảm đã copy đủ 3 ảnh chart (mục C) — thiếu là lỗi "file not found".
2. `pdflatex main.tex` → `pdflatex main.tex` (2 lần cho mục lục + cross-ref).
3. Kiểm không còn `\TODO` (chữ đỏ) sót + không còn `??` (ref hỏng) trong PDF.

## F. Tự kiểm cuối (data accuracy)
- Mọi số "nhóm" trong bảng == giá trị trong `summary.csv` tương ứng (không gõ sai).
- Số "paper/SpectroMotion" giữ nguyên (đã verify 2 lớp, đừng đụng).
- Văn xuôi không mâu thuẫn bảng (vd PSNR nhắc trong text == bảng).

---

### Ghi chú
- D-NeRF mean nhóm **34,06** trùng paper **34,05** (Δ +0,009) → tái hiện chuẩn, là điểm mạnh nhất.
- FPS/Train thấp hơn paper là do GPU T4 ≠ RTX 3090 (đã caveat trong báo cáo) — đọc theo xu hướng.
- Nếu KHÔNG kịp chạy HyperNeRF/NeRF-DS trên Colab: PHẦN 5 partial-friendly, báo cáo vẫn hợp lệ với dataset đã có; khi đó xóa/ghi "chưa chạy" cho phần thiếu thay vì để `\TODO`.
