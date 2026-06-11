# Kết quả reproduce 4D Gaussian Splatting (D-NeRF dataset)

Môi trường: Google Colab free — GPU Tesla T4 (16GB), Python 3.12, PyTorch 2.11.0+cu128.
Ngày chạy: 11/06/2026. Train đầy đủ 2 stage (coarse 3.000 + fine 20.000 iteration)/scene.
Code: [hustvl/4DGaussians](https://github.com/hustvl/4DGaussians) (qua fork chỉnh cho Colab của Tasmay-Tibrewal — thuật toán không đổi).
Metrics tính bằng `metrics.py` chính thức của repo tại iteration 20.000.
Số paper lấy từ bảng per-scene trong supplementary của paper (CVPR 2024, đo trên RTX 3090).

## Bảng đối sánh PSNR / SSIM (test set)

| Scene | PSNR (reproduce) | PSNR (paper) | Δ | SSIM (reproduce) | SSIM (paper) |
|---|---|---|---|---|---|
| bouncingballs | **40.85** | 40.62 | +0.23 | **0.9944** | 0.9942 |
| hellwarrior | 28.69 | 28.71 | −0.02 | 0.9733 | 0.9733 |
| hook | **32.97** | 32.73 | +0.24 | **0.9768** | 0.9760 |
| jumpingjacks | 35.25 | 35.42 | −0.17 | 0.9854 | 0.9857 |
| lego | **25.06** | 25.03 | +0.03 | **0.9379** | 0.9376 |
| mutant | **37.71** | 37.59 | +0.12 | 0.9879 | 0.9880 |
| standup | 38.00 | 38.11 | −0.11 | 0.9898 | 0.9898 |
| trex | 33.98 | 34.23 | −0.25 | 0.9842 | 0.9850 |
| **Trung bình** | **34.06** | **34.06** | **0.00** | 0.9787 | 0.9787 |

## LPIPS và FPS render (T4)

| Scene | LPIPS-vgg | LPIPS-alex | FPS (video 160 frame) |
|---|---|---|---|
| bouncingballs | 0.0149 | 0.0057 | 36.0 |
| hellwarrior | 0.0372 | 0.0248 | 41.2 |
| hook | 0.0268 | 0.0165 | 43.5 |
| jumpingjacks | 0.0201 | 0.0131 | 40.1 |
| lego | 0.0561 | 0.0382 | 34.5 |
| mutant | 0.0166 | 0.0085 | 41.8 |
| standup | 0.0139 | 0.0073 | 44.0 |
| trex | 0.0223 | 0.0139 | 36.2 |

## Nhận xét

- PSNR trung bình 8 scene **trùng khớp số paper (34.06 dB)**; từng scene chênh trong khoảng ±0.25 dB — sai số ngẫu nhiên bình thường giữa các lần train, kết luận reproduce thành công.
- FPS 34–44 trên T4, thấp hơn số paper (~82 FPS trên RTX 3090) do GPU yếu hơn — xu hướng real-time vẫn được xác nhận.
- Thời gian train ~14 phút/scene trên T4 (paper: ~8 phút trên 3090) — xác nhận tốc độ hội tụ nhanh của phương pháp.
- Scene `lego` thấp (25 dB) ở cả reproduce lẫn paper — đặc thù scene này của D-NeRF (test set lệch so với train), không phải lỗi reproduce.

## Ghi chú kỹ thuật khi chạy trên Colab (2026)

- `requirements.txt` chứa `argparse` (package PyPI cũ, trùng stdlib) → gỡ sau khi cài để tránh cảnh báo "restart runtime".
- `simple-knn` (commit cũ) không compile được với toolchain mới → vá thêm `#include <float.h>` vào `simple_knn.cu` trước khi build.

## Cấu trúc thư mục

- `<scene>/results.json` — metrics chính thức (PSNR/SSIM/LPIPS/MS-SSIM/D-SSIM)
- `<scene>/renders/` + `<scene>/gt/` — 5 cặp ảnh render vs ground-truth (cùng tên = cùng frame) làm hình so sánh
- `<scene>/video_rgb.mp4` — video render quỹ đạo camera mới
