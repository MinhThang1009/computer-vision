# RUN SUMMARY — 4DGS report

- Ngày: 2026-06-14 17:42
- GPU: Tesla T4
- Fork commit: 209ba00
- Datasets đã chạy: ['dnerf', 'hypernerf', 'nerfds']

## Headline (mean)
| dataset | #scene | PSNR | SSIM | MS-SSIM | LPIPS-vgg |
|---|---|---|---|---|---|
| dnerf | 8 | 34.06 | 0.979 | 0.988 | 0.026 |
| hypernerf | 4 | 25.13 | 0.685 | 0.841 | 0.337 |
| nerfds | 7 | 22.76 | 0.822 | 0.837 | 0.219 |

_Lưu ý: chạy trên torch 2.x / CUDA 12 (Colab) — số có thể lệch nhẹ so với paper (torch 1.13.1)._