<div align="center">

# 🖼️ Computer Vision Project

**Bộ tài liệu LaTeX phục vụ môn học Thị giác máy tính / Xử lý ảnh — UET (ĐHQGHN)**

[![LaTeX](https://img.shields.io/badge/Made%20with-LaTeX-008080?logo=latex&logoColor=white)](https://www.latex-project.org/)
[![Language](https://img.shields.io/badge/Ng%C3%B4n%20ng%E1%BB%AF-Ti%E1%BA%BFng%20Vi%E1%BB%87t-red)](#)
[![Platform](https://img.shields.io/badge/Platform-MiKTeX%20%7C%20TeX%20Live-blue)](#-yêu-cầu-môi-trường)
[![Maintained](https://img.shields.io/badge/Maintained-yes-brightgreen)](#)

[Giới thiệu](#-giới-thiệu) •
[Cấu trúc](#-cấu-trúc-thư-mục) •
[Bắt đầu](#-bắt-đầu) •
[Biên dịch](#-biên-dịch) •
[Đóng góp](#-đóng-góp) •
[Liên hệ](#-liên-hệ)

</div>

---

## 📖 Giới thiệu

Repo này tập hợp tài liệu LaTeX cho dự án môn học **Computer Vision**, gồm 3 phần chính:

- **`template/`** — Template báo cáo môn học chuẩn UET (font 13pt, giãn dòng 1.3, lề trái 3cm / phải 2cm, heading "Chương N", mục lục kiểu khung đỏ, code nền xám). Dùng làm khung cho mọi báo cáo mới.
- **`example/`** — Báo cáo mẫu hoàn chỉnh: *"Tăng cường ảnh sử dụng bộ lọc bảo toàn cạnh (Clustering Filter)"*, kèm hình minh họa (MRI, cameraman, kết quả so sánh) và file PDF đã biên dịch.
- **`4dgs-paper/`** — Mã nguồn LaTeX paper *"4D Gaussian Splatting for Real-Time Dynamic Scene Rendering"* theo template chính thức CVPR/ICCV/3DV, kèm hình vẽ và tài liệu rebuttal — dùng làm tài liệu tham khảo nghiên cứu.

## 📁 Cấu trúc thư mục

```text
Computer Vision Project/
├── template/               # Template báo cáo môn học (LaTeX)
│   ├── main.tex            # File nguồn chính — điền nội dung vào đây
│   ├── main.pdf            # Bản PDF đã biên dịch sẵn
│   ├── images/             # Ảnh dùng trong báo cáo (logo trường có sẵn)
│   └── README.md           # Hướng dẫn chi tiết cách dùng template
├── example/                # Báo cáo mẫu: Clustering Filter
│   ├── main.tex            # Nguồn báo cáo hoàn chỉnh
│   ├── main.pdf            # PDF kết quả
│   └── images/             # Hình minh họa (MRI, cameraman, so sánh…)
└── 4dgs-paper/             # Paper 4D Gaussian Splatting (template CVPR)
    ├── main.tex            # File nguồn chính của paper
    ├── sec/                # Các section (abstract, intro, approach…)
    ├── fig/                # Hình vẽ trong paper (PDF)
    ├── rebuttal/           # Tài liệu rebuttal
    └── cvpr.sty            # Style file chính thức CVPR
```

## 🔧 Yêu cầu môi trường

| Thành phần | Yêu cầu |
|---|---|
| TeX distribution | [MiKTeX](https://miktex.org/) hoặc [TeX Live](https://www.tug.org/texlive/) |
| Engine | `pdflatex` |
| Gói tiếng Việt | `babel-vietnamese`, fontenc `T5` (có sẵn trong bản cài đầy đủ) |

## 🚀 Bắt đầu

```bash
git clone https://github.com/MinhThang1009/computer-vision.git
cd computer-vision
```

Viết báo cáo mới:

1. Copy cả thư mục `template/` thành thư mục mới cho báo cáo của bạn.
2. Mở `main.tex`, điền các chỗ `<...>` trên trang bìa.
3. Viết nội dung vào các chương/section có sẵn (đã có ví dụ mẫu cho công thức, trích dẫn, hình, bảng, code).
4. Chép ảnh vào thư mục `images/`, tham chiếu bằng `\includegraphics{ten_anh}`.
5. Cập nhật tài liệu tham khảo ở cuối file.

## ⚙️ Biên dịch

```bash
pdflatex main.tex
pdflatex main.tex   # chạy 2 lần để mục lục/tham chiếu cập nhật
```

> [!TIP]
> Khi đổi cấu trúc lớn mà gặp lỗi lạ, xóa các file phụ `main.aux`, `main.toc`, `main.out` rồi biên dịch lại.

> [!NOTE]
> - `images/logo_uet.png` là logo trường trên trang bìa — đừng xóa.
> - Code mẫu đặt `language=Python` trong `\lstset` — đổi nếu dùng ngôn ngữ khác.

## 🤝 Đóng góp

Mọi đóng góp đều được hoan nghênh:

1. Fork repo này.
2. Tạo branch mới: `git checkout -b feat/ten-tinh-nang`.
3. Commit thay đổi: `git commit -m "feat: mô tả ngắn gọn"`.
4. Push lên branch: `git push origin feat/ten-tinh-nang`.
5. Mở Pull Request.

## 📚 Tài liệu tham khảo

- [4D Gaussian Splatting for Real-Time Dynamic Scene Rendering](https://arxiv.org/abs/2310.08528) — paper gốc.
- [CVPR Official LaTeX Template](https://github.com/cvpr-org/author-kit) — template paper.
- [Overleaf — Learn LaTeX](https://www.overleaf.com/learn) — tài liệu học LaTeX.

## 📄 Giấy phép

Dự án phục vụ mục đích **học tập và nghiên cứu**. Template CVPR và nội dung paper trong `4dgs-paper/` thuộc bản quyền của các tác giả gốc tương ứng.

## 📬 Liên hệ

- **GitHub**: [@MinhThang1009](https://github.com/MinhThang1009)
- **Issues**: [Mở issue tại đây](https://github.com/MinhThang1009/computer-vision/issues)

---

<div align="center">

⭐ Nếu repo hữu ích, hãy để lại một star!

*Made with ❤️ for Computer Vision course — UET, VNU Hanoi*

</div>
