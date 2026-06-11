# Template báo cáo môn học (LaTeX)

Template báo cáo môn học chuẩn UET: font 13pt, giãn dòng 1.3,
lề trái 3cm / phải 2cm, heading "Chương N", mục lục kiểu khung đỏ, code nền xám.

## Cách dùng

1. Copy cả thư mục này cho mỗi báo cáo mới.
2. Mở `main.tex`, điền các chỗ `<...>` trên trang bìa.
3. Viết nội dung vào các chương/section có sẵn (đã có ví dụ mẫu cho công thức,
   trích dẫn, hình, bảng, code).
4. Chép ảnh vào thư mục `images/`, tham chiếu bằng `\includegraphics{ten_anh}`
   (khối figure mẫu trong Chương 1 đang comment — bỏ comment khi có ảnh).
5. Cập nhật tài liệu tham khảo ở cuối file.

## Biên dịch

```
pdflatex main.tex
pdflatex main.tex   (chạy 2 lần để mục lục/tham chiếu cập nhật)
```

Yêu cầu: MiKTeX hoặc TeX Live (pdfLaTeX). Khi đổi cấu trúc lớn mà gặp lỗi lạ,
xóa các file phụ `main.aux`, `main.toc`, `main.out` rồi biên dịch lại.

## Lưu ý

- `images/logo_uet.png` là logo trường trên bìa — đừng xóa.
- Code mẫu đặt `language=Python` trong `\lstset` — đổi nếu dùng ngôn ngữ khác.
- Template chưa tinh chỉnh style cho `\subsection` (báo cáo gốc không dùng cấp này).
