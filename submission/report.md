# Lab 16 GCP CPU Benchmark Report

1. Tôi dùng Google Cloud Platform tại `us-central1-a` trên VM `e2-medium`; source infrastructure bắt đầu từ commit `55539f67`.
2. Dataset Credit Card Fraud Detection có 284,807 dòng, trong đó 492 dòng gian lận; dữ liệu được chia train/validation/test là 170,883/56,962/56,962 với seed 16 và stratification.
3. Thời gian load dữ liệu là 3.67 giây, training là 4.86 giây và LightGBM chọn best iteration 68.
4. Trên tập test, AUC-ROC là 0.976848, Accuracy là 0.999508, F1 là 0.847826, Precision là 0.906977 và Recall là 0.795918.
5. Inference một dòng mất 1.83 ms (median của 50 lần, warm-up bị loại); batch 1,000 dòng đạt 216,210 dòng/giây (median của 10 lần).
6. Sau benchmark, VM có 3.8 GiB RAM tổng, 3.3 GiB available; ảnh CPU, RAM và network được lưu trong `screenshots/` và được ghi nhận là trạng thái sau chạy.
7. Billing Report của Project GCP tại thời điểm chụp hiển thị 0 đồng; dữ liệu Billing có thể cập nhật chậm nên ảnh Billing được đính kèm kèm thời điểm quan sát.
8. Kết quả và source đã được tải về `submission/`; Terraform đã xóa thành công 16 tài nguyên GCP vào ngày 2026-10-03 và `terraform state list` trả về trống.
