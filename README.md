# Qwen-Guard: Secure LLM Interface with Multi-stage Defense

Dự án triển khai một lớp bảo mật (**Security Layer**) cho LLM (Qwen 2.5), tập trung vào việc ngăn chặn Prompt Injection bằng cơ chế lọc đa lớp theo mô hình **Defense in Depth**.

## Hệ thống Kiến trúc
Xử lý dữ liệu qua 3 giai đoạn chính:
1. **Input Guardrail:** Kết hợp Heuristics (Regex) và Intent Scoring (ML) để phân loại truy vấn.
2. **LLM Core:** Xử lý hội thoại qua Ollama Backend với System Prompt được bảo vệ.
3. **Output Sanitizer:** Kiểm soát đầu ra, chặn thực thi mã độc và ẩn các liên kết không tin cậy.

## Tính năng Kỹ thuật
* **Hybrid Injection Detection:** Kết hợp Regex pattern (Jailbreak, System Reveal) và ML-based scoring.
* **Intent-based Adjustment:** Thuật toán điều chỉnh Scoring dựa trên ý định người dùng (Research/Learning vs. Attack) để giảm tỷ lệ **False Positive**.
* **Response Hardening:**
    * Chặn lệnh hệ thống nguy hiểm (`rm -rf`, `sudo`, `exec`,...).
    * **URL Masking:** Chỉ cho phép các domain trong whitelist.
    * **CJK Ratio Check:** Ngăn chặn phản hồi bằng ngôn ngữ lạ (Trung/Nhật/Hàn).
* **Audit Logging:** Ghi nhật ký vi phạm chi tiết tại `safety.log`.

## Cấu trúc thư mục
```text
├── app.py                # Entry point: Gradio UI & Guard Logic
├── qwen_guard_project/   # Core ML Backend & Scoring Runtime
├── prompts/              # System Prompt Configuration
├── tools/                # Utilities for dataset & conversion
└── requirements.txt      # Dependency list
