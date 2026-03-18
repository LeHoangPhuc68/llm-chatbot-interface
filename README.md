Dự án triển khai một lớp bảo mật (Security Layer) cho LLM (Qwen 2.5), tập trung vào việc ngăn chặn Prompt Injection bằng cơ chế cơ chế lọc đa lớp.

Hệ thống hoạt động theo mô hình Defense in Depth, xử lý dữ liệu qua 3 giai đoạn chính:
1. Input Guardrail: Kết hợp Heuristics (Regex) và Intent Scoring (ML) để phân loại truy vấn người dùng.
2. LLM Core: Xử lý hội thoại thông qua Ollama Backend với System Prompt được bảo vệ.
3. Output Sanitizer: Kiểm soát dữ liệu đầu ra, chặn thực thi mã độc và ẩn các liên kết không tin cậy.

Technical Features
    Hybrid Injection Detection: Sử dụng bộ lọc Regex cho các pattern tấn công phổ biến (Jailbreak, System Reveal) kết hợp với ML-based scoring để đánh giá rủi ro ngữ cảnh.
    Intent-based Adjustment: Thuật toán điều chỉnh điểm số (Scoring) dựa trên ý định người dùng (Research/Learning vs. Attack) nhằm tối ưu hóa tỷ lệ False Positive.
    Response Hardening: * Chặn các lệnh hệ thống nguy hiểm (rm -rf, sudo, exec,...).
        URL Masking: Lọc và ẩn các URL lạ, chỉ cho phép các domain trong whitelist.
        CJK Ratio Check: Ngăn chặn model bị "jailbreak" dẫn đến phản hồi bằng ngôn ngữ lạ (Trung/Nhật/Hàn).
    Audit Logging: Toàn bộ lịch sử vi phạm được ghi lại chi tiết tại safety.log phục vụ công tác giám sát.

├── app.py                # Entry point: Gradio UI & Guard Logic
├── qwen_guard_project/   # Core ML Backend & Scoring Runtime
├── prompts/              # System Prompt Configuration
├── tools/                # Utilities for dataset & conversion
└── requirements.txt      # Dependency list

Quick Start
    Python 3.10+, Ollama (Model: qwen2.5:7b-instruct).
    pip install -r requirements.txt
    python app.py