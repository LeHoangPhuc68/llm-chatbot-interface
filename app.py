# app.py — Gradio UI + Integrated Guard (lazy ML, intent-based scoring)
# RUN: python app.py
import os
import re
import logging
import pathlib
import requests
import unicodedata
from typing import Tuple, List

import gradio as gr

# ========= BOOTSTRAP IMPORT (để tìm thấy qwen_guard_project/guard_ml) =========
import sys
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
PKG_DIR = PROJECT_ROOT / "qwen_guard_project"
if (PKG_DIR / "guard_ml").exists():
    sys.path.insert(0, str(PKG_DIR))
# ==============================================================================

# ======================================================
# CẤU HÌNH CHUNG
# ======================================================
PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_default.txt"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")

# Tùy chọn
APP_STRICT = os.environ.get("APP_STRICT", "lenient").lower()  # "strict" | "lenient"
SHOW_BLOCK_REASON = os.environ.get("SHOW_BLOCK_REASON", "0") == "1"  # chỉ log, không lộ ra UI
USE_GUARD_ML = os.environ.get("USE_GUARD_ML", "1") == "1"  # bật/tắt ML backend (torch)

# ------------------------------------------------------
# LOGGING — ghi lại hành vi nguy hiểm hoặc lỗi
# ------------------------------------------------------
LOG_PATH = PROJECT_ROOT / "safety.log"
logger = logging.getLogger("chat_safety")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ======================================================
# HÀM ĐỌC SYSTEM PROMPT
# ======================================================
def load_system() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.error(f"Không thể đọc system prompt: {e}")
        return ""

SYSTEM_PROMPT = load_system()

# ======================================================
# GUARD BACKEND (lazy import + fallback)
# ======================================================
class DummyGuard:
    """Fallback không dùng torch; luôn trả về 0.0 (không tăng điểm nguy cơ)."""
    threshold: float = 0.80
    def score(self, _: str) -> float:
        return 0.0

_GUARD_BACKEND = None           # sẽ lazy-load khi cần
_GUARD_BACKEND_ERR = None       # lưu thông báo lỗi nếu có

def _load_guard_backend():
    """Thử import backend ML (có thể nặng). Nếu lỗi -> dùng DummyGuard."""
    global _GUARD_BACKEND, _GUARD_BACKEND_ERR
    if _GUARD_BACKEND is not None:
        return _GUARD_BACKEND
    if not USE_GUARD_ML:
        _GUARD_BACKEND = DummyGuard()
        logger.info("USE_GUARD_ML=0 → dùng DummyGuard (regex-only).")
        return _GUARD_BACKEND
    try:
        # Trì hoãn import cho tới khi thực sự cần
        try:
            from qwen_guard_project.guard_ml.runtime import guard as guard_factory
        except Exception:
            from guard_ml.runtime import guard as guard_factory
        _GUARD_BACKEND = guard_factory()
        logger.info("Guard ML backend loaded OK.")
        return _GUARD_BACKEND
    except Exception as e:
        _GUARD_BACKEND_ERR = str(e)
        logger.warning(f"Không load được ML backend, fallback DummyGuard. Lỗi: {e}")
        _GUARD_BACKEND = DummyGuard()
        return _GUARD_BACKEND

def _ml_threshold_default() -> float:
    be = _load_guard_backend()
    return getattr(be, "threshold", 0.80)

# ======================================================
# THAM SỐ QUYẾT ĐỊNH (đã nới để giảm nhạy)
# ======================================================
HARD_BLOCK = float(os.environ.get("GUARD_HARD_BLOCK") or 0.60)
SOFT_LOG   = float(os.environ.get("GUARD_SOFT_LOG") or 0.30)
ML_WEIGHT  = float(os.environ.get("GUARD_ML_WEIGHT") or 0.35)  # ↓ từ 0.50 -> 0.35
ML_THR     = float(os.environ.get("GUARD_ML_THR") or _ml_threshold_default())

# Regex ngữ cảnh & pattern
IMPERATIVE_RE = re.compile(r"\b(ignore|forget|disregard|bỏ qua|hãy|thực hiện|làm ngay)\b", re.I)
CODEBLOCK_RE  = re.compile(r"```.*?```", re.S)

PATTERNS: List[Tuple[re.Pattern, float]] = [
    # EN
    (re.compile(r"\bignore\b.*\b(previous|all)\b.*\binstructions\b", re.I), 0.55),
    (re.compile(r"\breveal (the )?system prompt\b", re.I),                  0.50),
    (re.compile(r"\bshow (me|your) (system|internal) (prompt|instruction)s?\b", re.I), 0.45),
    (re.compile(r"\bjailbreak\b", re.I),                                     0.40),
    (re.compile(r"\boverride\b", re.I),                                      0.35),
    (re.compile(r"\bdisable (safety|guard)\b", re.I),                        0.40),
    (re.compile(r"\bbypass\b", re.I),                                        0.35),
    # VI
    (re.compile(r"\b(bỏ qua|bỏ qua tất cả|bỏ qua lệnh trước)\b", re.I),     0.45),
    (re.compile(r"\b(tiết lộ|hiện|hiện ra)\b.*\b(system|hệ thống|prompt)\b", re.I), 0.45),
    (re.compile(r"\b(bẻ|bẻ khóa|jailbreak)\b", re.I),                        0.40),
    (re.compile(r"\b(vô hiệu hóa|vô hiệu|tắt)\b.*\b(bảo mật|guard)\b", re.I),0.40),
]

# --- Heuristic phụ trợ để giảm false-positive ---
GREEK_RE = re.compile(r"[\u0370-\u03FF]")  # Greek & Coptic
TEMPLATE_BRACES_RE = re.compile(r"\{\{[^{}]{0,120}\}\}")

# Ý định “học thuật/giải thích”
LEARNING_INTENT_RE = re.compile(
    r"(?:\bví dụ\b|\bexample\b|\bgiải thích\b|\btìm hiểu\b|\bnghiên cứu\b|\bresearch\b|\bstudy\b)",
    re.I
)
# Ý định “thẩm định an toàn/độ tin cậy”
META_SAFETY_INTENT_RE = re.compile(
    r"(?:an\s*toàn|bảo\s*mật|độ\s*tin\s*cậy|uy\s*tín|độ\s*chính\s*xác|nguồn\s*tin\s*cậy|"
    r"trust(?:worthy)?|reliab(?:le|ility)|accuracy|safety\s*concern)",
    re.I
)
QUESTION_MARK_RE = re.compile(r"\?")

# Dấu hiệu lệnh override mạnh
STRONG_DIRECTIVE_RE = re.compile(
    r"(?:\bignore\b.*\binstructions\b|\breveal\b.*\bsystem\s*prompt\b|"
    r"\bdisable\b.*\b(safety|guard)\b|\bbypass\b|\boverride\b|"
    r"\bbỏ\s*qua\b.*\blệnh\b|\btiết\s*lộ\b.*\bsystem\b)",
    re.I
)

def normalize_text(text: str) -> str:
    """NFKC normalize có điều kiện để giảm Introduced_by_norm."""
    s = text or ""
    # Chỉ normalize mạnh khi có >=3 ký tự Greek hoặc Unicode khác thường
    if len(GREEK_RE.findall(s)) >= 3:
        s = unicodedata.normalize("NFKC", s)
    return s.strip()

def _strip_codeblocks(s: str) -> str:
    return CODEBLOCK_RE.sub("", s)

def _regex_score(t: str):
    score = 0.0
    reasons = []
    for pat, w in PATTERNS:
        if pat.search(t):
            score += w
            reasons.append((pat.pattern, w))
    return score, reasons

def _template_literal_safe(t: str) -> bool:
    """Cho qua các literal {{ ... }} nếu không kèm directive nguy hiểm."""
    if not TEMPLATE_BRACES_RE.search(t):
        return False
    danger = any(pat.search(t) for pat, _ in PATTERNS) or re.search(r"(render|include|template)", t, re.I)
    return not danger

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

# ======================================================
# GUARD DECISION — ML + RULE (intent-based)
# ======================================================
    """
    - Tính rule score (0..1) từ regex.
    - Điều chỉnh điểm theo Ý Định (learning / meta-safety / question mark).
    - ML lazy-load; nếu lỗi → 0.0.
    - Block khi:
        * final_score >= HARD_BLOCK, hoặc
        * ml_prob >= 0.95, hoặc
        * (ml_prob >= ML_THR và (rule >= 0.15 hoặc có STRONG_DIRECTIVE)).
    """
def check_input_guard(user_text: str) -> Tuple[bool, str]:

    raw = user_text or ""
    t = normalize_text(raw)
    t_lower = t.lower()
    t_wo_code = _strip_codeblocks(t_lower)

    # 1) Cho qua literal template an toàn
    if _template_literal_safe(t_lower):
        logger.info(f"Template literal allowed | text={raw[:160]!r}")
        return True, ""

    # 2) Điều chỉnh điểm theo Ý Định (không allowlist cứng)
    intent_delta = 0.0
    if LEARNING_INTENT_RE.search(t_lower):
        intent_delta -= 0.30
    if META_SAFETY_INTENT_RE.search(t_lower):
        intent_delta -= 0.30
    if QUESTION_MARK_RE.search(t):
        intent_delta -= 0.10

    # 3) Rule-based
    score, reasons = _regex_score(t_wo_code)
    if IMPERATIVE_RE.search(t_wo_code):
        score += 0.10; reasons.append(("imperative", 0.10))
    score += intent_delta
    reasons.append(("intent_adjust", intent_delta))

    short_len = len(t_wo_code.replace(" ", ""))
    if short_len < 60 and any(p.search(t_wo_code) for p, _ in PATTERNS):
        score += 0.10; reasons.append(("short_text", 0.10))
    score = clamp01(score)

    # 4) ML (lazy)
    ml_prob = 0.0
    try:
        be = _load_guard_backend()
        ml_prob = float(be.score(raw))
    except Exception as e:
        logger.warning(f"ML backend error → dùng regex-only. {e}")

    ml_contrib = ML_WEIGHT * ml_prob
    final_score = clamp01(score + ml_contrib)

    has_strong_directive = bool(STRONG_DIRECTIVE_RE.search(t_lower))

    logger.debug(
        f"Guard check: rule={score:.3f}, ml={ml_prob:.3f}, add={ml_contrib:.3f}, "
        f"final={final_score:.3f}, ml_thr={ML_THR:.3f}, strong={has_strong_directive}, "
        f"reasons={reasons}, text={raw[:200]!r}"
    )

    # 5) Quyết định (ML không tự block khi rule thấp, trừ khi cực cao)
    if (final_score >= HARD_BLOCK) or (ml_prob >= 0.95) or ((ml_prob >= ML_THR) and (score >= 0.15 or has_strong_directive)):
        logger.warning(
            f"⛔ Blocked input (rule={score:.2f}, ml={ml_prob:.2f}, final={final_score:.2f}, "
            f"ml_thr={ML_THR:.2f}, strong={has_strong_directive}) | text={raw[:200]!r}"
        )
        reason = f"(final={final_score:.2f}; ml={ml_prob:.2f})"
        return False, reason
    elif final_score >= SOFT_LOG:
        logger.info(
            f"⚠️ Suspicious input (rule={score:.2f}, ml={ml_prob:.2f}, final={final_score:.2f}) | text={raw[:200]!r}"
        )
        return True, ""
    else:
        return True, ""

# ======================================================
# HÀM LỌC / SANITIZE OUTPUT MODEL
# ======================================================
CJK_RE = re.compile(r"[\u4E00-\u9FFF]")
DANGEROUS_RE = re.compile(
    r"(?:\brm\s+-rf\b|sudo\b|curl\s+https?://|wget\s+https?://|\beval\s*\(|\bexec\s*\(|\bsystem\s*\()",
    re.I,
)
URL_RE = re.compile(r"https?://[^\s)]+", re.I)
CODEBLOCK_RE_OUT = re.compile(r"```.*?```", re.S)
WHITELIST_DOMAINS = {"localhost", "127.0.0.1", "0.0.0.0"}

def _strip_codeblocks_out(s: str):
    blocks = CODEBLOCK_RE_OUT.findall(s or "")
    text_wo = CODEBLOCK_RE_OUT.sub("", s or "")
    return text_wo, blocks

def _alpha_num_len(s: str) -> int:
    return sum(1 for ch in s if ch.isalnum())

def _count_cjk_chars(s: str) -> int:
    return sum(1 for ch in s if '\u4E00' <= ch <= '\u9FFF')

def _mask_urls(text: str) -> str:
    def repl(m):
        url = m.group(0)
        host = re.sub(r'^https?://', '', url, flags=re.I).split('/')[0].split(':')[0]
        if host in WHITELIST_DOMAINS:
            return url
        return "[link đã ẩn]"
    return URL_RE.sub(repl, text)

def sanitize_response(text: str):
    if not text:
        return text, False
    text = unicodedata.normalize("NFC", text)
    text_wo_code, code_blocks = _strip_codeblocks_out(text)

    if DANGEROUS_RE.search(text_wo_code):
        logger.warning("⚠️ Dangerous command in output.")
        return "Nội dung bị chặn vì có yếu tố nguy hiểm.", True

    cjk_count = _count_cjk_chars(text_wo_code)
    denom = max(1, _alpha_num_len(text_wo_code))
    if cjk_count >= 5 and (cjk_count / denom) > 0.15:
        logger.warning("⚠️ High CJK ratio in output.")
        return "Xin lỗi, mình chỉ sử dụng tiếng Việt. Vui lòng hỏi lại bằng tiếng Việt.", True

    masked_wo = _mask_urls(text_wo_code)
    if masked_wo != text_wo_code:
        rebuilt = CODEBLOCK_RE_OUT.sub(lambda _: code_blocks.pop(0) if code_blocks else "``` ```",
                                       text.replace(text_wo_code, masked_wo, 1))
        logger.info("🔗 Masked URL(s) in output.")
        return rebuilt, True

    return text, False

# ======================================================
# HÀM CHÍNH GỌI MODEL (Gradio handler)
# ======================================================
def chat_ollama(message, chat_history):
    """Gửi câu hỏi tới Ollama model và nhận phản hồi (kèm system prompt + bảo vệ đầu vào/ra)."""
    # --- Guard input ---
    allowed, reason = check_input_guard(message)
    if not allowed:
        user_msg = "Xin lỗi, yêu cầu này không hợp lệ hoặc vi phạm quy tắc bảo mật. Vui lòng đặt câu hỏi khác nhé 💬"
        if SHOW_BLOCK_REASON:
            user_msg += f" {reason}"
        chat_history.append((message, user_msg))
        return "", chat_history

    # --- Build messages ---
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    for u, a in chat_history:
        if u: messages.append({"role": "user", "content": u})
        if a: messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": message})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "language": "vi-VN",
            "temperature": 0.2 if APP_STRICT == "strict" else 0.3,
            "top_p": 0.8
        },
    }

    try:
        r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        answer = data.get("message", {}).get("content", "")

        # --- Sanitize output ---
        safe_answer, modified = sanitize_response(answer)
        if modified:
            logger.info(f"Modified model output. Original snippet: {answer[:120]!r}")

        chat_history.append((message, safe_answer))
    except Exception as e:
        logger.exception("Lỗi khi gọi model")
        chat_history.append((message, f"⚠️ Lỗi khi gọi model: {e}"))

    return "", chat_history

# ======================================================
# GIAO DIỆN GRADIO
# ======================================================
custom_css = """
html, body, .gradio-container { height: 100%; }
#page { height: 100vh; display: flex; flex-direction: column; }
#header { padding-bottom: .25rem; }
#chatwrap { flex: 1; min-height: 0; }
#chatbot { height: 78vh !important; }
#bottombar { position: sticky; bottom: 0; padding: 6px 0 10px 0; }
#sendbtn { max-width: 110px; min-width: 90px; }
"""

with gr.Blocks(
    title="Qwen 2.5 Chat (Ollama Backend)",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    css=custom_css,
    elem_id="page",
) as demo:
    gr.Markdown("### 💬 Qwen 2.5 Chat (Ollama Backend)", elem_id="header")

    with gr.Column(elem_id="chatwrap"):
        chatbot = gr.Chatbot(label="", elem_id="chatbot", show_label=False)

    with gr.Row(elem_id="bottombar"):
        msg = gr.Textbox(
            placeholder="Nhập câu hỏi... (Shift+Enter để xuống dòng)",
            label="",
            scale=10,
        )
        send = gr.Button("Gửi", elem_id="sendbtn", variant="secondary", scale=1)

    send.click(chat_ollama, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(chat_ollama, inputs=[msg, chatbot], outputs=[msg, chatbot])

demo.launch(server_name="127.0.0.1", server_port=7860)
