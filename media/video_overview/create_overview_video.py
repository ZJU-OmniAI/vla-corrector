#!/usr/bin/env python3
"""Generate bilingual VLA-Corrector overview videos.

The script intentionally uses the paper figures and real-robot clips already
stored in docs/assets. It avoids generated toy diagrams and builds a compact
academic teaser with voice-over, result cards, and demo footage.
"""

from __future__ import annotations

import json
import math
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
OUT_VIDEOS = ASSETS / "videos"
OUT_SUBTITLES = ASSETS / "subtitles"
OUT_IMAGES = ASSETS / "images"
BUILD = ROOT / "media" / "video_overview" / "build"
SCRIPT_DIR = ROOT / "media" / "video_overview"

W, H = 1280, 720
FPS = 24

BLUE = "#2f5f9f"
BLUE_DARK = "#173b68"
BLUE_SOFT = "#edf3fb"
TEXT = "#17202a"
MUTED = "#5f6b7a"
LINE = "#dbe3ee"
PAPER = "#fbfcfe"
PANEL = "#ffffff"
SOFT = "#f5f8fc"
GREEN = "#2f8f5b"


@dataclass
class Scene:
    key: str
    min_duration: float
    en_voice: str
    zh_voice: str


SCENES = [
    Scene(
        "title",
        8.0,
        "Meet VLA-Corrector: a lightweight detect-and-correct layer for action-chunked vision-language-action policies.",
        "这是 VLA-Corrector：一个面向动作块 VLA 策略的轻量级检测与纠错推理框架。",
    ),
    Scene(
        "problem",
        13.0,
        "Action chunks reduce expensive VLA calls, but they also create an open-loop blind spot. After a small drift, the robot may keep executing stale actions until the horizon ends.",
        "动作块可以减少昂贵的 VLA 调用，但也带来了开环执行盲区。一次小的偏移之后，机器人可能继续执行已经过时的动作，直到 horizon 结束。",
    ),
    Scene(
        "method",
        15.0,
        "VLA-Corrector keeps the backbone policy frozen. A latent-space vision monitor detects visual dynamics mismatch, truncates stale actions, and triggers OGG-guided corrective replanning.",
        "VLA-Corrector 保持 VLA 主干冻结。Latent-space Vision Monitor 检测视觉动态不一致，截断过时动作，并触发 OGG 引导的纠错式重规划。",
    ),
    Scene(
        "corrector",
        12.0,
        "The trainable part is only a small external corrector, about forty million parameters in the paper. It learns local latent dynamics from demonstrations instead of retraining the full VLA.",
        "可训练部分只是一个外部轻量纠错器。论文中的默认规模约为四千万参数，它从 demonstration 中学习局部 latent dynamics，而不是重新训练完整 VLA。",
    ),
    Scene(
        "results",
        15.0,
        "Across simulation and real-world tasks, the paper reports clear gains: plus fifteen point six five points on MetaWorld with PI zero point five, plus three point eight on LIBERO, and plus seventeen point seven on AgileX PiPER real-world tasks.",
        "在仿真和真实机器人任务上，论文报告了明显提升：PI 零点五在 MetaWorld 上提升十五点六五个百分点，LIBERO 提升三点八个百分点，AgileX PiPER 真机任务平均提升十七点七个百分点。",
    ),
    Scene(
        "demos",
        15.0,
        "In real-robot disturbance demos, VLA-Corrector is designed to stop trusting stale chunks and recover execution when the object, target, or drawer is moved during the task.",
        "在真实机器人扰动展示中，当物体、目标碗或抽屉在执行中被移动时，VLA-Corrector 的目标是停止信任过时动作块，并恢复执行。",
    ),
    Scene(
        "closing",
        8.0,
        "The code and project page are available now. The paper and arXiv link are coming soon.",
        "代码和项目主页已经公开。论文和 arXiv 链接即将发布。",
    ),
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def capture(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def duration(path: Path) -> float:
    return float(
        capture(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
        )
    )


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


FONT_REG = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf"
FONT_GEORGIA = "/usr/share/fonts/truetype/msttcorefonts/georgiab.ttf"


def load_font(size: int, bold: bool = False, serif: bool = False) -> ImageFont.FreeTypeFont:
    if serif:
        return font(FONT_GEORGIA, size)
    return font(FONT_BOLD if bold else FONT_REG, size)


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def rounded(draw: ImageDraw.ImageDraw, box, radius: int, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text_wrap(draw: ImageDraw.ImageDraw, text: str, font_obj, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        line = ""
        for word in words:
            test = (line + " " + word).strip()
            if draw.textbbox((0, 0), test, font=font_obj)[2] <= max_width:
                line = test
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
    return lines


def draw_wrapped(draw, xy, text, font_obj, fill, max_width, line_gap=6):
    x, y = xy
    for line in text_wrap(draw, text, font_obj, max_width):
        draw.text((x, y), line, font=font_obj, fill=fill)
        y += draw.textbbox((0, 0), line, font=font_obj)[3] + line_gap
    return y


def fit_image(img: Image.Image, box: tuple[int, int, int, int], cover=False) -> Image.Image:
    bw, bh = box[2] - box[0], box[3] - box[1]
    iw, ih = img.size
    scale = max(bw / iw, bh / ih) if cover else min(bw / iw, bh / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    out = img.resize((nw, nh), Image.LANCZOS)
    if cover:
        left = max(0, (nw - bw) // 2)
        top = max(0, (nh - bh) // 2)
        out = out.crop((left, top, left + bw, top + bh))
    return out


def paste_center(canvas: Image.Image, img: Image.Image, box, cover=False):
    fitted = fit_image(img, box, cover=cover)
    x = box[0] + ((box[2] - box[0]) - fitted.size[0]) // 2
    y = box[1] + ((box[3] - box[1]) - fitted.size[1]) // 2
    canvas.paste(fitted, (x, y))


def base_canvas() -> Image.Image:
    img = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(img)
    for i in range(H):
        t = i / H
        r = int(251 * (1 - t) + 245 * t)
        g = int(252 * (1 - t) + 248 * t)
        b = int(254 * (1 - t) + 252 * t)
        draw.line([(0, i), (W, i)], fill=(r, g, b))
    return img


def card(draw, box, radius=22):
    rounded(draw, box, radius, PANEL, LINE, 1)


def header(draw, title: str, section: str):
    draw.text((64, 38), section.upper(), font=load_font(16, bold=True), fill=BLUE)
    draw.text((64, 64), title, font=load_font(38, bold=True, serif=True), fill=TEXT)


def chip(draw, xy, label: str, fill=BLUE_SOFT, color=BLUE):
    x, y = xy
    f = load_font(15, bold=True)
    bbox = draw.textbbox((0, 0), label, font=f)
    w = bbox[2] - bbox[0] + 24
    rounded(draw, (x, y, x + w, y + 32), 16, fill, None)
    draw.text((x + 12, y + 7), label, font=f, fill=color)
    return x + w


def load_assets():
    return {
        "logo": Image.open(ASSETS / "branding" / "vla_corrector_logo.png").convert("RGB"),
        "lab": Image.open(ASSETS / "branding" / "lab_logo.jpg").convert("RGB"),
        "teaser": Image.open(ASSETS / "images" / "teaser_open_loop_vs_closed_loop.png").convert("RGB"),
        "method": Image.open(ASSETS / "images" / "method_overview.png").convert("RGB"),
        "results": Image.open(ASSETS / "images" / "results_pareto.png").convert("RGB"),
        "truncation": Image.open(ASSETS / "images" / "truncation_phase_analysis.png").convert("RGB"),
        "recovery": Image.open(ASSETS / "images" / "qualitative_recovery.png").convert("RGB"),
    }


def slide_title(assets, p: float) -> Image.Image:
    img = base_canvas()
    draw = ImageDraw.Draw(img)
    logo_box = (76, 96, 258, 278)
    paste_center(img, assets["logo"], logo_box, cover=False)
    draw.text((300, 112), "VLA-Corrector", font=load_font(68, bold=True, serif=True), fill=BLUE_DARK)
    draw.text(
        (304, 196),
        "Lightweight Detect-and-Correct Inference\nfor Adaptive Action Horizon",
        font=load_font(30, bold=True),
        fill=TEXT,
        spacing=8,
    )
    y = 310
    for label in ["Embodied AI", "Vision-Language-Action", "Action Correction", "~40M Corrector"]:
        x = 304 if label == "Embodied AI" else x + 12
        x = chip(draw, (x, y), label)
    card(draw, (86, 430, 1194, 610), 28)
    draw.text((124, 464), "The question", font=load_font(22, bold=True), fill=BLUE)
    draw_wrapped(
        draw,
        (124, 500),
        "Can a frozen VLA keep long-horizon efficiency while recovering from execution drift before stale actions accumulate?",
        load_font(28, bold=True),
        TEXT,
        1030,
        8,
    )
    draw.text((88, 650), "Code: github.com/ZJU-OmniAI/vla-corrector   ·   Paper / arXiv: Coming soon", font=load_font(17), fill=MUTED)
    return img


def slide_problem(assets, p: float) -> Image.Image:
    img = base_canvas()
    draw = ImageDraw.Draw(img)
    header(draw, "Open-loop action chunks leave blind spots.", "01  Motivation")
    card(draw, (64, 130, 1216, 548), 18)
    paste_center(img, assets["teaser"], (94, 158, 1186, 492), cover=False)
    draw.text((88, 572), "Long horizon: fewer policy calls, but stale actions may continue after drift.", font=load_font(22, bold=True), fill=TEXT)
    draw.text((88, 610), "Strict H=1 replanning is reactive, but expensive for VLA policies.", font=load_font(20), fill=MUTED)
    for i, label in enumerate(["stale actions", "pose drift", "late recovery"]):
        chip(draw, (810 + i * 125, 584), label, fill="#fff2e8", color="#a4511f")
    return img


def slide_method(assets, p: float) -> Image.Image:
    img = base_canvas()
    draw = ImageDraw.Draw(img)
    header(draw, "Correct the execution, not the whole VLA.", "02  Method")
    card(draw, (64, 130, 1216, 560), 18)
    paste_center(img, assets["method"], (88, 158, 1192, 520), cover=False)
    labels = [
        ("LVM detects drift", 94),
        ("Truncate stale actions", 414),
        ("OGG-guided replan", 742),
    ]
    for text, x in labels:
        rounded(draw, (x, 580, x + 260, 626), 23, BLUE_SOFT, None)
        draw.text((x + 18, 592), text, font=load_font(18, bold=True), fill=BLUE_DARK)
    draw.text((88, 652), "Frozen VLA backbone · External latent dynamics corrector · Event-triggered adaptive horizon", font=load_font(18), fill=MUTED)
    return img


def slide_corrector(assets, p: float) -> Image.Image:
    img = base_canvas()
    draw = ImageDraw.Draw(img)
    header(draw, "A small external module carries the correction signal.", "03  Corrector")
    cards = [
        ("~40M", "MLP corrector", "38--42M parameters across evaluated settings"),
        ("Frozen", "VLA backbone", "No full policy retraining at inference time"),
        ("Local", "latent dynamics", "Predicts short-horizon visual residuals"),
    ]
    for i, (big, title, desc) in enumerate(cards):
        x = 74 + i * 390
        card(draw, (x, 145, x + 350, 345), 24)
        draw.text((x + 28, 178), big, font=load_font(56, bold=True, serif=True), fill=BLUE_DARK)
        draw.text((x + 30, 248), title, font=load_font(24, bold=True), fill=TEXT)
        draw_wrapped(draw, (x + 30, 284), desc, load_font(17), MUTED, 285, 5)
    card(draw, (74, 390, 1206, 626), 20)
    paste_center(img, assets["truncation"], (104, 418, 1176, 586), cover=False)
    draw.text((96, 646), "Interrupts concentrate in critical phases such as grasping and alignment.", font=load_font(20), fill=MUTED)
    return img


def result_card(draw, x, y, title, value, sub, accent=GREEN):
    card(draw, (x, y, x + 270, y + 166), 22)
    draw.text((x + 22, y + 22), title, font=load_font(18, bold=True), fill=TEXT)
    draw.text((x + 22, y + 58), value, font=load_font(42, bold=True, serif=True), fill=accent)
    draw_wrapped(draw, (x + 22, y + 112), sub, load_font(15), MUTED, 220, 3)


def slide_results(assets, p: float) -> Image.Image:
    img = base_canvas()
    draw = ImageDraw.Draw(img)
    header(draw, "Reported gains across simulation and real robots.", "04  Results")
    result_card(draw, 76, 150, "MetaWorld / PI0.5", "+15.65 pts", "48.70% to 64.35% average success")
    result_card(draw, 376, 150, "LIBERO / PI0.5", "+3.80 pts", "94.00% to 97.80% few-shot average success")
    result_card(draw, 676, 150, "AgileX PiPER", "+17.7 pts", "55.6% to 73.3% real-world average success")
    result_card(draw, 976, 150, "Critical phases", "83.7%", "truncations occur in manually labeled critical phases", accent=BLUE)
    card(draw, (76, 370, 1204, 628), 20)
    paste_center(img, assets["results"], (106, 398, 600, 600), cover=False)
    paste_center(img, assets["recovery"], (640, 425, 1170, 560), cover=False)
    draw.text((640, 585), "Controlled recovery: truncate stale chunk, replan with OGG, complete the task.", font=load_font(18), fill=MUTED)
    return img


def read_demo_frames(caps):
    frames = []
    for cap in caps:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 80)
            ok, frame = cap.read()
        if not ok:
            frame = np.zeros((540, 960, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    return frames


def slide_demos(assets, p: float, demo_frames=None) -> Image.Image:
    img = base_canvas()
    draw = ImageDraw.Draw(img)
    header(draw, "Real-robot disturbance recovery demos.", "05  Demos")
    boxes = [(64, 145, 438, 356), (453, 145, 827, 356), (842, 145, 1216, 356)]
    titles = ["Drawer alignment", "Block to blue bowl", "Block to white bowl"]
    if demo_frames is None:
        demo_frames = [Image.new("RGB", (960, 540), "#0f172a") for _ in boxes]
    for frame, box, title in zip(demo_frames, boxes, titles):
        card(draw, (box[0] - 8, box[1] - 8, box[2] + 8, box[3] + 68), 18)
        paste_center(img, frame, box, cover=True)
        draw.text((box[0], box[3] + 22), title, font=load_font(18, bold=True), fill=TEXT)
        draw.text((box[0], box[3] + 48), "Human perturbation during execution", font=load_font(15), fill=MUTED)
    card(draw, (104, 505, 1176, 632), 24)
    draw.text((134, 532), "VLA-Corrector is designed for the moment the current action chunk should no longer be trusted.", font=load_font(27, bold=True), fill=TEXT)
    draw.text((134, 584), "Detect drift · Interrupt stale actions · Guide the next recovery query", font=load_font(22), fill=BLUE_DARK)
    return img


def slide_closing(assets, p: float) -> Image.Image:
    img = base_canvas()
    draw = ImageDraw.Draw(img)
    paste_center(img, assets["logo"], (505, 80, 775, 350), cover=False)
    draw.text((306, 378), "VLA-Corrector", font=load_font(66, bold=True, serif=True), fill=BLUE_DARK)
    draw.text((315, 462), "Lightweight correction for action-chunked VLA execution", font=load_font(27, bold=True), fill=TEXT)
    rounded(draw, (338, 536, 942, 596), 30, BLUE, None)
    draw.text((392, 553), "github.com/ZJU-OmniAI/vla-corrector", font=load_font(23, bold=True), fill="#ffffff")
    draw.text((487, 638), "Paper / arXiv: Coming soon", font=load_font(20), fill=MUTED)
    return img


SLIDE_FN = {
    "title": slide_title,
    "problem": slide_problem,
    "method": slide_method,
    "corrector": slide_corrector,
    "results": slide_results,
    "demos": slide_demos,
    "closing": slide_closing,
}


def synthesize_voice(lang: str, scenes: list[Scene]) -> tuple[list[Path], list[float]]:
    voice = "en-US-GuyNeural" if lang == "en" else "zh-CN-YunxiNeural"
    audio_dir = BUILD / lang / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    durations = []
    for i, scene in enumerate(scenes):
        text = scene.en_voice if lang == "en" else scene.zh_voice
        out = audio_dir / f"{i:02d}_{scene.key}.mp3"
        if not out.exists():
            run(["python", "-m", "edge_tts", "--voice", voice, "--text", text, "--write-media", str(out)])
        d = duration(out)
        paths.append(out)
        durations.append(max(scene.min_duration, d + 0.7))
    return paths, durations


def concat_audio(lang: str, paths: list[Path], durations: list[float]) -> Path:
    padded_dir = BUILD / lang / "padded_audio"
    padded_dir.mkdir(parents=True, exist_ok=True)
    padded_paths = []
    for i, (path, dur) in enumerate(zip(paths, durations)):
        padded = padded_dir / f"{i:02d}.wav"
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-af",
                "apad",
                "-t",
                f"{dur:.3f}",
                "-c:a",
                "pcm_s16le",
                str(padded),
            ]
        )
        padded_paths.append(padded)
    list_path = BUILD / lang / "audio_list.txt"
    list_path.write_text("\n".join(f"file '{p.resolve()}'" for p in padded_paths) + "\n")
    out = BUILD / lang / "voice.m4a"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c:a", "aac", "-b:a", "128k", str(out)])
    return out


def frame_to_bgr(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def render_video(lang: str, durations: list[float]) -> Path:
    assets = load_assets()
    raw = BUILD / lang / "silent_raw.mp4"
    raw.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    demo_paths = [
        OUT_VIDEOS / "drawer_alignment_perturbation.mp4",
        OUT_VIDEOS / "block_to_blue_bowl_perturbation.mp4",
        OUT_VIDEOS / "block_to_white_bowl_perturbation.mp4",
    ]
    static_cache = {
        scene.key: frame_to_bgr(SLIDE_FN[scene.key](assets, 0.0))
        for scene in SCENES
        if scene.key != "demos"
    }
    for scene, dur in zip(SCENES, durations):
        n = int(round(dur * FPS))
        caps = None
        if scene.key == "demos":
            caps = [cv2.VideoCapture(str(p)) for p in demo_paths]
            for cap in caps:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 80)
        for fidx in range(n):
            p = fidx / max(1, n - 1)
            if scene.key == "demos" and caps is not None:
                frames = read_demo_frames(caps)
                img = slide_demos(assets, p, frames)
                writer.write(frame_to_bgr(img))
            else:
                writer.write(static_cache[scene.key])
        if caps:
            for cap in caps:
                cap.release()
    writer.release()
    return raw


def write_srt(lang: str, durations: list[float]) -> Path:
    OUT_SUBTITLES.mkdir(parents=True, exist_ok=True)
    out = OUT_SUBTITLES / f"vla_corrector_overview_{lang}.srt"
    t = 0.0
    lines = []
    for idx, (scene, dur) in enumerate(zip(SCENES, durations), 1):
        text = scene.en_voice if lang == "en" else scene.zh_voice
        start = t
        end = t + dur
        lines.append(str(idx))
        lines.append(f"{fmt_time(start)} --> {fmt_time(end)}")
        lines.extend(wrap_caption(text, lang, scene.key))
        lines.append("")
        t = end
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def text_units(text: str) -> int:
    return sum(2 if ord(ch) > 127 else 1 for ch in text)


def zh_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    token = ""
    for ch in text:
        is_ascii_word = ch.isascii() and (ch.isalnum() or ch in ".-/+_")
        if is_ascii_word:
            token += ch
            continue
        if token:
            tokens.append(token)
            token = ""
        tokens.append(" " if ch.isspace() else ch)
    if token:
        tokens.append(token)
    return tokens


ZH_CAPTION_LINES = {
    "title": [
        "这是 VLA-Corrector：一个面向动作块 VLA 策略的",
        "轻量级检测与纠错推理框架。",
    ],
    "problem": [
        "动作块可以减少昂贵的 VLA 调用，",
        "但也带来了开环执行盲区。",
        "一次小的偏移之后，机器人可能继续执行过时动作，",
        "直到 horizon 结束。",
    ],
    "method": [
        "VLA-Corrector 保持 VLA 主干冻结。",
        "Latent-space Vision Monitor 检测视觉动态不一致，",
        "截断过时动作，并触发 OGG 引导的纠错式重规划。",
    ],
    "corrector": [
        "可训练部分只是一个外部轻量纠错器。",
        "论文中的默认规模约为四千万参数，",
        "它从 demonstration 中学习局部 latent dynamics，",
        "而不是重新训练完整 VLA。",
    ],
    "results": [
        "在仿真和真实机器人任务上，论文报告了明显提升：",
        "PI0.5 在 MetaWorld 上提升十五点六五个百分点，",
        "LIBERO 提升三点八个百分点，",
        "AgileX PiPER 真机任务平均提升十七点七个百分点。",
    ],
    "demos": [
        "在真实机器人扰动展示中，",
        "当物体、目标碗或抽屉在执行中被移动时，",
        "VLA-Corrector 的目标是停止信任过时动作块，",
        "并恢复执行。",
    ],
    "closing": [
        "代码和项目主页已经公开。",
        "论文和 arXiv 链接即将发布。",
    ],
}


def wrap_caption(text: str, lang: str, scene_key: str) -> list[str]:
    if lang == "en":
        return textwrap.wrap(text, width=74, break_long_words=False, break_on_hyphens=False)
    if scene_key in ZH_CAPTION_LINES:
        return ZH_CAPTION_LINES[scene_key]

    lines: list[str] = []
    line = ""
    for tok in zh_tokens(text):
        if tok == " ":
            if line and not line.endswith(" "):
                line += " "
            continue
        candidate = f"{line}{tok}"
        if line and text_units(candidate) > 42:
            lines.append(line.strip())
            line = tok
        else:
            line = candidate
    if line.strip():
        lines.append(line.strip())
    return lines


def fmt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_storyboard(en_durs, zh_durs):
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    md = [
        "# VLA-Corrector Video Overview",
        "",
        "Generated academic teaser assets for the project page.",
        "",
        "| Scene | Visual source | English narration | Chinese narration |",
        "| --- | --- | --- | --- |",
    ]
    visual = {
        "title": "Project logo and title card",
        "problem": "Paper teaser figure: open-loop versus closed-loop execution",
        "method": "Paper method overview figure",
        "corrector": "Paper truncation phase analysis plus corrector cards",
        "results": "Paper result summary, Pareto figure, and qualitative recovery figure",
        "demos": "Three compressed real-robot demonstration clips",
        "closing": "Project URL and paper status",
    }
    for s in SCENES:
        md.append(f"| {s.key} | {visual[s.key]} | {s.en_voice} | {s.zh_voice} |")
    md += [
        "",
        f"English duration: {sum(en_durs):.1f}s",
        f"Chinese duration: {sum(zh_durs):.1f}s",
        "",
        "The videos use paper figures and project-owned real-robot clips only. Voice-over is generated with edge-tts.",
    ]
    (SCRIPT_DIR / "storyboard.md").write_text("\n".join(md), encoding="utf-8")
    (SCRIPT_DIR / "timings.json").write_text(
        json.dumps({"en": en_durs, "zh": zh_durs}, indent=2), encoding="utf-8"
    )


def poster() -> None:
    assets = load_assets()
    img = slide_title(assets, 0)
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    img.save(OUT_IMAGES / "vla_corrector_video_poster.png")
    img.save(OUT_IMAGES / "vla_corrector_video_poster.webp", quality=82)


def mux(lang: str, raw: Path, audio: Path) -> Path:
    OUT_VIDEOS.mkdir(parents=True, exist_ok=True)
    out = OUT_VIDEOS / f"vla_corrector_overview_{lang}.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(out),
        ]
    )
    return out


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    poster()
    en_audio, en_durs = synthesize_voice("en", SCENES)
    zh_audio, zh_durs = synthesize_voice("zh", SCENES)
    en_voice = concat_audio("en", en_audio, en_durs)
    zh_voice = concat_audio("zh", zh_audio, zh_durs)
    en_raw = render_video("en", en_durs)
    zh_raw = render_video("zh", zh_durs)
    write_srt("en", en_durs)
    write_srt("zh", zh_durs)
    en_out = mux("en", en_raw, en_voice)
    zh_out = mux("zh", zh_raw, zh_voice)
    write_storyboard(en_durs, zh_durs)
    print("Generated:")
    for p in [en_out, zh_out, OUT_SUBTITLES / "vla_corrector_overview_en.srt", OUT_SUBTITLES / "vla_corrector_overview_zh.srt"]:
        print(p, p.stat().st_size)


if __name__ == "__main__":
    main()
