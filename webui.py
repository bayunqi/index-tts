import base64
import html
import json
import os
import shutil
import sys
import threading
import time
import zipfile

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

# Serve .wav as "audio/wav" (Python defaults to "audio/x-wav"). Gradio only
# serves files inline when the mime type is in its XSS-safe set, which contains
# "audio/wav" but not "audio/x-wav"; without this, wavs go out as
# "application/octet-stream", which some gateways/proxies reject (e.g. ROW OG
# 4018) and which prevents in-browser playback when accessed through a proxy.
import mimetypes
mimetypes.add_type("audio/wav", ".wav")

import argparse
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=9003, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="[::]", help="Host to run the web UI on (use [::] to bind IPv6)")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="Auto")
MODE = 'local'
tts = IndexTTS2(model_dir=cmd_args.model_dir,
                cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
                use_fp16=cmd_args.fp16,
                use_deepspeed=cmd_args.deepspeed,
                use_cuda_kernel=cmd_args.cuda_kernel,
                )
# 支持的语言列表
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES_ALL = [i18n("与音色参考音频相同"),
                i18n("使用情感参考音频"),
                i18n("使用情感向量控制"),
                i18n("使用情感描述文本控制")]
EMO_CHOICES_OFFICIAL = EMO_CHOICES_ALL[:-1]  # skip experimental features
BATCH_REFERENCE_EXTENSIONS = (".wav", ".mp3")

os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)

MAX_LENGTH_TO_USE_SPEED = 70
example_cases = []
with open("examples/cases.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        if example.get("emo_audio",None):
            emo_audio_path = os.path.join("examples",example["emo_audio"])
        else:
            emo_audio_path = None

        example_cases.append([os.path.join("examples", example.get("prompt_audio", "sample_prompt.wav")),
                              EMO_CHOICES_ALL[example.get("emo_mode",0)],
                              example.get("text"),
                             emo_audio_path,
                             example.get("emo_weight",1.0),
                             example.get("emo_text",""),
                             example.get("emo_vec_1",0),
                             example.get("emo_vec_2",0),
                             example.get("emo_vec_3",0),
                             example.get("emo_vec_4",0),
                             example.get("emo_vec_5",0),
                             example.get("emo_vec_6",0),
                             example.get("emo_vec_7",0),
                             example.get("emo_vec_8",0),
                             ])

def get_example_cases(include_experimental = False):
    if include_experimental:
        return example_cases  # show every example

    # exclude emotion control mode 3 (emotion from text description)
    return [x for x in example_cases if x[1] != EMO_CHOICES_ALL[3]]

def format_glossary_markdown():
    """将词汇表转换为Markdown表格格式"""
    if not tts.normalizer.term_glossary:
        return i18n("暂无术语")

    lines = [f"| {i18n('术语')} | {i18n('中文读法')} | {i18n('英文读法')} |"]
    lines.append("|---|---|---|")

    for term, reading in tts.normalizer.term_glossary.items():
        zh = reading.get("zh", "") if isinstance(reading, dict) else reading
        en = reading.get("en", "") if isinstance(reading, dict) else reading
        lines.append(f"| {term} | {zh} | {en} |")

    return "\n".join(lines)

def audio_data_uri(path):
    """Read an audio file and return it as a base64 `data:` URI."""
    mime = mimetypes.guess_type(path)[0] or "audio/wav"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def inline_audio_html(path, download_name=None):
    """Build an inline <audio> player (optionally with a download link) that
    embeds the audio as a base64 data URI, so it is delivered inside the page
    response instead of a separate /gradio_api/file= request that some proxies
    /gateways (e.g. ROW OG 4018) reject."""
    if not path or not os.path.isfile(path):
        return ""
    data_uri = audio_data_uri(path)
    player = f'<audio controls style="width:100%" src="{data_uri}"></audio>'
    if download_name:
        player += (f'<div style="margin-top:8px">'
                   f'<a download="{download_name}" href="{data_uri}">⬇ {download_name}</a></div>')
    return player

def file_data_uri(path, mime):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def inline_download_html(path, download_name, label=None, mime="application/zip"):
    if not path or not os.path.isfile(path):
        return ""
    data_uri = file_data_uri(path, mime)
    link_text = html.escape(label or download_name)
    return (f'<a download="{html.escape(download_name)}" href="{data_uri}" '
            f'style="display:inline-block;margin-top:8px">{link_text}</a>')

def unique_output_name(name, used_names, suffix="_output"):
    stem, _ = os.path.splitext(os.path.basename(name))
    candidate = f"{stem}{suffix}.wav"
    index = 2
    while candidate in used_names:
        candidate = f"{stem}{suffix}_{index}.wav"
        index += 1
    used_names.add(candidate)
    return candidate

def unique_reference_name(name, used_names):
    base = os.path.basename(name)
    stem, ext = os.path.splitext(base)
    candidate = f"{stem}{ext.lower()}"
    index = 2
    while candidate in used_names:
        candidate = f"{stem}_{index}{ext.lower()}"
        index += 1
    used_names.add(candidate)
    return candidate

def extract_reference_audios_from_zip(zip_path, extract_dir):
    wav_entries = []
    used_names = set()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            base = os.path.basename(member.filename)
            if not base or base.startswith(".") or not base.lower().endswith(BATCH_REFERENCE_EXTENSIONS):
                continue
            ref_name = unique_reference_name(base, used_names)
            ref_path = os.path.join(extract_dir, ref_name)
            with zf.open(member) as src, open(ref_path, "wb") as dst:
                dst.write(src.read())
            wav_entries.append((ref_path, base))
    return wav_entries

def normalize_uploaded_paths(paths):
    if not paths:
        return []
    if isinstance(paths, (str, os.PathLike)):
        return [str(paths)]
    normalized = []
    for item in paths:
        if isinstance(item, (str, os.PathLike)):
            normalized.append(str(item))
        elif isinstance(item, dict) and item.get("path"):
            normalized.append(item["path"])
    return normalized

def collect_wavs_from_uploads(paths, refs_dir):
    wav_entries = []
    used_names = set()
    for path in normalize_uploaded_paths(paths):
        if not path or not os.path.isfile(path) or not path.lower().endswith(BATCH_REFERENCE_EXTENSIONS):
            continue
        base = os.path.basename(path)
        ref_name = unique_reference_name(base, used_names)
        ref_path = os.path.join(refs_dir, ref_name)
        shutil.copyfile(path, ref_path)
        wav_entries.append((ref_path, base))
    return wav_entries

def batch_mode_to_index(batch_clone_mode):
    if isinstance(batch_clone_mode, int):
        return batch_clone_mode
    if hasattr(batch_clone_mode, "value"):
        return batch_clone_mode.value
    if batch_clone_mode == i18n("使用情感向量控制"):
        return 1
    return 0

def gen_batch(batch_folder, batch_zip, batch_clone_mode, text,
              batch_emo_weight,
              batch_vec1, batch_vec2, batch_vec3, batch_vec4,
              batch_vec5, batch_vec6, batch_vec7, batch_vec8,
              max_text_tokens_per_segment=120,
              *args, progress=gr.Progress()):
    if not batch_folder and not batch_zip:
        return gr.update(value="<p style='color:#c00'>请先上传包含 wav/mp3 的文件夹或 zip 包。</p>", visible=True)
    if not text or not text.strip():
        return gr.update(value="<p style='color:#c00'>请输入目标文本。</p>", visible=True)
    if batch_zip and not zipfile.is_zipfile(batch_zip):
        return gr.update(value="<p style='color:#c00'>批量 zip 输入需要是有效 zip 包。</p>", visible=True)

    task_name = f"batch_{int(time.time())}"
    task_dir = os.path.join("outputs", "tasks", task_name)
    refs_dir = os.path.join(task_dir, "refs")
    outputs_dir = os.path.join(task_dir, "outputs")
    os.makedirs(refs_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    ref_entries = collect_wavs_from_uploads(batch_folder, refs_dir)
    if not ref_entries and batch_zip:
        ref_entries = extract_reference_audios_from_zip(batch_zip, refs_dir)
    if not ref_entries:
        return gr.update(value="<p style='color:#c00'>没有找到 wav/mp3 文件。</p>", visible=True)

    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }

    batch_clone_mode = batch_mode_to_index(batch_clone_mode)
    use_vectors = batch_clone_mode == 1
    vec = None
    if use_vectors:
        vec = [batch_vec1, batch_vec2, batch_vec3, batch_vec4,
               batch_vec5, batch_vec6, batch_vec7, batch_vec8]
        vec = tts.normalize_emo_vec(vec, apply_bias=True)

    generated = []
    failed = []
    used_output_names = set()
    total = len(ref_entries)
    old_tts_progress = tts.gr_progress
    tts.gr_progress = None
    try:
        for idx, (ref_path, original_name) in enumerate(ref_entries, start=1):
            output_name = unique_output_name(original_name, used_output_names)
            output_path = os.path.join(outputs_dir, output_name)
            try:
                progress((idx - 1, total), desc=f"样本 {idx}/{total}: {os.path.basename(ref_path)}")
                output = tts.infer(
                    spk_audio_prompt=ref_path,
                    text=text,
                    output_path=output_path,
                    emo_audio_prompt=None,
                    emo_alpha=batch_emo_weight,
                    emo_vector=vec,
                    use_emo_text=False,
                    emo_text=None,
                    use_random=False,
                    verbose=cmd_args.verbose,
                    max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                    **kwargs,
                )
                if output and os.path.isfile(output):
                    generated.append(output)
                else:
                    failed.append(os.path.basename(ref_path))
            except Exception as e:
                failed.append(f"{os.path.basename(ref_path)} ({e})")
                print(f"Batch generation failed for {ref_path}: {e}")
            finally:
                progress((idx, total), desc=f"样本 {idx}/{total}: {os.path.basename(ref_path)}")
    finally:
        tts.gr_progress = old_tts_progress

    if not generated:
        return gr.update(value="<p style='color:#c00'>批量生成失败，没有可打包的输出。</p>", visible=True)

    zip_output_path = os.path.join(task_dir, f"{task_name}_outputs.zip")
    with zipfile.ZipFile(zip_output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in generated:
            zf.write(path, arcname=os.path.basename(path))

    summary = [
        f"<p>批量生成完成：成功 {len(generated)} / {total}</p>",
        inline_download_html(
            zip_output_path,
            os.path.basename(zip_output_path),
            label=f"⬇ 下载结果 zip ({os.path.basename(zip_output_path)})",
        ),
    ]
    if failed:
        failed_items = "".join(f"<li>{html.escape(item)}</li>" for item in failed)
        summary.append(f"<details style='margin-top:8px'><summary>失败 {len(failed)} 个</summary><ul>{failed_items}</ul></details>")
    summary.append(f"<p style='font-size:12px;color:#666'>服务器路径：{html.escape(os.path.abspath(outputs_dir))}</p>")
    return gr.update(value="\n".join(summary), visible=True)

def gen_single(emo_control_method,prompt, text,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text,emo_random,
               max_text_tokens_per_segment=120,
                *args, progress=gr.Progress()):
    output_path = None
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    # set gradio progress
    tts.gr_progress = progress
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        # "typical_sampling": bool(typical_sampling),
        # "typical_mass": float(typical_mass),
    }
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:  # emotion from speaker
        emo_ref_path = None  # remove external reference audio
    if emo_control_method == 1:  # emotion from reference audio
        pass
    if emo_control_method == 2:  # emotion from custom vectors
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec = tts.normalize_emo_vec(vec, apply_bias=True)
    else:
        # don't use the emotion vector inputs for the other modes
        vec = None

    if emo_text == "":
        # erase empty emotion descriptions; `infer()` will then automatically use the main prompt
        emo_text = None

    print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")
    output = tts.infer(spk_audio_prompt=prompt, text=text,
                       output_path=output_path,
                       emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                       emo_vector=vec,
                       use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                       verbose=cmd_args.verbose,
                       max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                       **kwargs)
    if not output or not os.path.isfile(output):
        return gr.update(value="<p style='color:#c00'>生成失败 / Generation failed</p>", visible=True)
    return gr.update(value=inline_audio_html(output, download_name=os.path.basename(output)), visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

def create_warning_message(warning_text):
    return gr.HTML(f"<div style=\"padding: 0.5em 0.8em; border-radius: 0.5em; background: #ffa87d; color: #000; font-weight: bold\">{html.escape(warning_text)}</div>")

def create_experimental_warning_message():
    return create_warning_message(i18n('提示：此功能为实验版，结果尚不稳定，我们正在持续优化中。'))

# Hide gr.Audio's built-in player for the reference input: it loads audio via
# /gradio_api/file=, which the proxy gateway blocks. The upload/microphone
# controls (SelectSource) are siblings of the player and stay visible; playback
# is handled by the inline data-URI preview below the component instead.
_REF_AUDIO_CSS = """
.ref-audio-noplayer .component-wrapper,
.ref-audio-noplayer audio { display: none !important; }
.ref-audio-noplayer .audio-container { height: auto !important; }
"""

with gr.Blocks(title="IndexTTS Demo", css=_REF_AUDIO_CSS) as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h2>
<p align="center">
<a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
</p>
    ''')

    with gr.Tab(i18n("音频生成")):
        with gr.Row():
            os.makedirs("prompts",exist_ok=True)
            with gr.Column():
                prompt_audio = gr.Audio(label=i18n("音色参考音频"),key="prompt_audio",
                                        elem_classes=["ref-audio-noplayer"],
                                        sources=["upload","microphone"],type="filepath")
                # Inline player so the uploaded/recorded reference can be auditioned
                # in the browser without a /gradio_api/file= fetch (blocked by the proxy).
                prompt_audio_preview = gr.HTML(visible=False, key="prompt_audio_preview")
            prompt_list = os.listdir("prompts")
            default = ''
            if prompt_list:
                default = prompt_list[0]
            with gr.Column():
                input_text_single = gr.TextArea(label=i18n("文本"),key="input_text_single", placeholder=i18n("请输入目标文本"), info=f"{i18n('当前模型版本')}{tts.model_version or '1.0'}")
                gen_button = gr.Button(i18n("生成语音"), key="gen_button",interactive=True)
            # Inline the result as a base64 data URI inside an HTML player so the
            # audio rides in the page response (SSE/JSON) instead of a separate
            # /gradio_api/file= binary request. Some proxies/gateways (e.g. ROW OG
            # 4018) reject the binary audio response, which would otherwise break
            # in-browser playback/download when accessed through the proxy.
            output_audio = gr.HTML(label=i18n("生成结果"), show_label=True,
                                   visible=True, key="output_audio")

        with gr.Row():
            experimental_checkbox = gr.Checkbox(label=i18n("显示实验功能"), value=False)
            glossary_checkbox = gr.Checkbox(label=i18n("开启术语词汇读音"), value=tts.normalizer.enable_glossary)
        with gr.Accordion(i18n("功能设置")):
            # 情感控制选项部分
            with gr.Row():
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES_OFFICIAL,
                    type="index",
                    value=EMO_CHOICES_OFFICIAL[0],label=i18n("情感控制方式"))
                # we MUST have an extra, INVISIBLE list of *all* emotion control
                # methods so that gr.Dataset() can fetch ALL control mode labels!
                # otherwise, the gr.Dataset()'s experimental labels would be empty!
                emo_control_method_all = gr.Radio(
                    choices=EMO_CHOICES_ALL,
                    type="index",
                    value=EMO_CHOICES_ALL[0], label=i18n("情感控制方式"),
                    visible=False)  # do not render
        # 情感参考音频部分
        with gr.Group(visible=False) as emotion_reference_group:
            with gr.Row():
                emo_upload = gr.Audio(label=i18n("上传情感参考音频"), type="filepath")

        # 情感随机采样
        with gr.Row(visible=False) as emotion_randomize_group:
            emo_random = gr.Checkbox(label=i18n("情感随机采样"), value=False)

        # 情感向量控制部分
        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label=i18n("喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec2 = gr.Slider(label=i18n("怒"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec3 = gr.Slider(label=i18n("哀"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec4 = gr.Slider(label=i18n("惧"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                with gr.Column():
                    vec5 = gr.Slider(label=i18n("厌恶"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec6 = gr.Slider(label=i18n("低落"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec7 = gr.Slider(label=i18n("惊喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec8 = gr.Slider(label=i18n("平静"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)

        with gr.Group(visible=False) as emo_text_group:
            create_experimental_warning_message()
            with gr.Row():
                emo_text = gr.Textbox(label=i18n("情感描述文本"),
                                      placeholder=i18n("请输入情绪描述（或留空以自动使用目标文本作为情绪描述）"),
                                      value="",
                                      info=i18n("例如：委屈巴巴、危险在悄悄逼近"))

        with gr.Row(visible=False) as emo_weight_group:
            emo_weight = gr.Slider(label=i18n("情感权重"), minimum=0.0, maximum=1.0, value=0.65, step=0.01)

        # 术语词汇表管理
        with gr.Accordion(i18n("自定义术语词汇读音"), open=False, visible=tts.normalizer.enable_glossary) as glossary_accordion:
            gr.Markdown(i18n("自定义个别专业术语的读音"))
            with gr.Row():
                with gr.Column(scale=1):
                    glossary_term = gr.Textbox(
                        label=i18n("术语"),
                        placeholder="IndexTTS2",
                    )
                    glossary_reading_zh = gr.Textbox(
                        label=i18n("中文读法"),
                        placeholder="Index T-T-S 二",
                    )
                    glossary_reading_en = gr.Textbox(
                        label=i18n("英文读法"),
                        placeholder="Index T-T-S two",
                    )
                    btn_add_term = gr.Button(i18n("添加术语"), scale=1)
                with gr.Column(scale=2):
                    glossary_table = gr.Markdown(
                        value=format_glossary_markdown()
                    )

        with gr.Accordion(i18n("高级生成参数设置"), open=False, visible=True) as advanced_settings_group:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"**{i18n('GPT2 采样设置')}** _{i18n('参数会影响音频多样性和生成速度详见')} [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info=i18n("是否进行采样"))
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info=i18n("生成Token最大数量，过小导致音频被截断"), key="max_mel_tokens")
                    # with gr.Row():
                    #     typical_sampling = gr.Checkbox(label="typical_sampling", value=False, info="不建议使用")
                    #     typical_mass = gr.Slider(label="typical_mass", value=0.9, minimum=0.0, maximum=1.0, step=0.1)
                with gr.Column(scale=2):
                    gr.Markdown(f'**{i18n("分句设置")}** _{i18n("参数会影响音频质量和生成速度")}_')
                    with gr.Row():
                        initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                        max_text_tokens_per_segment = gr.Slider(
                            label=i18n("分句最大Token数"), value=initial_value, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_segment",
                            info=i18n("建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高"),
                        )
                    with gr.Accordion(i18n("预览分句结果"), open=True) as segments_settings:
                        segments_preview = gr.Dataframe(
                            headers=[i18n("序号"), i18n("分句内容"), i18n("Token数")],
                            key="segments_preview",
                            wrap=True,
                        )
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                # typical_sampling, typical_mass,
            ]

        with gr.Accordion(i18n("批量生成"), open=False):
            with gr.Row():
                with gr.Column():
                    batch_folder = gr.File(
                        label=i18n("批量参考音频文件夹（包含 wav/mp3 文件）"),
                        file_count="directory",
                        file_types=[".wav", ".mp3"],
                        type="filepath",
                    )
                    batch_zip = gr.File(
                        label=i18n("批量参考音频 zip（文件夹上传不可用时使用）"),
                        file_count="single",
                        file_types=[".zip"],
                        type="filepath",
                    )
                with gr.Column():
                    batch_clone_mode = gr.Radio(
                        choices=[
                            i18n("与音色参考音频相同"),
                            i18n("使用情感向量控制"),
                        ],
                        type="index",
                        value=i18n("与音色参考音频相同"),
                        label=i18n("批量 Clone 模式"),
                    )
                    batch_gen_button = gr.Button(i18n("批量生成"), interactive=True)
            with gr.Group(visible=False) as batch_emotion_vector_group:
                with gr.Row():
                    batch_emo_weight = gr.Slider(label=i18n("情感权重"), minimum=0.0, maximum=1.0, value=0.65, step=0.01)
                with gr.Row():
                    with gr.Column():
                        batch_vec1 = gr.Slider(label=i18n("喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        batch_vec2 = gr.Slider(label=i18n("怒"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        batch_vec3 = gr.Slider(label=i18n("哀"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        batch_vec4 = gr.Slider(label=i18n("惧"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    with gr.Column():
                        batch_vec5 = gr.Slider(label=i18n("厌恶"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        batch_vec6 = gr.Slider(label=i18n("低落"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        batch_vec7 = gr.Slider(label=i18n("惊喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        batch_vec8 = gr.Slider(label=i18n("平静"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
            batch_output = gr.HTML(label=i18n("批量结果"), show_label=True,
                                   visible=True, key="batch_output")

    def on_prompt_audio_change(path):
        # Render an inline player so the uploaded/recorded reference can be
        # auditioned in the browser without a /gradio_api/file= fetch.
        if not path or not os.path.isfile(path):
            return gr.update(value="", visible=False)
        return gr.update(value=inline_audio_html(path), visible=True)

    prompt_audio.change(on_prompt_audio_change,
                        inputs=[prompt_audio],
                        outputs=[prompt_audio_preview])
    prompt_audio.clear(lambda: gr.update(value="", visible=False),
                       outputs=[prompt_audio_preview])

    def on_batch_clone_mode_change(batch_clone_mode):
        return gr.update(visible=batch_mode_to_index(batch_clone_mode) == 1)

    batch_clone_mode.change(
        on_batch_clone_mode_change,
        inputs=[batch_clone_mode],
        outputs=[batch_emotion_vector_group],
    )

    def on_input_text_change(text, max_text_tokens_per_segment):
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)

            segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
            data = []
            for i, s in enumerate(segments):
                segment_str = ''.join(s)
                tokens_count = len(s)
                data.append([i, segment_str, tokens_count])
            return {
                segments_preview: gr.update(value=data, visible=True, type="array"),
            }
        else:
            df = pd.DataFrame([], columns=[i18n("序号"), i18n("分句内容"), i18n("Token数")])
            return {
                segments_preview: gr.update(value=df),
            }

    # 术语词汇表事件处理函数
    def on_add_glossary_term(term, reading_zh, reading_en):
        """添加术语到词汇表并自动保存"""
        term = term.rstrip()
        reading_zh = reading_zh.rstrip()
        reading_en = reading_en.rstrip()

        if not term:
            gr.Warning(i18n("请输入术语"))
            return gr.update()
            
        if not reading_zh and not reading_en:
            gr.Warning(i18n("请至少输入一种读法"))
            return gr.update()
        

        # 构建读法数据
        if reading_zh and reading_en:
            reading = {"zh": reading_zh, "en": reading_en}
        elif reading_zh:
            reading = {"zh": reading_zh}
        elif reading_en:
            reading = {"en": reading_en}
        else:
            reading = reading_zh or reading_en

        # 添加到词汇表
        tts.normalizer.term_glossary[term] = reading

        # 自动保存到文件
        try:
            tts.normalizer.save_glossary_to_yaml(tts.glossary_path)
            gr.Info(i18n("词汇表已更新"), duration=1)
        except Exception as e:
            gr.Error(i18n("保存词汇表时出错"))
            print(f"Error details: {e}")
            return gr.update()

        # 更新Markdown表格
        return gr.update(value=format_glossary_markdown())
        

    def on_method_change(emo_control_method):
        if emo_control_method == 1:  # emotion reference audio
            return (gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 2:  # emotion vectors
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 3:  # emotion text description
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True)
                    )
        else:  # 0: same as speaker voice
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )

    emo_control_method.change(on_method_change,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group,
                 emotion_randomize_group,
                 emotion_vector_group,
                 emo_text_group,
                 emo_weight_group]
    )

    def on_experimental_change(is_experimental, current_mode_index):
        # 切换情感控制选项
        new_choices = EMO_CHOICES_ALL if is_experimental else EMO_CHOICES_OFFICIAL
        # if their current mode selection doesn't exist in new choices, reset to 0.
        # we don't verify that OLD index means the same in NEW list, since we KNOW it does.
        new_index = current_mode_index if current_mode_index < len(new_choices) else 0

        return gr.update(choices=new_choices, value=new_choices[new_index])

    experimental_checkbox.change(
        on_experimental_change,
        inputs=[experimental_checkbox, emo_control_method],
        outputs=[emo_control_method]
    )

    def on_glossary_checkbox_change(is_enabled):
        """控制术语词汇表的可见性"""
        tts.normalizer.enable_glossary = is_enabled
        return gr.update(visible=is_enabled)

    glossary_checkbox.change(
        on_glossary_checkbox_change,
        inputs=[glossary_checkbox],
        outputs=[glossary_accordion]
    )

    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    max_text_tokens_per_segment.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button])

    def on_demo_load():
        """页面加载时重新加载glossary数据"""
        try:
            tts.normalizer.load_glossary_from_yaml(tts.glossary_path)
        except Exception as e:
            gr.Error(i18n("加载词汇表时出错"))
            print(f"Failed to reload glossary on page load: {e}")
        return gr.update(value=format_glossary_markdown())

    # 术语词汇表事件绑定
    btn_add_term.click(
        on_add_glossary_term,
        inputs=[glossary_term, glossary_reading_zh, glossary_reading_en],
        outputs=[glossary_table]
    )

    # 页面加载时重新加载glossary
    demo.load(
        on_demo_load,
        inputs=[],
        outputs=[glossary_table]
    )

    gen_button.click(gen_single,
                     inputs=[emo_control_method,prompt_audio, input_text_single, emo_upload, emo_weight,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                             emo_text,emo_random,
                             max_text_tokens_per_segment,
                             *advanced_params,
                     ],
                     outputs=[output_audio])

    batch_gen_button.click(
        gen_batch,
        inputs=[
            batch_folder, batch_zip, batch_clone_mode, input_text_single,
            batch_emo_weight,
            batch_vec1, batch_vec2, batch_vec3, batch_vec4,
            batch_vec5, batch_vec6, batch_vec7, batch_vec8,
            max_text_tokens_per_segment,
            *advanced_params,
        ],
        outputs=[batch_output],
    )



if __name__ == "__main__":
    demo.queue(20)
    # For IPv6, pass a bracketed host (e.g. "[::]"); Gradio strips the brackets
    # when binding and builds a valid health-check URL (http://[::]:port/).
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port)
