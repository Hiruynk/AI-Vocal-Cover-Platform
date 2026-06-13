import streamlit as st
import os
import sys
import subprocess
import glob
import shutil
import time
import uuid
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import parselmouth
import zipfile
import itertools

# pydub 與靜音偵測
from pydub import AudioSegment
from pydub.silence import split_on_silence
import torch

import psutil   # 新增：用於偵測 RAM 記憶體
import gc

import soundfile as sf
from audiosr import build_model, super_resolution

# =========================================================
# 【新增匯入：擴散模型與高階 DSP 混音】
# =========================================================


from pedalboard import Pedalboard, Compressor, HighpassFilter, Reverb, Chorus
from pedalboard.io import AudioFile


# =========================================================
# 【系統層級設定與 FFmpeg Shared DLL 強行注入】
# =========================================================
st.set_page_config(page_title="極致 AI 翻唱工作站", layout="wide")

# 1. 【破解 Python 3.8+ 的 DLL 資安限制】
shared_ffmpeg_path = glob.glob(r"C:\ffmpeg_shared\**\bin", recursive=True)
if shared_ffmpeg_path:
    ffmpeg_bin_dir = shared_ffmpeg_path[0]
    os.environ["PATH"] = ffmpeg_bin_dir + os.path.pathsep + os.environ.get("PATH", "")
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(ffmpeg_bin_dir)

# 2. 保留 winget EXE 注入 (防呆)
winget_ffmpeg_pattern = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages", "**", "ffmpeg.exe")
found_ffmpegs = glob.glob(winget_ffmpeg_pattern, recursive=True)
if found_ffmpegs:
    os.environ["PATH"] = os.path.dirname(found_ffmpegs[0]) + os.path.pathsep + os.environ["PATH"]

# =========================================================
# 【app2.py 全域圖表設定與常數】
# =========================================================
plt.rcParams['font.family'] = [
    'MS Gothic', 'Yu Gothic', 'Hiragino Sans', 
    'Noto Sans CJK JP', 'Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False 

MIN_MIDI = 21
MAX_MIDI = 108
MIDI_KEYS = np.arange(MIN_MIDI, MAX_MIDI + 1)
OUTLIER_THRESHOLD = 25 

# =========================================================
# 【通用終端機輸出擷取函數】
# =========================================================
def run_and_stream(cmd, cwd, log_box):
    process = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
        text=True, encoding="utf-8", errors="replace", bufsize=1
    )
    log_text = ""
    for line in iter(process.stdout.readline, ''):
        log_text += line
        display_text = log_text[-2000:] if len(log_text) > 2000 else log_text
        log_box.code(display_text, language="text")
    process.stdout.close()
    return process.wait()

# =========================================================
# 【資源管理：記憶體與 VRAM 監控告警】
# =========================================================
def check_and_warn_memory_usage():
    """偵測 RAM 與 VRAM 佔用率，過高時在終端機發出紅字警告並嘗試回收垃圾"""
    # 1. 檢查系統記憶體 (RAM)
    ram_percent = psutil.virtual_memory().percent
    if ram_percent > 85.0:
        sys.stderr.write(f"\n\033[91m[🚨 警告] 系統記憶體 (RAM) 佔用率已達 {ram_percent}%！隨時可能崩潰！\033[0m\n")
        gc.collect() # 強制回收未使用的 Python 記憶體

    # 2. 檢查顯示卡記憶體 (VRAM)
    if torch.cuda.is_available():
        free_vram, total_vram = torch.cuda.mem_get_info()
        vram_percent = ((total_vram - free_vram) / total_vram) * 100
        if vram_percent > 85.0:
            sys.stderr.write(f"\n\033[91m[🚨 警告] 5070 Ti 顯示記憶體 (VRAM) 佔用達 {vram_percent:.1f}%！可能導致 CUDA Out of Memory！\033[0m\n")
            torch.cuda.empty_cache() # 強制清空 PyTorch 快取

# 每次網頁刷新或按鈕觸發時，自動執行一次檢查
check_and_warn_memory_usage()

# =========================================================
# 【修正：高階 AI 人聲後處理管線】
# =========================================================
def apply_pre_diffusion_denoise(input_path, output_path):
    """在進入 AudioSR 前，先進行降噪與高頻截斷，防止電音偽影被擴散模型放大"""
    from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, NoiseGate
    from pedalboard.io import AudioFile
    
    # 建立去電音與基礎降噪濾波鏈
    board = Pedalboard([
        # 1. 消除輕微底噪與 RVC 運算漏音
        NoiseGate(threshold_db=-45, ratio=2.0, attack_ms=1.0, release_ms=100),
        # 2. 砍除 80Hz 以下無用轟鳴，防止低頻干擾擴散模型
        HighpassFilter(cutoff_frequency_hz=80),
        # 3. 核心：RVC 金屬電音通常在 10kHz 以上，我們先強制削弱它，
        # 後面讓 AudioSR 無中生有重新「畫出」真人的高頻泛音
        LowpassFilter(cutoff_frequency_hz=10000)
    ])
    
    with AudioFile(input_path) as f:
        with AudioFile(output_path, 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(f.samplerate)
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)

def run_audiosr_restoration(input_path, output_path):
    """調用 AudioSR 擴散模型消除 HiFi-GAN 金屬音與偽影 (修正長度偏移同步問題)"""
    import soundfile as sf
    import numpy as np
    import torch
    import gc
    from audiosr import build_model, super_resolution
    
    data, sr = sf.read(input_path)
    audiosr_model = build_model(model_name="basic", device="cuda:0")
    
    chunk_duration = 5  
    chunk_samples = chunk_duration * sr
    total_samples = len(data)
    
    import os
    temp_chunk_in = input_path + "_tmp_chunk_in.wav"
    restored_chunks = []
    
    for i in range(0, total_samples, chunk_samples):
        chunk_data = data[i:i + chunk_samples]
        sf.write(temp_chunk_in, chunk_data, sr)
        
        waveform = super_resolution(
            audiosr_model, 
            temp_chunk_in,
            guidance_scale=3.5,  
            ddim_steps=50
        )
        
        wav = np.squeeze(waveform) 
        if wav.ndim == 2 and wav.shape[0] < wav.shape[1]:
            wav = wav.T 
            
        # 🚀 核心同步修正：強制將輸出的長度剪裁至與輸入完全一致！
        expected_out_len = int(len(chunk_data) * 48000 / sr)
        if len(wav) > expected_out_len:
            # 切除 AudioSR 擅自加入的 Padding 延遲
            wav = wav[:expected_out_len]
        elif len(wav) < expected_out_len:
            # 若有短缺則用靜音補齊，確保時序絕對鎖定
            padding = np.zeros((expected_out_len - len(wav),) + wav.shape[1:]) if wav.ndim > 1 else np.zeros(expected_out_len - len(wav))
            wav = np.concatenate([wav, padding])
            
        restored_chunks.append(wav)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    if restored_chunks[0].ndim == 2:
        final_wav = np.vstack(restored_chunks)
    else:
        final_wav = np.concatenate(restored_chunks)
        
    sf.write(output_path, final_wav, 48000)
    
    if os.path.exists(temp_chunk_in):
        try: os.remove(temp_chunk_in)
        except Exception: pass
            
    del audiosr_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def apply_studio_dsp(input_path, output_path):
    """調用 Pedalboard 進行廣播級動態塑形與自動混響"""
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80),
        Compressor(threshold_db=-15, ratio=4.0, attack_ms=2.0, release_ms=50),
        Compressor(threshold_db=-20, ratio=2.0, attack_ms=15.0, release_ms=250),
        Chorus(rate_hz=1.0, depth=0.1, centre_delay_ms=2.0, mix=0.15),
        Reverb(room_size=0.5, damping=0.5, wet_level=0.25, dry_level=0.85)
    ])
    
    with AudioFile(input_path) as f:
        with AudioFile(output_path, 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(f.samplerate)
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)

# =========================================================
# 【AI 模型常數設定】
# =========================================================
MODEL_ROFORMER = "model_bs_roformer_ep_317_sdr_12.9755.ckpt" 
MODEL_BVE = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"                      # 新增：和音消除專用模型
MODEL_DEREVERB = "UVR-DeEcho-DeReverb.pth"      
MODEL_DEMUCS = "htdemucs.yaml"

# =========================================================
# 【Tab 4 專用：Demucs 快速人聲分離】
# =========================================================
def run_demucs_fast(input_file_path, output_dir, task_id):
    """
    調用 HTDemucs 進行極速的 4 軌分離 (人聲/貝斯/鼓/其他)。
    專注於速度，不求完美無瑕的音質，只求快速拿到可供音高分析的乾聲。
    """
    temp_dir = os.path.join(output_dir, f"demucs_{task_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    venv_scripts = os.path.join(current_dir, "venv", "Scripts")
    audio_separator_exe = os.path.join(venv_scripts, "audio-separator.exe")
    cli_cmd = audio_separator_exe if os.path.exists(audio_separator_exe) else "audio-separator"

    cmd = [
        cli_cmd,
        input_file_path,
        "--model_filename", MODEL_DEMUCS,
        "--output_dir", temp_dir,
        "--output_format", "WAV",
        "--single_stem", "Vocals", # 只輸出人聲軌，節省 I/O 時間
        "--log_level", "INFO" 
    ]
    
    try:
        subprocess.run(
            cmd, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except Exception as e:
        raise Exception(f"Demucs 快速分離失敗: {e}")
        
    try:
        # Demucs 輸出檔名通常會包含 "(Vocals)"
        all_wavs = glob.glob(os.path.join(temp_dir, "*.wav"))
        vocal_file = next((f for f in all_wavs if "(Vocals)" in f), all_wavs[0])
        return vocal_file
    except Exception:
        raise Exception("Demucs 執行完畢，但找不到人聲輸出檔案。")

def run_uvr5_safely(input_file_path, output_dir, model_name, task_id):
    """
    呼叫最強開源庫 audio-separator。
    自動下載並調用指定的 BS-Roformer 或 VR 模型進行物理級提純。
    """
    temp_dir = os.path.join(output_dir, f"uvr5_{task_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    venv_scripts = os.path.join(current_dir, "venv", "Scripts")
    
    # 修正 1：正確呼叫 audio-separator 的執行檔，而不是過時的 module 寫法
    audio_separator_exe = os.path.join(venv_scripts, "audio-separator.exe")
    if os.path.exists(audio_separator_exe):
        cli_cmd = audio_separator_exe
    else:
        # 退回使用系統全域變數
        cli_cmd = "audio-separator"

    # 在 cmd 陣列中，加入強制使用 GPU 的參數
    cmd = [
        cli_cmd,
        input_file_path,
        "--model_filename", model_name,
        "--output_dir", temp_dir,
        "--output_format", "WAV",
        # 👇 移除錯誤的參數，換成 DEBUG 模式來監控底層 GPU 呼叫狀態
        "--log_level", "DEBUG"
    ]
    
    try:
        # 強制攔截標準輸出 (stdout) 與錯誤輸出 (stderr)
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True, 
            encoding="utf-8", 
            errors="replace"
        )
    except subprocess.CalledProcessError as e:
        # 修正 2：確保發生崩潰時，印出所有可能的日誌
        st.error(f"❌ audio-separator 執行崩潰 (退出碼 {e.returncode})！")
        st.code(f"【錯誤日誌 STDERR】:\n{e.stderr}\n\n【標準輸出 STDOUT】:\n{e.stdout}", language="text")
        raise Exception("底層分離引擎崩潰，無法生成檔案。")
    except FileNotFoundError:
        st.error("❌ 找不到 `audio-separator` 執行檔！請確認是否已啟動虛擬環境並安裝 `pip install audio-separator[gpu]`")
        raise Exception("未安裝 audio-separator 套件。")
    
    # === 將原本尋找檔案的 try 區塊替換成這段 ===
    # === 將原本尋找檔案的 try 區塊替換成這段 ===
    try:
        all_wavs = glob.glob(os.path.join(temp_dir, "*.wav"))
        if len(all_wavs) < 2:
            raise Exception("引擎未產出足夠的輸出檔案。")
            
        # 根據模型動態辨識後綴
        if "DeReverb" in model_name:
            vocal_file = next((f for f in all_wavs if "(No Reverb)" in f), all_wavs[0])
            inst_file = next((f for f in all_wavs if "(Reverb)" in f and "(No Reverb)" not in f), all_wavs[1])
            
        elif "BVE" in model_name or "karaoke" in model_name.lower():
            # BVE/Karaoke 模型在 audio-separator 預設輸出：(Vocals) 為主唱，(Instrumental) 為和音
            backing = next((f for f in all_wavs if "(Backing)" in f or "(Secondary)" in f or "(Instrumental)" in f), None)
            
            if backing:
                inst_file = backing
                vocal_file = next((f for f in all_wavs if f != backing), all_wavs[0])
            else:
                # 防呆：確保不會因為字母排序 I (Instrumental) 在 V (Vocals) 前面導致主副反轉
                all_wavs_sorted = sorted(all_wavs)
                inst_file = next((f for f in all_wavs_sorted if "Inst" in f or "inst" in f), all_wavs_sorted[0])
                vocal_file = next((f for f in all_wavs_sorted if f != inst_file), all_wavs_sorted[1])
                
        else:
            # Roformer 階段
            vocal_file = next((f for f in all_wavs if "(Vocals)" in f), all_wavs[0])
            inst_file = next((f for f in all_wavs if "(Instrumental)" in f), all_wavs[1])
            
        return vocal_file, inst_file
        
    except Exception as e:
        st.warning(f"⚠️ 檔案擷取失敗: {e}。完整日誌如下：")
        st.code(f"【STDERR】:\n{result.stderr}\n\n【STDOUT】:\n{result.stdout}", language="text")
        raise Exception("分離失敗：無法配對輸出檔案。")

           

# =========================================================
# 【全新：針對 RVC 優化的 VAD 靜音裁切引擎】
# =========================================================
def optimize_audio_for_rvc(input_path, output_path, min_silence_len=500, silence_thresh=-55):
    """
    精準裁切無聲底噪區段，並在人聲樂句之間補入標準的 0.5 秒絕對靜音。
    這能確保 RVC 訓練時獲得最乾淨的特徵，同時不會因為檔案過度連續而爆顯存。
    """
    audio = AudioSegment.from_file(input_path)
    
    # 尋找所有有人聲的區塊
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=400 # 保留頭尾 200ms 的呼吸聲，讓語氣自然
    )
    
    if not chunks:
        # 防呆：如果判斷全為靜音，則原樣輸出
        audio.export(output_path, format="wav")
        return
        
    # 0.5 秒的純淨靜音分隔符 (為 RVC 的切片機製造完美斷點)
    spacer = AudioSegment.silent(duration=500)
    
    processed_audio = chunks[0]
    for chunk in chunks[1:]:
        processed_audio += spacer + chunk
        
    processed_audio.export(output_path, format="wav")

# =========================================================
# 【音域分析函數 (Tab 4)】
# =========================================================
@st.cache_data(show_spinner=False)
def extract_pitch_distribution_fast(file_bytes, file_name):
    audio_io = io.BytesIO(file_bytes)
    y, sr = librosa.load(audio_io, sr=16000)
    sound = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=65.0, pitch_ceiling=2093.0)
    f0_voiced = pitch.selected_array['frequency']
    f0_voiced = f0_voiced[f0_voiced > 0]
    
    if len(f0_voiced) == 0:
        return np.zeros(len(MIDI_KEYS))
        
    midi_notes = np.round(librosa.hz_to_midi(f0_voiced))
    counts, _ = np.histogram(midi_notes, bins=np.arange(MIN_MIDI - 0.5, MAX_MIDI + 1.5, 1))
    return counts

def plot_piano_roll_distribution(counts, title):
    fig, (ax, ax_piano) = plt.subplots(
        2, 1, 
        figsize=(14, 6), 
        gridspec_kw={'height_ratios': [4, 0.8], 'hspace': 0.05}
    )
    
    cmap = plt.get_cmap('Purples')
    norm = mcolors.Normalize(vmin=0, vmax=np.max(counts) if np.max(counts) > 0 else 1)
    colors = cmap(norm(counts))
    
    ax.bar(MIDI_KEYS, counts, color=colors, edgecolor='black', linewidth=0.3, width=1.0, zorder=3)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    
    for midi in MIDI_KEYS:
        if midi % 12 in [1, 3, 6, 8, 10]:
            ax.axvspan(midi - 0.5, midi + 0.5, color='gray', alpha=0.04, zorder=1)

    ax.set_xlim(MIN_MIDI - 1, MAX_MIDI + 1)
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_ylabel('Pitch Frequency (Frames)', fontsize=11)
    ax.set_xticklabels([]) 
    
    ax_piano.set_xlim(MIN_MIDI - 1, MAX_MIDI + 1)
    ax_piano.set_ylim(0, 1)
    
    rect_base = plt.Rectangle((MIN_MIDI - 0.5, 0), MAX_MIDI - MIN_MIDI + 1, 1, facecolor='white', edgecolor='black', linewidth=0.8, zorder=1)
    ax_piano.add_patch(rect_base)
    
    for midi in range(MIN_MIDI, MAX_MIDI):
        is_curr_black = (midi % 12) in [1, 3, 6, 8, 10]
        is_next_black = ((midi + 1) % 12) in [1, 3, 6, 8, 10]
        y_max = 1.0 if (not is_curr_black and not is_next_black) else 0.35
        ax_piano.plot([midi + 0.5, midi + 0.5], [0, y_max], color='black', linewidth=0.6, zorder=2)
            
    for midi in MIDI_KEYS:
        if midi % 12 in [1, 3, 6, 8, 10]:
            rect_black = plt.Rectangle((midi - 0.4, 0.35), 0.8, 0.65, facecolor='#1a1a1a', edgecolor='black', linewidth=0.5, zorder=10)
            ax_piano.add_patch(rect_black)
            
    c_notes_midi = [24, 36, 48, 60, 72, 84, 96, 108] 
    c_notes_labels = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    
    ax_piano.set_xticks(c_notes_midi)
    ax_piano.set_xticklabels(c_notes_labels, fontweight='bold', fontsize=10)
    ax_piano.get_yaxis().set_visible(False) 
    
    for spine in ['top', 'left', 'right', 'bottom']:
        ax_piano.spines[spine].set_visible(False)
    ax_piano.tick_params(bottom=False)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax, ax_piano], aspect=25, pad=0.03)
    cbar.set_label('Occurrence Density', rotation=270, labelpad=15)
    
    plt.tight_layout()
    return fig

# =========================================================
# 【目錄初始化與狀態鎖 (含網頁刷新自動清理機制)】
# =========================================================
UPLOAD_DIR = "temp_audio"
TOOL_DIR = "temp_toolkit"
RVC_FULL_DIR = r"C:\Users\User\Desktop\Work\RVC20240604Nvidia50x0"

for d in [UPLOAD_DIR, TOOL_DIR]:
    os.makedirs(d, exist_ok=True)

# 💡 新增：網頁剛開啟或刷新 (F5) 時，清空所有輸入輸出暫存檔
if 'init_cleanup' not in st.session_state:
    st.session_state.init_cleanup = True
    
    # 清理 Page 1 & Page 4 的暫存音訊 (不包含 RVC 訓練資料)
    for f in glob.glob(os.path.join(UPLOAD_DIR, "*")):
        try:
            if os.path.isfile(f):
                os.remove(f)
            else:
                shutil.rmtree(f)
        except Exception:
            pass
        
    # 清理 Page 3 的分離暫存與 ZIP 包
    for f in glob.glob(os.path.join(TOOL_DIR, "*")):
        try:
            if os.path.isfile(f):
                os.remove(f)
            else:
                shutil.rmtree(f)
        except Exception:
            pass
        
    # 加入 _ = 防止 sys.stderr.write 的回傳值(字節數)被印在網頁上
    _ = sys.stderr.write("\n\033[93m[🔄 系統] 網頁已刷新，已自動清空 temp_audio 與 temp_toolkit 暫存資料夾以釋放空間。\033[0m\n")

if 'step1' not in st.session_state: st.session_state.step1 = False
if 'step2' not in st.session_state: st.session_state.step2 = False
if 'step3' not in st.session_state: st.session_state.step3 = False
if 'step4_preview' not in st.session_state: st.session_state.step4_preview = False
if 'history' not in st.session_state: st.session_state.history = []
if 'current_filename' not in st.session_state: st.session_state.current_filename = ""
if 'uid' not in st.session_state: st.session_state.uid = ""
if 'preview_path' not in st.session_state: st.session_state.preview_path = ""

with st.sidebar:
    st.header("💻 系統硬體狀態")
    if torch.cuda.is_available():
        st.success(f"GPU 已就緒: {torch.cuda.get_device_name(0)}")
    else:
        st.error("未偵測到 NVIDIA GPU，將使用 CPU 處理。")
    st.markdown("---")
    st.markdown("### 當前推論進度")
    st.checkbox("1. 檔案上傳", value=st.session_state.step1, disabled=True)
    st.checkbox("2. 人聲分離", value=st.session_state.step2, disabled=True)
    st.checkbox("3. 音色轉換", value=st.session_state.step3, disabled=True)

tab1, tab2, tab3, tab4 = st.tabs(["🎙️ 日常翻唱工作站", "⚙️ 建立新音色", "✂️ 訓練素材提取", "📊 音域分析可視化"])

# ==========================================
# 分頁 1：日常翻唱工作站 (升級擴散修復與 DSP 版)
# ==========================================
with tab1:
    st.title("🎙️ AI 歌聲轉換管線 (極致 DSP 版)")
    uploaded_file = st.file_uploader("1. 上傳目標歌曲 (MP3/WAV/FLAC)", type=["mp3", "wav", "flac"])
    
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.current_filename:
            st.session_state.current_filename = uploaded_file.name
            st.session_state.uid = uuid.uuid4().hex[:8] 
            st.session_state.step1 = True
            st.session_state.step2 = False
            st.session_state.step3 = False
            st.session_state.step4_preview = False
            st.session_state.preview_path = ""
            
        uid = st.session_state.uid
        file_ext = os.path.splitext(uploaded_file.name)[1]
        original_name = os.path.splitext(uploaded_file.name)[0]
        safe_basename = f"cover_{uid}" 
        file_path = os.path.join(UPLOAD_DIR, f"{safe_basename}{file_ext}")
        
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
        st.markdown("---")
        st.subheader(f"2. 極限人聲分離與 AI 去混響 (UVR5) - 當前處理: `{original_name}`")
        
        inst_path = os.path.join(UPLOAD_DIR, f"{safe_basename}_Instrumental.wav")
        vocals_backing_path = os.path.join(UPLOAD_DIR, f"{safe_basename}_Vocals_Backing.wav")
        vocals_wet_path = os.path.join(UPLOAD_DIR, f"{safe_basename}_Vocals_Wet.wav")
        vocals_dry_path = os.path.join(UPLOAD_DIR, f"{safe_basename}_Vocals_Dry.wav")
        vocals_effects_path = os.path.join(UPLOAD_DIR, f"{safe_basename}_Vocals_Reverb.wav")
        
        if not st.session_state.step2:
            if st.button("🚀 啟動 5070 Ti 頂級分離管線 (Roformer + BVE + DeReverb)", type="primary", key="infer_uvr5_btn"):
                with st.spinner("1/3 正在調用 BS-Roformer 剔除背景樂器..."):
                    try:
                        v_wet, inst = run_uvr5_safely(file_path, UPLOAD_DIR, MODEL_ROFORMER, f"{uid}_step1")
                        shutil.copy(inst, inst_path)
                        
                        with st.spinner("2/3 正在調用 BVE 模型剝離並保留原曲和音 (Backing Vocals)..."):
                            v_lead_wet, v_backing = run_uvr5_safely(v_wet, UPLOAD_DIR, MODEL_BVE, f"{uid}_step1b")
                            shutil.copy(v_backing, vocals_backing_path)
                        
                        with st.spinner("3/3 正在調用 DeReverb 模型進行 AI 極限乾濕分離..."):
                            v_dry, v_fx = run_uvr5_safely(v_lead_wet, UPLOAD_DIR, MODEL_DEREVERB, f"{uid}_step2")
                            shutil.copy(v_dry, vocals_dry_path)
                            shutil.copy(v_fx, vocals_effects_path)
                            
                        st.session_state.step2 = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ UVR5 管線執行失敗：{e}")
                        st.stop()

        if st.session_state.step2 and os.path.exists(vocals_dry_path):
            st.success("✅ UVR5 極限分離與環境特徵提取完成！")
            
            st.markdown("---")
            st.subheader("3. 智慧音色轉換與後期修復")
            
            # 定義所有可能的輸出路徑
            out_vocal_raw = os.path.join(UPLOAD_DIR, f"{safe_basename}_ai_lead_raw.wav")
            out_backing_raw = os.path.join(UPLOAD_DIR, f"{safe_basename}_ai_backing_raw.wav")
            out_vocal_final = os.path.join(UPLOAD_DIR, f"{safe_basename}_ai_lead_final.wav")
            out_backing_final = os.path.join(UPLOAD_DIR, f"{safe_basename}_ai_backing_final.wav")
            
            rvc_weights_dir = os.path.join(RVC_FULL_DIR, "assets", "weights")
            available_models = [f for f in os.listdir(rvc_weights_dir) if f.endswith(".pth")]
            
            if not available_models:
                st.warning(f"⚠️ 尚未偵測到模型！請確認檔案已放入: {rvc_weights_dir}")
            else:
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    selected_model = st.selectbox("選擇主唱音色模型", available_models)
                with col_m2:
                    pitch_shift = st.slider("整體人聲升降調 (Pitch Shift)", -12, 12, 0, 1)

                st.markdown("**進階處理選項：**")
                do_backing = st.checkbox("🎤 開啟『和音同步推理』 (將原曲和聲一併轉換為你的 AI 音色)", value=True)
                do_audiosr = st.checkbox("✨ 開啟『AudioSR 擴散修復』 (消除 HiFi-GAN 機械音與高頻偽影，極致擬真)", value=True)
                do_dsp = st.checkbox("🎛️ 開啟『Pedalboard 自動混音』 (加入錄音室級動態壓縮與自然空間混響)", value=True)
                
                # 🚀 【修改點 1】：解除 if not st.session_state.step3 的包裹限制，按鈕改為動態文字
                # 這樣參數配置區永遠不會消失，用戶換了模型或變調後，隨時可以再次點擊按鈕重新推理
                btn_text = "🚀 啟動完整轉換與修復管線" if not st.session_state.step3 else "🔄 更換音色或參數並重新推理"
                if st.button(btn_text, type="primary" if not st.session_state.step3 else "secondary"):
                    log_box = st.empty() 
                    rvc_python = os.path.join(RVC_FULL_DIR, "runtime", "python.exe")
                    rvc_script = os.path.join(RVC_FULL_DIR, "tools", "infer_cli.py")
                    
                    model_basename = os.path.splitext(selected_model)[0]
                    found_indexes = glob.glob(os.path.join(RVC_FULL_DIR, "logs", model_basename, "added_*.index"))
                    index_path = found_indexes[0] if found_indexes else ""

                    # 內部 RVC 執行閉包
                    def run_rvc(in_path, out_path, role_name):
                        st.info(f"正在執行 RVC 深度推理 ({role_name})...")
                        cmd = [
                            rvc_python, rvc_script,
                            "--f0up_key", str(pitch_shift),
                            "--input_path", os.path.abspath(in_path),
                            "--opt_path", os.path.abspath(out_path),
                            "--model_name", selected_model,
                            "--f0method", "rmvpe",
                            "--index_path", index_path,
                            "--device", "cuda:0",
                            "--index_rate", "0.75",       
                            "--protect", "0.45",          
                            "--resample_sr", "48000",     
                            "--filter_radius", "5"        
                        ]
                        if run_and_stream(cmd, RVC_FULL_DIR, log_box) != 0:
                            raise Exception(f"{role_name} 推理失敗")

                    try:
                        # 1. RVC 主唱推理
                        run_rvc(vocals_dry_path, out_vocal_raw, "主唱")
                        current_lead = out_vocal_raw
                        current_backing = vocals_backing_path 

                        # 2. RVC 和音推理
                        if do_backing and os.path.exists(vocals_backing_path):
                            run_rvc(vocals_backing_path, out_backing_raw, "和音")
                            current_backing = out_backing_raw

                        # 3. AudioSR 擴散修復
                        if do_audiosr:
                            st.info("正在執行 Pre-Diffusion 降噪與濾波 (抑制電音與底噪)...")
                            denoised_lead_path = os.path.join(UPLOAD_DIR, f"{safe_basename}_denoised_lead.wav")
                            apply_pre_diffusion_denoise(current_lead, denoised_lead_path)
                            current_lead = denoised_lead_path
                            
                            if do_backing and os.path.exists(current_backing):
                                denoised_backing_path = os.path.join(UPLOAD_DIR, f"{safe_basename}_denoised_backing.wav")
                                apply_pre_diffusion_denoise(current_backing, denoised_backing_path)
                                current_backing = denoised_backing_path

                            st.info("正在執行 AudioSR 擴散模型修復 (重建高保真細節)...")
                            sr_lead_path = os.path.join(UPLOAD_DIR, f"{safe_basename}_sr_lead.wav")
                            run_audiosr_restoration(current_lead, sr_lead_path)
                            current_lead = sr_lead_path
                            
                            if do_backing and os.path.exists(current_backing):
                                sr_backing_path = os.path.join(UPLOAD_DIR, f"{safe_basename}_sr_backing.wav")
                                run_audiosr_restoration(current_backing, sr_backing_path)
                                current_backing = sr_backing_path

                        # 4. Pedalboard 高階 DSP
                        if do_dsp:
                            st.info("正在應用 Pedalboard 錄音室級動態與混響處理...")
                            dsp_lead_path = out_vocal_final
                            apply_studio_dsp(current_lead, dsp_lead_path)
                            current_lead = dsp_lead_path
                            
                            if do_backing and os.path.exists(current_backing):
                                dsp_backing_path = out_backing_final
                                apply_studio_dsp(current_backing, dsp_backing_path)
                                current_backing = dsp_backing_path
                        else:
                            shutil.copy(current_lead, out_vocal_final)
                            if os.path.exists(current_backing):
                                shutil.copy(current_backing, out_backing_final)

                        # 記錄狀態
                        st.session_state.final_lead_path = out_vocal_final
                        st.session_state.final_backing_path = out_backing_final if do_backing else vocals_backing_path
                        st.session_state.used_dsp = do_dsp
                        st.session_state.step3 = True
                        
                        # 🚀 【修改點 2】：重新推理時，強制將第四步的「最終混音預覽」重置，防止舊的混音成果與新音色發生衝突
                        st.session_state.step4_preview = False
                        st.session_state.preview_path = ""
                        st.rerun()

                    except Exception as e:
                        st.error(f"系統執行失敗: {e}")

            # 🚀 【修改點 3】：將第四步的多軌混音控制台移出 else 區塊。
            # 當 step3 為 True 且輸出檔案存在時，混音面板會直接在下方展開，而不會擋住上方的模型切換區。
            if st.session_state.step3 and os.path.exists(st.session_state.get('final_lead_path', '')):
                st.markdown("---")
                st.subheader("4. 多軌特徵混音微調 (最終輸出)")
                
                col_v_vol, col_i_vol, col_b_vol, col_e_vol, col_i_pitch = st.columns(5)
                with col_v_vol: vocal_volume = st.slider("AI 主聲音量 (dB)", -15.0, 15.0, 2.0, 1.0, key=f"v_vol_{uid}")
                with col_i_vol: inst_volume = st.slider("純伴奏音量 (dB)", -15.0, 15.0, 0.0, 1.0, key=f"i_vol_{uid}")
                with col_b_vol: backing_volume = st.slider("和聲音量 (dB)", -15.0, 15.0, -2.0, 1.0, key=f"b_vol_{uid}")
                
                default_eff_vol = -20.0 if st.session_state.get('used_dsp', False) else -2.0
                with col_e_vol: effects_volume = st.slider("原曲殘響音量 (dB)", -30.0, 10.0, default_eff_vol, 1.0, key=f"e_vol_{uid}")
                with col_i_pitch: inst_pitch = st.slider("伴奏升降調", -12, 12, value=pitch_shift, step=1, key=f"i_pitch_{uid}")
                
                temp_preview_path = os.path.join(UPLOAD_DIR, f"{safe_basename}_preview.wav")
                
                if st.button("🎧 產生最終混音預覽", type="secondary"):
                    with st.spinner("正在處理音軌立體聲重構與多層特徵疊加..."):
                        current_inst_path = inst_path
                        current_backing_path = st.session_state.final_backing_path
                        
                        if inst_pitch != 0:
                            shifted_inst_path = os.path.join(UPLOAD_DIR, f"shifted_inst_{uid}.wav")
                            shift_script = os.path.join(TOOL_DIR, "shift_pitch.py")
                            if not os.path.exists(shift_script):
                                with open(shift_script, "w", encoding="utf-8") as f:
                                    f.write("""import sys, numpy as np, soundfile as sf, librosa\ninput_file, output_file, n_steps = sys.argv[1], sys.argv[2], float(sys.argv[3])\ny, sr = librosa.load(input_file, sr=None, mono=False)\nif y.ndim > 1:\n    y_shifted = np.array([librosa.effects.pitch_shift(y[i], sr=sr, n_steps=n_steps) for i in range(y.shape[0])]).T\nelse:\n    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps).reshape(-1, 1)\nsf.write(output_file, y_shifted, sr)""")
                            
                            rvc_python = os.path.join(RVC_FULL_DIR, "runtime", "python.exe")
                            
                            if not os.path.exists(shifted_inst_path):
                                subprocess.run([rvc_python, shift_script, inst_path, shifted_inst_path, str(inst_pitch)], check=True)
                            current_inst_path = shifted_inst_path
                            
                            if current_backing_path == vocals_backing_path:
                                shifted_backing_path = os.path.join(UPLOAD_DIR, f"shifted_backing_{uid}.wav")
                                if not os.path.exists(shifted_backing_path):
                                    subprocess.run([rvc_python, shift_script, vocals_backing_path, shifted_backing_path, str(inst_pitch)], check=True)
                                current_backing_path = shifted_backing_path
                        
                        inst_audio = AudioSegment.from_file(current_inst_path) + inst_volume
                        backing_audio = AudioSegment.from_file(current_backing_path) + backing_volume
                        vocal_audio = AudioSegment.from_file(st.session_state.final_lead_path) + vocal_volume
                        effects_audio = AudioSegment.from_file(vocals_effects_path) + effects_volume
                        
                        final_vocal = vocal_audio.overlay(effects_audio)
                        final_inst = inst_audio.overlay(backing_audio)
                        final_audio = final_inst.overlay(final_vocal)
                        
                        final_audio.export(temp_preview_path, format="wav")
                        
                        st.session_state.step4_preview = True
                        st.session_state.preview_path = temp_preview_path
                        st.rerun()

                if st.session_state.step4_preview and os.path.exists(st.session_state.preview_path):
                    st.success("✅ 預覽音訊已產生！")
                    st.audio(st.session_state.preview_path)
                    
                    if st.button("💾 合併並存入歷史紀錄", type="primary"):
                        import uuid
                        # 💡 每次儲存都產生一個獨立的「混音專屬 ID」，避免檔案與按鍵衝突
                        mix_id = uuid.uuid4().hex[:6]
                        final_mix_path = os.path.join(UPLOAD_DIR, f"{original_name}_AI_{uid}_{mix_id}.wav")
                        shutil.copy(st.session_state.preview_path, final_mix_path)
                        
                        history_item = {
                            "title": f"{original_name} (Vol: V{vocal_volume}/I{inst_volume})", 
                            "path": final_mix_path, 
                            "id": f"{uid}_{mix_id}" # 確保 ID 絕對唯一
                        }
                        st.session_state.history.append(history_item)
                        
                        st.success("🎉 已儲存新混音版本！")
                        st.balloons()

    if st.session_state.history:
        st.markdown("---")
        st.header("📚 歷史翻唱紀錄庫")
        for i, record in enumerate(reversed(st.session_state.history)):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**🎧 {record['title']}**")
                    st.audio(record['path'])
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    with open(record['path'], "rb") as f:
                        st.download_button("⬇️ 下載", f, f"{record['title']}.wav", "audio/wav", key=f"dl_{record['id']}", use_container_width=True)
            st.divider()

# ==========================================
# 分頁 2：建立新音色 (Local Training Pipeline)
# ==========================================
with tab2:
    st.title("⚙️ 本機 AI 聲音模型訓練")
    st.markdown("💡 **進階提示：** 若要**斷點續訓**，請輸入已存在的模型名稱，並「留空」音檔上傳區，系統將自動從上次進度繼續訓練。")
    
    if "is_training" not in st.session_state:
        st.session_state.is_training = False
        
    CURRENT_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # --- UI 參數設定區 ---
    new_model_name = st.text_input("1. 為新模型命名 (英文/數字) - 相同名稱將觸發續訓", disabled=st.session_state.is_training)
    dataset_files = st.file_uploader("2. 上傳乾淨人聲 (WAV) - 續訓可略過", accept_multiple_files=True, type=['wav'], disabled=st.session_state.is_training)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        total_epochs = st.slider("3. 訓練總輪數 (Total Epochs)", 10, 500, 100, 10, disabled=st.session_state.is_training)
    with col_t2:
        # 新增：定期存檔頻率滑桿
        save_freq = st.slider("4. 定期存檔頻率 (每 N 輪存檔一次)", 5, 100, 20, 5, disabled=st.session_state.is_training)
        st.caption("💡 存檔將保留為 `模型名_e輪數_s步數.pth`，可於 Tab 1 隨時載入試聽。")
    
    if st.session_state.is_training:
        st.button("🔥 正在全速深層學習中，請勿重新整理網頁...", type="secondary", disabled=True)
    else:
        start_train_btn = st.button("🚀 啟動全自動訓練 (或續訓)", type="primary")
        
    # --- 訓練邏輯驗證 ---
    if not st.session_state.is_training and start_train_btn:
        if not new_model_name:
            st.error("請填寫模型名稱！")
        else:
            # 判斷是「全新訓練」還是「斷點續訓」
            is_resuming = False
            rvc_logs_rel = f"logs/{new_model_name}"
            log_dir_path = os.path.join(RVC_FULL_DIR, rvc_logs_rel)
            
            if os.path.exists(log_dir_path) and not dataset_files:
                st.info(f"偵測到現有模型目錄且未上傳新資料，將啟動【斷點續訓】模式！")
                is_resuming = True
            elif not dataset_files:
                st.error("全新模型必須上傳訓練音檔！")
                st.stop()
                
            st.session_state.is_training = True
            st.session_state.is_resuming = is_resuming
            st.rerun()

    if st.session_state.is_training:
        status_text = st.empty()
        log_box = st.empty() 
        is_resuming = st.session_state.get("is_resuming", False)
        
        def find_rvc_script(target_names):
            for name in target_names:
                match = glob.glob(os.path.join(RVC_FULL_DIR, "**", name), recursive=True)
                if match: return match[0]
            return None

        script_preprocess = find_rvc_script(["preprocess.py", "trainset_preprocess_pipeline_print.py"])
        script_extract_f0 = find_rvc_script(["extract_f0_print.py", "extract_f0_rmvpe.py", "extract_f0.py"])
        script_extract_feat = find_rvc_script(["extract_feature_print.py", "extract_feature.py"])
        script_train = find_rvc_script(["train.py", "train_v2.py"])

        # 🚀 移除對 train_index.py 的依賴，只檢查這四個核心檔案
        if not all([script_preprocess, script_extract_f0, script_extract_feat, script_train]):
            st.session_state.is_training = False
            st.error("❌ RVC 核心腳本定位失敗！請檢查 RVC 資料夾是否完整。")
            st.button("重試並解除鎖定")
        else:
            rvc_dataset_rel = f"dataset/{new_model_name}"
            rvc_logs_rel = f"logs/{new_model_name}"
            rvc_python = os.path.join(RVC_FULL_DIR, "runtime", "python.exe")
            
            try:
                chosen_sample_rate = "40k" # 預設值
                possible_paths = [
                    os.path.join(RVC_FULL_DIR, "configs", "48k.json"),
                    os.path.join(RVC_FULL_DIR, "configs", "v2", "48k.json"),
                    os.path.join(RVC_FULL_DIR, "configs", "40k.json"),
                    os.path.join(RVC_FULL_DIR, "configs", "32k.json")
                ]
                config_template_path = next((p for p in possible_paths if os.path.exists(p)), None)
                if config_template_path:
                    if "48k" in config_template_path: chosen_sample_rate = "48k"
                    elif "32k" in config_template_path: chosen_sample_rate = "32k"
                sr_numeric_map = {"48k": "48000", "40k": "40000", "32k": "32000"}
                preprocess_sr = sr_numeric_map[chosen_sample_rate]

                # ==========================================
                # 前置處理 (全新訓練時才執行)
                # ==========================================
                if not is_resuming:
                    os.makedirs(os.path.join(RVC_FULL_DIR, rvc_dataset_rel), exist_ok=True)
                    os.makedirs(os.path.join(RVC_FULL_DIR, rvc_logs_rel), exist_ok=True)
                    
                    for f in dataset_files:
                        with open(os.path.join(RVC_FULL_DIR, rvc_dataset_rel, f.name), "wb") as out_f:
                            out_f.write(f.getbuffer())

                    status_text.markdown(f"### ⏳ 步驟 1/4：處理音頻切片 (目標採樣率: {preprocess_sr} Hz)...")
                    if run_and_stream([rvc_python, script_preprocess, rvc_dataset_rel, preprocess_sr, "8", rvc_logs_rel, "False", "3.0"], RVC_FULL_DIR, log_box) != 0: 
                        raise Exception("音頻切片失敗")
                    
                    status_text.markdown("### ⏳ 步驟 2/4：提取神經網路特徵 (Hubert & RMVPE)...")
                    if run_and_stream([rvc_python, script_extract_f0, rvc_logs_rel, "8", "rmvpe"], RVC_FULL_DIR, log_box) != 0: 
                        raise Exception("F0提取失敗")
                    if run_and_stream([rvc_python, script_extract_feat, "cuda:0", "1", "0", "0", rvc_logs_rel, "v2", "True"], RVC_FULL_DIR, log_box) != 0: 
                        raise Exception("特徵提取失敗")
                    
                    # 生成 filelist.txt
                    gt_wavs_dir = os.path.join(RVC_FULL_DIR, rvc_logs_rel, "0_gt_wavs")
                    feature_dir = os.path.join(RVC_FULL_DIR, rvc_logs_rel, "3_feature768")
                    f0_dir = os.path.join(RVC_FULL_DIR, rvc_logs_rel, "2a_f0")
                    f0nsf_dir = os.path.join(RVC_FULL_DIR, rvc_logs_rel, "2b-f0nsf")
                    
                    names = set(name.split(".")[0] for name in os.listdir(gt_wavs_dir)) & \
                            set(name.split(".")[0] for name in os.listdir(feature_dir)) & \
                            set(name.split(".")[0] for name in os.listdir(f0_dir)) & \
                            set(name.split(".")[0] for name in os.listdir(f0nsf_dir))
                    
                    opt_lines = [f"logs/{new_model_name}/0_gt_wavs/{name}.wav|logs/{new_model_name}/3_feature768/{name}.npy|logs/{new_model_name}/2a_f0/{name}.wav.npy|logs/{new_model_name}/2b-f0nsf/{name}.wav.npy|0" for name in sorted(list(names))]
                    with open(os.path.join(RVC_FULL_DIR, rvc_logs_rel, "filelist.txt"), "w", encoding="utf-8") as f:
                        f.write("\n".join(opt_lines))
                    
                    import json
                    with open(config_template_path, "r", encoding="utf-8") as f: 
                        config_data = json.load(f)
                        
                    config_data["train"]["batch_size"] = 16 # 稍微調降 batch size 確保 VRAM 安全
                    config_data["train"]["total_epoch"] = total_epochs
                    config_data["data"]["exp_dir"] = rvc_logs_rel
                    config_data["data"]["training_files"] = f"./logs/{new_model_name}/filelist.txt"
                    
                    with open(os.path.join(RVC_FULL_DIR, rvc_logs_rel, "config.json"), "w", encoding="utf-8") as f:
                        json.dump(config_data, f, indent=4, ensure_ascii=False)

                # ==========================================
                # 核心訓練 (全新與續訓共用)
                # ==========================================
                step_str = "3/4" if not is_resuming else "1/2"
                status_text.markdown(f"### 🔥 步驟 {step_str}：模型高強度訓練中 (目標 Epoch: {total_epochs})...")
                cmd_train = [
                    rvc_python, script_train, 
                    "-e", new_model_name, 
                    "-sr", chosen_sample_rate,
                    "-f0", "1", 
                    "-bs", "16", 
                    "-g", "0", 
                    "-te", str(total_epochs), 
                    "-se", str(save_freq), # 🚀 將存檔頻率綁定到這裡
                    "-v", "v2", 
                    "-l", "0", # 🚀 0 代表「保留」所有中途存檔，不要只留最後一個
                    "-c", "0"
                ]
                if run_and_stream(cmd_train, RVC_FULL_DIR, log_box) != 0: 
                    raise Exception("模型訓練本體失敗 (可能是 VRAM 爆滿或檔案遺失)")
                
                # ==========================================
                # 🚀 關鍵修復：動態生成獨立的 Index 編譯腳本 (絕對路徑修正)
                # ==========================================
                step_str_idx = "4/4" if not is_resuming else "2/2"
                status_text.markdown(f"### 🔍 步驟 {step_str_idx}：構建特徵檢索索引 (Index) 防止電音...")
                
                feat_dir = os.path.abspath(os.path.join(RVC_FULL_DIR, rvc_logs_rel, "3_feature768"))
                index_out = os.path.abspath(os.path.join(RVC_FULL_DIR, rvc_logs_rel, f"added_{new_model_name}_v2.index"))
                
                # 【修正】：將腳本直接寫入 RVC 的 logs 資料夾內，並強制獲取絕對路徑
                build_index_script = os.path.abspath(os.path.join(RVC_FULL_DIR, rvc_logs_rel, "build_index.py"))
                
                with open(build_index_script, "w", encoding="utf-8") as f:
                    f.write("""import sys, numpy as np, faiss
from pathlib import Path
feat_dir, out_index = sys.argv[1], sys.argv[2]
feats = [np.load(f) for f in Path(feat_dir).glob("*.npy")]
if not feats:
    print("No features found in 3_feature768!")
    sys.exit(1)
feats = np.concatenate(feats, axis=0)
# 直接使用 FlatL2 窮舉演算法，鎖死音準
index = faiss.IndexFlatL2(feats.shape[1])
index.add(feats)
faiss.write_index(index, out_index)
print(f"Index successfully saved to {out_index}")
""")

                # 調用剛寫入的腳本來產生 .index 檔
                cmd_index = [rvc_python, build_index_script, feat_dir, index_out]
                if run_and_stream(cmd_index, RVC_FULL_DIR, log_box) != 0:
                    st.warning("⚠️ 模型訓練成功，但 Index 索引建立失敗。這可能會導致推理時發生電音。")
                else:
                    # 💡 訓練成功後，順便把 Index 複製到推理區
                    index_infer_dst = os.path.join(CURRENT_PROJECT_DIR, "rvc_engine", "logs", new_model_name)
                    os.makedirs(index_infer_dst, exist_ok=True)
                    shutil.copy(index_out, os.path.join(index_infer_dst, os.path.basename(index_out)))
                
                # 訓練結束後的檔案複製與清理
                trained_model_src = os.path.join(RVC_FULL_DIR, "assets", "weights", f"{new_model_name}.pth")
                infer_model_dst = os.path.join(CURRENT_PROJECT_DIR, "rvc_engine", "assets", "weights", f"{new_model_name}.pth")
                
                st.session_state.is_training = False 
                st.session_state.is_resuming = False
                
                if os.path.exists(trained_model_src):
                    # 同步複製到 inference 資料夾 (如果有設的話)
                    os.makedirs(os.path.dirname(infer_model_dst), exist_ok=True)
                    shutil.copy(trained_model_src, infer_model_dst)
                    
                    st.success(f"🎉 專屬模型 `{new_model_name}` 訓練與 Index 構建已全數完成！")
                    st.info(f"💡 系統已依照你的設定，每 {save_freq} 輪保留了一份權重檔。你可以在 Tab 1 的下拉選單中找到如 `{new_model_name}_e{save_freq}_s...pth` 等檔案進行試聽。")
                    st.balloons()
                else:
                    st.error("訓練流程結束，但找不到最終的 `.pth` 權重檔，請檢查終端機紅字報錯。")
                    
            except Exception as e:
                st.session_state.is_training = False 
                st.session_state.is_resuming = False
                st.error(f"❌ 發生錯誤：{str(e)}")
                st.button("確認並解除鎖定")

# ==========================================
# 分頁 3：訓練素材極限提取 (UVR5 + VAD)
# ==========================================
with tab3:
    st.title("✂️ 訓練素材極限提取工具 (UVR5 + VAD 靜音提純)")
    st.markdown("使用最強 BS-Roformer 與 DeReverb 模型，外加 VAD 靜音裁切引擎，為 RVC 打造完美的 0 雜訊高密度資料集。")
    
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = []
    
    raw_files = st.file_uploader("上傳歌曲 (MP3/WAV/FLAC)", type=["mp3", "wav", "flac"], accept_multiple_files=True, key="toolkit_upload")
    
    apply_bve = st.checkbox("啟用『和聲剝離 (BVE)』(分離主唱與和音，避免 RVC 抓錯雙音高產生機械音)", value=True)
    apply_dereverb = st.checkbox("啟用『AI 極限乾聲提純』(自動調用 DeReverb 剝除空間音與殘響)", value=True)
    apply_vad = st.checkbox("啟用『自動靜音裁切 (VAD)』(自動刪除無人聲的空白底噪，並插入安全斷點)", value=True)
    
    if st.button("🚀 啟動 GPU 批次極限提取", type="primary"):
        # 1. 增加防呆：確認使用者有上傳檔案
        if not raw_files:
            st.warning("⚠️ 請先上傳至少一個音訊檔案！")
        else:
            st.session_state.processed_results = [] 
            with st.spinner("正在喚醒 5070 Ti 進行深度特徵分離與時序提純..."):
                for uploaded_file in raw_files:
                    safe_name = f"proc_{int(time.time())}_{uploaded_file.name.replace(' ', '_')}"
                    tool_file_path = os.path.join(TOOL_DIR, safe_name)
                    
                    with open(tool_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        # 1. 頂級人聲分離 (Roformer)
                        v_wet, _ = run_uvr5_safely(tool_file_path, TOOL_DIR, MODEL_ROFORMER, f"t3_{safe_name}")
                        final_vocal_path = v_wet

                        # 2. 和音消除 (BVE)
                        if apply_bve:
                            v_lead_wet, _ = run_uvr5_safely(final_vocal_path, TOOL_DIR, MODEL_BVE, f"t3_bve_{safe_name}")
                            final_vocal_path = v_lead_wet

                        # 3. 頂級乾濕分離 (DeReverb)
                        if apply_dereverb:
                            # 💡 已經將 v_wet 修正為 final_vocal_path，接續上一步的純主唱！
                            v_dry, _ = run_uvr5_safely(final_vocal_path, TOOL_DIR, MODEL_DEREVERB, f"t3_dr_{safe_name}")
                            final_vocal_path = v_dry
                            
                        # 4. VAD 靜音裁切
                        if apply_vad:
                            v_vad_path = os.path.join(TOOL_DIR, f"vad_{safe_name}.wav")
                            optimize_audio_for_rvc(final_vocal_path, v_vad_path)
                            final_vocal_path = v_vad_path

                        # 將最終結果重新命名
                        output_final = os.path.join(TOOL_DIR, f"{os.path.splitext(uploaded_file.name)[0]}_AI_Dataset.wav")
                        shutil.copy(final_vocal_path, output_final)
                        
                        st.session_state.processed_results.append({"name": uploaded_file.name, "path": output_final})
                    except Exception as e:
                        st.error(f"❌ 檔案 {uploaded_file.name} 處理失敗: {e}")
            
            # 💡 移除 st.rerun()！
            # Streamlit 在這裡會自然往下接續執行，如果有成功的結果，就會直接顯示 ZIP 下載與音訊預覽畫面。

    if st.session_state.processed_results:
        st.success(f"✅ 已處理完成 {len(st.session_state.processed_results)} 個檔案")
        
        zip_path = os.path.join(TOOL_DIR, "all_dataset_vocals.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for item in st.session_state.processed_results:
                orig_name_no_ext = os.path.splitext(item['name'])[0]
                arc_name = f"{orig_name_no_ext}_dataset.wav"
                zipf.write(item['path'], arcname=arc_name)

        with open(zip_path, "rb") as f:
            st.download_button("📦 批次下載純淨訓練集 (ZIP)", f, "RVC_Dataset_Vocals.zip", "application/zip", type="primary")

        for item in st.session_state.processed_results:
            st.markdown("---")
            st.write(f"**{item['name']}**")
            st.audio(item['path'])

# ==========================================
# 分頁 4：音域分析可視化
# ==========================================
with tab4:
    st.title("🎙️ 人聲音頻音域分析可視化工具 (高精確鋼琴版)")
    st.markdown("上傳純人聲音頻，或上傳原曲並啟用快速分離，系統將繪製高精度的音高分佈鋼琴卷軸。")
    
    uploaded_files_tab4 = st.file_uploader("請上傳音頻檔案 (WAV/MP3/FLAC)", type=['wav', 'mp3', 'flac', 'ogg'], accept_multiple_files=True, key="tab4_uploader")
    
    # 新增：Demucs 快速分離選項
    use_demucs = st.checkbox("⚡ 上傳的檔案包含背景音樂，請先使用 Demucs 進行快速人聲提取 (大幅節省效能)", value=False)
    
    if "recom_computed" not in st.session_state: st.session_state.recom_computed = False
    if "best_5_songs" not in st.session_state: st.session_state.best_5_songs = []
    if "best_5_counts" not in st.session_state: st.session_state.best_5_counts = None
    
    current_files_hash = ",".join(sorted([f.name for f in uploaded_files_tab4])) if uploaded_files_tab4 else ""
    if "files_hash" not in st.session_state or st.session_state.files_hash != current_files_hash:
        st.session_state.files_hash = current_files_hash
        st.session_state.recom_computed = False
        st.session_state.best_5_songs = []
        st.session_state.best_5_counts = None
    
    if uploaded_files_tab4:
        # 新增：觸發分析的按鈕，避免上傳後立刻卡住
        if st.button("📊 開始繪製音域分析圖", type="primary"):
            total_counts = np.zeros(len(MIDI_KEYS))
            images_dict = {}
            song_data = [] 
            
            st.markdown("### 🎵 單曲音域分析")
            progress_bar = st.progress(0)
            num_files = len(uploaded_files_tab4)
            cols_per_row = 1 if num_files <= 6 else 2 if num_files <= 12 else 3 if num_files <= 18 else 4
            
            for i, file in enumerate(uploaded_files_tab4):
                if i % cols_per_row == 0: cols = st.columns(cols_per_row)
                with cols[i % cols_per_row]:
                    # 狀態提示更新
                    status_text = f"正在分析 ({i+1}/{num_files}): {file.name}"
                    if use_demucs: status_text = f"正在分離並分析 ({i+1}/{num_files}): {file.name}"
                    
                    with st.spinner(status_text):
                        # 處理檔案儲存與 Demucs 分離邏輯
                        file.seek(0)
                        
                        if use_demucs:
                            # 如果需要分離，先將檔案存入暫存區
                            safe_name = f"t4_{int(time.time())}_{file.name.replace(' ', '_')}"
                            temp_input_path = os.path.join(UPLOAD_DIR, safe_name)
                            with open(temp_input_path, "wb") as f:
                                f.write(file.read())
                            
                            try:
                                # 調用輕量級 Demucs
                                vocal_path = run_demucs_fast(temp_input_path, UPLOAD_DIR, safe_name)
                                # 讀取分離後的乾聲給 librosa
                                with open(vocal_path, "rb") as f:
                                    file_bytes = f.read()
                            except Exception as e:
                                st.error(f"檔案 {file.name} 分離失敗: {e}")
                                continue # 跳過此檔案，繼續下一個
                        else:
                            # 不需要分離，直接讀取
                            file_bytes = file.read()
                        
                        # 核心音高分析 (不變)
                        single_counts = extract_pitch_distribution_fast(file_bytes, file.name)
                        total_counts += single_counts
                        song_data.append((file.name, single_counts))
                        
                        fig = plot_piano_roll_distribution(single_counts, f"Vocal Range: {file.name}")
                        st.pyplot(fig)
                        
                        img_buf = io.BytesIO()
                        fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=150)
                        images_dict[f"Vocal_Range_{file.name.rsplit('.', 1)[0]}.png"] = img_buf.getvalue()
                        plt.close(fig) 
                progress_bar.progress((i + 1) / num_files)
                
            # --- 以下累積分析與推薦邏輯保持不變 ---
            if len(song_data) > 1:
                st.markdown("---")
                st.markdown("### 📈 所有歌曲累積音域分析")
                fig_total = plot_piano_roll_distribution(total_counts, f"Cumulative Vocal Range ({len(song_data)} Tracks)")
                st.pyplot(fig_total)
                img_buf = io.BytesIO()
                fig_total.savefig(img_buf, format='png', bbox_inches='tight', dpi=150)
                images_dict["Cumulative_Vocal_Range.png"] = img_buf.getvalue()
                plt.close(fig_total)
                    
            if len(song_data) >= 5:
                st.markdown("---")
                st.markdown("### 🔮 智能模型數據優化")
                if not st.session_state.recom_computed:
                    import scipy.sparse as sp
                    song_data_sorted = sorted(song_data, key=lambda x: x[0])
                    names = [item[0] for item in song_data_sorted]
                    counts_matrix = sp.csr_matrix([item[1] for item in song_data_sorted])
                    
                    best_density, best_total_frames, best_indices, best_counts = -1, -1, None, None
                    for idxs in itertools.combinations(range(len(song_data_sorted)), 5):
                        comb_counts = np.asarray(counts_matrix[list(idxs)].sum(axis=0)).flatten()
                        density_score = np.count_nonzero(comb_counts >= OUTLIER_THRESHOLD)
                        total_frames = comb_counts.sum()
                        if (density_score > best_density) or (density_score == best_density and total_frames > best_total_frames):
                            best_density, best_total_frames, best_indices, best_counts = density_score, total_frames, idxs, comb_counts
                                
                    st.session_state.best_5_songs = [names[i] for i in best_indices]
                    st.session_state.best_5_counts = best_counts  
                    st.session_state.recom_computed = True
                            
                if st.session_state.recom_computed:
                    st.info("💡 以下 5 首為最高效的訓練集組合：")
                    for idx, song_name in enumerate(st.session_state.best_5_songs, 1):
                        st.markdown(f"**{idx}.** ` {song_name} `")
                    fig_recom = plot_piano_roll_distribution(st.session_state.best_5_counts, "Smart Recommended Dataset: Cumulative Vocal Range")
                    st.pyplot(fig_recom)
                    img_buf = io.BytesIO()
                    fig_recom.savefig(img_buf, format='png', bbox_inches='tight', dpi=150)
                    images_dict["Smart_Recommended_Top5.png"] = img_buf.getvalue()
                    plt.close(fig_recom)
                    
            if images_dict:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    # 加上 _ = 防止 writestr 回傳的 None 跑出來
                    for img_name, img_bytes in images_dict.items(): _ = zip_file.writestr(img_name, img_bytes)
                
                st.markdown("---")
                st.download_button(
                    label="📥 一鍵下載所有音域圖 (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="vocal_range_plots.zip",
                    mime="application/zip",
                    type="primary",
                    use_container_width=True
                )