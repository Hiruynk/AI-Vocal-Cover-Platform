import streamlit as st
import os
import sys
import subprocess
import glob
import shutil
import time
import uuid

# app2.py 所需的套件
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import parselmouth
import zipfile
import itertools

# =========================================================
# 【系統層級設定與 FFmpeg Shared DLL 強行注入】
# =========================================================
st.set_page_config(page_title="本地 AI 翻唱工作站", layout="wide")

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

from pydub import AudioSegment
import torch

# =========================================================
# 【app2.py 全域圖表設定與常數】
# =========================================================
# 解決日文與中文顯示環境（防止檔名變方塊）
plt.rcParams['font.family'] = [
    'MS Gothic', 'Yu Gothic', 'Hiragino Sans', 
    'Noto Sans CJK JP', 'Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False 

# 設置 88 鍵鋼琴的 MIDI 範圍 (A0 到 C8)
MIN_MIDI = 21
MAX_MIDI = 108
MIDI_KEYS = np.arange(MIN_MIDI, MAX_MIDI + 1)
OUTLIER_THRESHOLD = 25  # 特定次數門檻：小於 25 幀 (0.25秒) 視為雜訊/異常值排除

# =========================================================
# 【通用終端機輸出擷取函數】
# =========================================================
def run_and_stream(cmd, cwd, log_box):
    """執行指令並將輸出即時打到 Streamlit 的介面上，完美相容 Windows 亂碼"""
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, 
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1
    )
    
    log_text = ""
    for line in iter(process.stdout.readline, ''):
        log_text += line
        display_text = log_text[-2000:] if len(log_text) > 2000 else log_text
        log_box.code(display_text, language="text")
        
    process.stdout.close()
    return_code = process.wait()
    return return_code

# =========================================================
# 【核心防護罩：安全的 Demucs 執行函數】
# =========================================================
def run_demucs_safely(input_file_path, output_dir, model):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(current_dir, "venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        venv_python = sys.executable

    custom_env = os.environ.copy()
    custom_env.pop("PYTHONPATH", None) 
    custom_env.pop("PYTHONHOME", None)
    venv_bin_dir = os.path.dirname(venv_python)
    custom_env["PATH"] = f"{venv_bin_dir};{custom_env.get('PATH', '')}"
    
    cmd = [
        venv_python, "-m", "demucs", 
        "-n", model, 
        "--two-stems=vocals", 
        "-o", output_dir, 
        input_file_path
    ]
    result = subprocess.run(cmd, check=True, env=custom_env)
    return result

# =========================================================
# 【app2.py 核心分析函數】
# =========================================================
@st.cache_data(show_spinner=False)
def extract_pitch_distribution_fast(file_bytes, file_name):
    """
    使用 parselmouth 提取基頻（高精確度 10ms 採樣）
    """
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
    """
    繪製音域分佈圖與完美對齊的 88 鍵鋼琴鍵盤
    """
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
    
    rect_base = plt.Rectangle(
        (MIN_MIDI - 0.5, 0), MAX_MIDI - MIN_MIDI + 1, 1, 
        facecolor='white', edgecolor='black', linewidth=0.8, zorder=1
    )
    ax_piano.add_patch(rect_base)
    
    for midi in range(MIN_MIDI, MAX_MIDI):
        is_curr_black = (midi % 12) in [1, 3, 6, 8, 10]
        is_next_black = ((midi + 1) % 12) in [1, 3, 6, 8, 10]
        y_max = 1.0 if (not is_curr_black and not is_next_black) else 0.35
        ax_piano.plot([midi + 0.5, midi + 0.5], [0, y_max], color='black', linewidth=0.6, zorder=2)
            
    for midi in MIDI_KEYS:
        if midi % 12 in [1, 3, 6, 8, 10]:
            rect_black = plt.Rectangle(
                (midi - 0.4, 0.35), 0.8, 0.65, 
                facecolor='#1a1a1a', edgecolor='black', linewidth=0.5, zorder=10
            )
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
# 【初始化與介面配置】
# =========================================================
UPLOAD_DIR = "temp_audio"
TOOL_DIR = "temp_toolkit"

# 全局綁定 2024 新版 RVC 路徑
RVC_FULL_DIR = r"C:\Users\User\Desktop\Work\RVC20240604Nvidia50x0"

for d in [UPLOAD_DIR, TOOL_DIR]:
    os.makedirs(d, exist_ok=True)

# --- 狀態鎖與歷史紀錄初始化 ---
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
        st.caption("CUDA 加速已啟用 (支援 Blackwell 架構)")
    else:
        st.error("未偵測到 NVIDIA GPU，將使用 CPU 處理。")
    
    st.markdown("---")
    st.markdown("### 當前推論進度")
    st.checkbox("1. 檔案上傳", value=st.session_state.step1, disabled=True)
    st.checkbox("2. 人聲分離", value=st.session_state.step2, disabled=True)
    st.checkbox("3. 音色轉換", value=st.session_state.step3, disabled=True)

# 加入了 Tab 4 (音域分析可視化)
tab1, tab2, tab3, tab4 = st.tabs(["🎙️ 日常翻唱工作站 (推論)", "⚙️ 建立新音色 (模型訓練)", "✂️ 訓練素材提取 (工具箱)", "📊 音域分析可視化"])

# ==========================================
# 分頁 1：日常翻唱工作站 (Inference Pipeline)
# ==========================================
with tab1:
    st.title("🎙️ AI 歌聲轉換管線")
    model_choice = st.selectbox(
        "選擇分離模型",
        options=["htdemucs", "mdx_extra_q"],
        index=0,
        help="htdemucs 是新版 Transformer 模型，mdx_extra_q 則是經典模型，若覺得效果不佳可嘗試切換。"
    )
    uploaded_file = st.file_uploader("1. 上傳目標歌曲 (MP3/WAV)", type=["mp3", "wav"])
    
    if uploaded_file is not None:
        # 【核心防覆蓋機制】：偵測到新檔案，立即派發全新 UUID 並重置進度
        if uploaded_file.name != st.session_state.current_filename:
            st.session_state.current_filename = uploaded_file.name
            st.session_state.uid = uuid.uuid4().hex[:8] # 產生 8 碼隨機 ID
            st.session_state.step1 = True
            st.session_state.step2 = False
            st.session_state.step3 = False
            st.session_state.step4_preview = False
            st.session_state.preview_path = ""
            
        uid = st.session_state.uid
        file_ext = os.path.splitext(uploaded_file.name)[1]
        original_name = os.path.splitext(uploaded_file.name)[0]
        safe_basename = f"cover_{uid}" # 每一首歌的專屬資料夾名稱
        file_path = os.path.join(UPLOAD_DIR, f"{safe_basename}{file_ext}")
        
        # 寫入上傳的檔案
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
        st.markdown("---")
        st.subheader(f"2. 人聲分離 (Demucs) - 當前處理: `{original_name}`")
        
        vocals_path = os.path.join(UPLOAD_DIR, "htdemucs", safe_basename, "vocals.wav")
        inst_path = os.path.join(UPLOAD_DIR, "htdemucs", safe_basename, "no_vocals.wav")
        
        if not st.session_state.step2:
            if st.button("啟動 GPU 人聲分離", type="primary", key="infer_demucs_btn"):
                with st.spinner("正在調用 5070 Ti 剔除背景樂器，請看終端機進度條..."):
                    try:
                        run_demucs_safely(file_path, UPLOAD_DIR, "mdx_extra_q")
                        st.session_state.step2 = True
                        st.rerun()
                    except subprocess.CalledProcessError:
                        st.error("❌ Demucs 執行失敗！請檢查終端機報錯。")
                        st.stop()

        if st.session_state.step2 and os.path.exists(vocals_path):
            st.success("✅ 分離完成！")
            col_v, col_i = st.columns(2)
            with col_v:
                st.caption("純人聲 (待轉換)")
                st.audio(vocals_path)
            with col_i:
                st.caption("純伴奏 (待合成)")
                st.audio(inst_path)
            
            st.markdown("---")
            st.subheader("3. 音色轉換 (RVC Inference)")
            output_vocal_path = os.path.join(UPLOAD_DIR, f"{safe_basename}_ai_vocals.wav")
            
            rvc_weights_dir = os.path.join(RVC_FULL_DIR, "assets", "weights")
            available_models = [f for f in os.listdir(rvc_weights_dir) if f.endswith(".pth")]
            
            if not available_models:
                st.warning(f"⚠️ 尚未偵測到模型！請確認檔案已放入: {rvc_weights_dir}")
            else:
                selected_model = st.selectbox("選擇你的音色模型", available_models)
                pitch_shift = st.slider("人聲升降調 (Pitch Shift)", -12, 12, 0, 1, help="男轉女建議 +12。注意：如果這裡調了 +4，下方的伴奏也必須調 +4，否則會走音！")
                
                if not st.session_state.step3:
                    if st.button("🚀 開始執行轉換"):
                        st.info(f"正在喚醒 5070 Ti，載入模型：{selected_model} ...")
                        log_box = st.empty() 
                        
                        abs_vocals = os.path.abspath(vocals_path)
                        abs_output = os.path.abspath(output_vocal_path)
                        
                        rvc_python = os.path.join(RVC_FULL_DIR, "runtime", "python.exe")
                        rvc_script = os.path.join(RVC_FULL_DIR, "tools", "infer_cli.py")
                        
                        model_basename = os.path.splitext(selected_model)[0]
                        index_search_pattern = os.path.join(RVC_FULL_DIR, "logs", model_basename, "added_*.index")
                        found_indexes = glob.glob(index_search_pattern)
                        index_path = found_indexes[0] if found_indexes else ""

                        cmd = [
                            rvc_python, rvc_script,
                            "--f0up_key", str(pitch_shift),
                            "--input_path", abs_vocals,
                            "--opt_path", abs_output,
                            "--model_name", selected_model,
                            "--f0method", "rmvpe",
                            "--index_path", index_path,
                            "--device", "cuda:0"
                        ]
                        
                        try:
                            return_code = run_and_stream(cmd, RVC_FULL_DIR, log_box)
                            if return_code != 0:
                                st.error("❌ RVC 引擎推理失敗！請查看上方日誌。")
                            else:
                                st.session_state.step3 = True
                                st.rerun()
                        except Exception as e:
                            st.error(f"系統執行失敗: {e}")

            if st.session_state.step3 and os.path.exists(output_vocal_path):
                st.markdown("---")
                st.subheader("4. 伴奏同步與混音微調 (預覽模式)")
                st.info("💡 提示：伴奏的升降調已經自動對齊你剛才在步驟 3 設定的人聲數值，確保絕對不會走音！")
                
                col_v_vol, col_i_vol, col_i_pitch = st.columns(3)
                with col_v_vol: vocal_volume = st.slider("人聲音量 (dB)", -15.0, 15.0, 0.0, 1.0, key=f"v_vol_{uid}")
                with col_i_vol: inst_volume = st.slider("伴奏音量 (dB)", -15.0, 15.0, 0.0, 1.0, key=f"i_vol_{uid}")
                
                # 【優化】：讓伴奏升降調的預設值，直接等於你在步驟 3 設定的 pitch_shift！
                with col_i_pitch: inst_pitch = st.slider("伴奏升降調 (半音)", -12, 12, value=pitch_shift, step=1, key=f"i_pitch_{uid}")
                
                # 暫存的預覽檔案路徑
                temp_preview_path = os.path.join(UPLOAD_DIR, f"{safe_basename}_preview.wav")
                
                if st.button("🎧 產生預覽試聽", type="secondary"):
                    with st.spinner("正在處理音軌與混音（若有調整伴奏 Key，需要約 10-20 秒進行立體聲重構）..."):
                        current_inst_path = inst_path
                        
                        # 【核心修正】：強化的立體聲無損變調演算法
                        if inst_pitch != 0:
                            shifted_inst_path = os.path.join(UPLOAD_DIR, "htdemucs", safe_basename, f"no_vocals_shifted_{inst_pitch}.wav")
                            if not os.path.exists(shifted_inst_path):
                                shift_script = os.path.join(TOOL_DIR, "shift_pitch.py")
                                # 寫入支援雙聲道 (Stereo) 的進階 librosa 腳本
                                with open(shift_script, "w", encoding="utf-8") as f:
                                    f.write("""
import sys
import numpy as np
import soundfile as sf
import librosa

input_file = sys.argv[1]
output_file = sys.argv[2]
n_steps = float(sys.argv[3])

# mono=False 確保讀取雙聲道立體聲
y, sr = librosa.load(input_file, sr=None, mono=False)

if y.ndim > 1:
    # 針對左右聲道分別進行變調，維持立體聲環繞感
    shifted_channels = []
    for i in range(y.shape[0]):
        shifted_channels.append(librosa.effects.pitch_shift(y[i], sr=sr, n_steps=n_steps))
    y_shifted = np.array(shifted_channels).T
else:
    # 單聲道防呆
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    y_shifted = y_shifted.reshape(-1, 1)

# 寫入新檔案
sf.write(output_file, y_shifted, sr)
                                    """)
                                
                                # 調用 RVC 環境裡的 Python 執行變調
                                rvc_python = os.path.join(RVC_FULL_DIR, "runtime", "python.exe")
                                try:
                                    subprocess.run([rvc_python, shift_script, inst_path, shifted_inst_path, str(inst_pitch)], check=True)
                                except Exception as e:
                                    st.error(f"❌ 伴奏變調失敗，請確認 RVC 環境是否安裝完整。錯誤訊息：{e}")
                                    st.stop()
                            
                            current_inst_path = shifted_inst_path
                        
                        # 混音疊加
                        inst_audio = AudioSegment.from_file(current_inst_path) + inst_volume
                        vocal_audio = AudioSegment.from_file(output_vocal_path) + vocal_volume
                        final_audio = inst_audio.overlay(vocal_audio)
                        final_audio.export(temp_preview_path, format="wav")
                        
                        st.session_state.step4_preview = True
                        st.session_state.preview_path = temp_preview_path
                        st.rerun()

                if st.session_state.step4_preview and os.path.exists(st.session_state.preview_path):
                    st.success("✅ 預覽音訊已產生！如果還是有電流音或走音，可以調整上方滑桿重新預覽。")
                    st.audio(st.session_state.preview_path)
                    
                    if st.button("💾 確認滿意，合併並存入歷史紀錄", type="primary"):
                        # 將最終滿意的版本搬移到暫存資料夾
                        final_mix_path = os.path.join(UPLOAD_DIR, f"{original_name}_AI_{uid}.wav")
                        shutil.copy(st.session_state.preview_path, final_mix_path)
                        
                        # 寫入歷史紀錄陣列
                        history_item = {
                            "title": f"{original_name} (Vol: V{vocal_volume}/I{inst_volume}, Inst Pitch: {inst_pitch})",
                            "path": final_mix_path,
                            "id": uid
                        }
                        
                        # 防止重複寫入
                        if not any(item['id'] == uid for item in st.session_state.history):
                            st.session_state.history.append(history_item)
                        else:
                            for item in st.session_state.history:
                                if item['id'] == uid:
                                    item['title'] = history_item['title']
                                    item['path'] = history_item['path']
                                    
                        st.success("🎉 已儲存！請至下方歷史紀錄庫查看。現在可以直接上傳下一首歌了！")
                        st.balloons()

    # ==========================================
    # 【新增】：歷史翻唱紀錄庫 (顯示於 Tab 1 底部)
    # ==========================================
    if st.session_state.history:
        st.markdown("---")
        st.header("📚 歷史翻唱紀錄庫")
        st.caption("此階段完成的所有作品都會保存在這裡。你可以直接上傳新歌，這裡的紀錄不會消失。")
        
        # 使用 reversed 讓最新製作的歌排在最上面
        for i, record in enumerate(reversed(st.session_state.history)):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**🎧 {record['title']}**")
                    st.audio(record['path'])
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    with open(record['path'], "rb") as f:
                        st.download_button(
                            label="⬇️ 點此下載", 
                            data=f, 
                            file_name=f"{record['title']}.wav", 
                            mime="audio/wav", 
                            key=f"dl_history_{record['id']}",
                            use_container_width=True
                        )
            st.divider()

# ==========================================
# 分頁 2：建立新音色 (Local Training Pipeline)
# ==========================================
with tab2:
    st.title("⚙️ 本機 AI 聲音模型訓練 (RTX 50 系列優化版)")
    st.markdown("將音檔交給底層 RVC 引擎，系統將自動定位腳本並調用 5070 Ti 進行加速訓練。")
    
    # 確保訓練狀態鎖初始化
    if "is_training" not in st.session_state:
        st.session_state.is_training = False
        
    # 指向你的 2024 新版 Nvidia50x0 資料夾
    RVC_FULL_DIR = r"C:\Users\User\Desktop\Work\RVC20240604Nvidia50x0" 
    CURRENT_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 當正在訓練時，鎖定輸入介面
    new_model_name = st.text_input(
        "1. 為新模型命名 (英文/數字)", 
        disabled=st.session_state.is_training,
        placeholder="例如: Akane_v1_5070"
    )
    dataset_files = st.file_uploader(
        "2. 上傳乾淨人聲 (WAV)", 
        accept_multiple_files=True, 
        type=['wav'],
        disabled=st.session_state.is_training
    )
    
    total_epochs = st.slider(
        "3. 訓練輪數 (Total Epochs)", 
        min_value=10, 
        max_value=300, 
        value=100, 
        step=10,
        disabled=st.session_state.is_training
    )
    
    # 根據訓練狀態動態更換按鈕
    if st.session_state.is_training:
        st.button("🔥 5070 Ti 正在全速深層學習中，請勿重新整理網頁...", type="secondary", disabled=True)
    else:
        start_train_btn = st.button("🚀 啟動全自動訓練", type="primary")
        
    # 正式進入訓練觸發邏輯
    if not st.session_state.is_training and 'start_train_btn' in locals() and start_train_btn:
        if not new_model_name or not dataset_files:
            st.error("請填寫模型名稱並上傳音檔！")
        elif not os.path.exists(RVC_FULL_DIR):
            st.error(f"找不到 RVC 核心資料夾，請檢查路徑：\n`{RVC_FULL_DIR}`")
        else:
            st.session_state.is_training = True
            st.rerun()

    # 當狀態鎖被激活時，執行後端自動化管線
    if st.session_state.is_training:
        st.info("🔍 正在初始化全自動訓練管線...")
        
        status_text = st.empty()
        log_box = st.empty() 
        
        # ---------------------------------------------------------
        # 【全自動腳本定位導航系統】
        # ---------------------------------------------------------
        def find_rvc_script(target_names):
            """在 RVC 資料夾中遞迴搜尋符合名稱的腳本，相容新舊版本目錄架構"""
            for name in target_names:
                match = glob.glob(os.path.join(RVC_FULL_DIR, "**", name), recursive=True)
                if match:
                    return match[0]
            return None

        # 將 2024 版的新檔名加入優先搜尋前排
        script_preprocess = find_rvc_script(["preprocess.py", "trainset_preprocess_pipeline_print.py"])
        script_extract_f0 = find_rvc_script(["extract_f0_print.py", "extract_f0_rmvpe.py", "extract_f0.py"])
        script_extract_feat = find_rvc_script(["extract_feature_print.py", "extract_feature.py"])
        script_train = find_rvc_script(["train.py", "train_v2.py"])

        if not all([script_preprocess, script_extract_f0, script_extract_feat, script_train]):
            st.session_state.is_training = False
            st.error("❌ 核心腳本定位失敗！某些組件未在 RVC 目錄中找到。")
            st.button("重試並解除鎖定")
        else:
            rvc_dataset_rel = f"dataset/{new_model_name}"
            rvc_logs_rel = f"logs/{new_model_name}"
            
            # 建立工作資料夾
            os.makedirs(os.path.join(RVC_FULL_DIR, rvc_dataset_rel), exist_ok=True)
            os.makedirs(os.path.join(RVC_FULL_DIR, rvc_logs_rel), exist_ok=True)
            
            # 寫入 WAV 檔案
            for f in dataset_files:
                with open(os.path.join(RVC_FULL_DIR, rvc_dataset_rel, f.name), "wb") as out_f:
                    out_f.write(f.getbuffer())
            
            rvc_python = os.path.join(RVC_FULL_DIR, "runtime", "python.exe")
                    
            try:
                # --- 步驟 A：處理資料集 ---
                status_text.markdown("### ⏳ 步驟 1/3：處理音頻切片...")
                cmd_preprocess = [rvc_python, script_preprocess, rvc_dataset_rel, "40000", "8", rvc_logs_rel, "False", "3.0"]
                if run_and_stream(cmd_preprocess, RVC_FULL_DIR, log_box) != 0: raise Exception("音頻切片失敗")
                
                # --- 步驟 B：提取特徵 ---
                status_text.markdown("### ⏳ 步驟 2/3：提取神經網路特徵 (Hubert & RMVPE)...")
                cmd_extract_f0 = [rvc_python, script_extract_f0, rvc_logs_rel, "8", "rmvpe"]
                if run_and_stream(cmd_extract_f0, RVC_FULL_DIR, log_box) != 0: raise Exception("F0音高提取失敗")
                
                cmd_extract_feature = [rvc_python, script_extract_feat, "cuda:0", "1", "0", "0", rvc_logs_rel, "v2", "True"]
                if run_and_stream(cmd_extract_feature, RVC_FULL_DIR, log_box) != 0: raise Exception("特徵向量提取失敗")
                
                # --- 【核心必殺技：模擬原廠 WebUI 自動生成 filelist.txt】 ---
                status_text.markdown("### 📝 正在手動編譯訓練清單 (filelist.txt)...")
                
                gt_wavs_dir = os.path.join(RVC_FULL_DIR, rvc_logs_rel, "0_gt_wavs")
                feature_dir = os.path.join(RVC_FULL_DIR, rvc_logs_rel, "3_feature768")
                f0_dir = os.path.join(RVC_FULL_DIR, rvc_logs_rel, "2a_f0")
                f0nsf_dir = os.path.join(RVC_FULL_DIR, rvc_logs_rel, "2b-f0nsf")
                
                if os.path.exists(gt_wavs_dir) and os.path.exists(feature_dir) and os.path.exists(f0_dir) and os.path.exists(f0nsf_dir):
                    # 執行演算法交集比對，確保通過前兩步的所有切片都入列
                    names = (
                        set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
                        & set([name.split(".")[0] for name in os.listdir(feature_dir)])
                        & set([name.split(".")[0] for name in os.listdir(f0_dir)])
                        & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
                    )
                    
                    if not names:
                        raise Exception("特徵提取未成功，生成的特徵交集為空！請檢查前兩步日誌。")
                        
                    opt_lines = []
                    for name in sorted(list(names)):
                        # 標準 RVC 訓練清單格式對齊
                        line = f"logs/{new_model_name}/0_gt_wavs/{name}.wav|logs/{new_model_name}/3_feature768/{name}.npy|logs/{new_model_name}/2a_f0/{name}.wav.npy|logs/{new_model_name}/2b-f0nsf/{name}.wav.npy|0"
                        opt_lines.append(line)
                    
                    target_filelist_txt = os.path.join(RVC_FULL_DIR, rvc_logs_rel, "filelist.txt")
                    with open(target_filelist_txt, "w", encoding="utf-8") as f:
                        f.write("\n".join(opt_lines))
                else:
                    raise Exception("特徵提取目錄不完整，無法生成 filelist.txt！")
                
                # --- 【自動動態生成 config.json】 ---
                status_text.markdown("### ⚙️ 正在動態生成訓練設定檔 (config.json)...")
                import json
                
                config_template_path = os.path.join(RVC_FULL_DIR, "configs", "v2", "40k.json")
                if not os.path.exists(config_template_path):
                    searched_templates = glob.glob(os.path.join(RVC_FULL_DIR, "configs", "**", "40k.json"), recursive=True)
                    if searched_templates: config_template_path = searched_templates[0]
                
                if os.path.exists(config_template_path):
                    with open(config_template_path, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                    
                    if "train" in config_data:
                        config_data["train"]["batch_size"] = 8
                        config_data["train"]["total_epoch"] = total_epochs
                    if "data" in config_data:
                        config_data["data"]["exp_dir"] = rvc_logs_rel
                        config_data["data"]["training_files"] = f"./logs/{new_model_name}/filelist.txt"
                    
                    target_config_json = os.path.join(RVC_FULL_DIR, rvc_logs_rel, "config.json")
                    with open(target_config_json, "w", encoding="utf-8") as f:
                        json.dump(config_data, f, indent=4, ensure_ascii=False)
                else:
                    raise Exception("核心組件缺失：在 configs 目錄下找不到 40k.json 配置範本！")
                
                # --- 步驟 C：正式訓練模型 ---
                status_text.markdown(f"### 🔥 步驟 3/3：模型高強度訓練中 (目標 Epoch: {total_epochs})...")
                cmd_train = [
                    rvc_python, script_train,
                    "-e", new_model_name, 
                    "-sr", "40k", 
                    "-f0", "1", 
                    "-bs", "8", 
                    "-g", "0", 
                    "-te", str(total_epochs), 
                    "-se", "50", 
                    "-v", "v2", 
                    "-l", "0",  
                    "-c", "1"   
                ]
                if run_and_stream(cmd_train, RVC_FULL_DIR, log_box) != 0: raise Exception("模型訓練本體失敗")
                
                # --- 完成與部署 ---
                status_text.markdown("### ✅ 訓練大功告成！正在自動掛載至翻唱工作站...")
                
                trained_model_src = os.path.join(RVC_FULL_DIR, "assets", "weights", f"{new_model_name}.pth")
                infer_model_dst = os.path.join(CURRENT_PROJECT_DIR, "rvc_engine", "assets", "weights", f"{new_model_name}.pth")
                
                st.session_state.is_training = False 
                
                if os.path.exists(trained_model_src):
                    shutil.copy(trained_model_src, infer_model_dst)
                    st.success(f"🎉 專屬模型 `{new_model_name}.pth` 已煉製完成並自動掛載！請前往「分頁 1」開始翻唱。")
                    st.balloons()
                else:
                    st.warning("訓練已結束，但未在 assets/weights 偵測到生成的模型檔，請檢查底層日誌。")
                    st.button("點擊刷新介面")
                    
            except Exception as e:
                st.session_state.is_training = False 
                st.error(f"❌ 發生致命錯誤：{str(e)}")
                st.button("確認並解除鎖定")

# ==========================================
# 分頁 3：訓練素材提取 (Data Preprocessing - 支援多檔案 + Zip 下載)
# ==========================================
with tab3:
    st.title("✂️ 訓練素材提取工具 (多檔案版)")
    
    # 建立一個用來暫存處理結果的 State
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = []
    
    raw_files = st.file_uploader("上傳歌曲 (MP3/WAV/FLAC)", type=["mp3", "wav", "flac"], accept_multiple_files=True, key="toolkit_upload")
    model_choice = st.selectbox(
    "選擇分離模型", 
    ["htdemucs", "mdx_extra_q"], 
    help="若覺得人聲分離效果不自然，請切換至 mdx_extra_q。"
)
    
    if st.button("🚀 啟動 GPU 批次提取人聲", type="primary"):
        st.session_state.processed_results = [] # 清空舊資料
        with st.spinner("正在進行批次分離..."):
            for uploaded_file in raw_files:
                file_ext = os.path.splitext(uploaded_file.name)[1]
                safe_name = f"proc_{int(time.time())}_{uploaded_file.name.replace(' ', '_')}"
                tool_file_path = os.path.join(TOOL_DIR, safe_name)
                
                with open(tool_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    run_demucs_safely(tool_file_path, TOOL_DIR, model=model_choice)
                    vocals_path = os.path.join(TOOL_DIR, "htdemucs", os.path.splitext(safe_name)[0], "vocals.wav")
                    if os.path.exists(vocals_path):
                        st.session_state.processed_results.append({"name": uploaded_file.name, "path": vocals_path})
                except Exception as e:
                    st.error(f"檔案 {uploaded_file.name} 處理失敗")
        st.rerun()

    # 顯示並提供下載
    if st.session_state.processed_results:
        st.success(f"✅ 已處理完成 {len(st.session_state.processed_results)} 個檔案")
        
        # 1. 提供打包下載按鈕
        zip_path = os.path.join(TOOL_DIR, "all_vocals.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for item in st.session_state.processed_results:
                orig_name_no_ext = os.path.splitext(item['name'])[0]
                arc_name = f"{orig_name_no_ext}_vocals.wav"
                zipf.write(item['path'], arcname=arc_name)

        with open(zip_path, "rb") as f:
            st.download_button("📦 下載所有人聲 (ZIP)", f, "All_Vocals.zip", "application/zip", type="primary")

        # 2. 顯示個別下載與播放
        for item in st.session_state.processed_results:
            st.markdown("---")
            st.write(f"**{item['name']}**")
            st.audio(item['path'])
            with open(item['path'], "rb") as f:
                st.download_button(f"⬇️ 下載 {item['name']}", f, f"{item['name']}_vocal.wav", "audio/wav")

# ==========================================
# 分頁 4：音域分析可視化 (從 app2.py 完美移植)
# ==========================================
with tab4:
    # --- Streamlit 網頁介面設計 ---
    st.title("🎙️ 人聲音頻音域分析可視化工具 (高精確鋼琴版)")
    st.markdown("上傳純人聲音頻檔案，系統將透過數據科學方法提取基頻，並於下方實體 88 鍵鋼琴矩陣上呈現精確音域密度分佈。")
    
    # 增加獨立的 key 防止不同分頁 File Uploader 報錯衝突
    uploaded_files_tab4 = st.file_uploader("請上傳純人聲音頻 (支援 wav, mp3, flac 等多個檔案)", type=['wav', 'mp3', 'flac', 'ogg'], accept_multiple_files=True, key="tab4_uploader")
    
    # 初始化/維護智慧推薦的模型狀態，防止下載時因重新刷新而消失
    if "recom_computed" not in st.session_state:
        st.session_state.recom_computed = False
    if "best_5_songs" not in st.session_state:
        st.session_state.best_5_songs = []
    if "best_5_counts" not in st.session_state:
        st.session_state.best_5_counts = None
    
    # 當上傳新檔案清單改變時，自動重置智慧推薦狀態
    current_files_hash = ",".join(sorted([f.name for f in uploaded_files_tab4])) if uploaded_files_tab4 else ""
    if "files_hash" not in st.session_state or st.session_state.files_hash != current_files_hash:
        st.session_state.files_hash = current_files_hash
        st.session_state.recom_computed = False
        st.session_state.best_5_songs = []
        st.session_state.best_5_counts = None
    
    if uploaded_files_tab4:
        total_counts = np.zeros(len(MIDI_KEYS))
        images_dict = {}
        song_data = [] # 用於存儲每首歌的名稱與頻次數據，供後續組合優化使用
        
        st.markdown("### 🎵 單曲音域分析")
        progress_bar = st.progress(0)
        
        # --- 動態網格計算 ---
        num_files = len(uploaded_files_tab4)
        if num_files <= 6:
            cols_per_row = 1
        elif num_files <= 12:
            cols_per_row = 2
        elif num_files <= 18:
            cols_per_row = 3
        else:
            cols_per_row = 4
        
        for i, file in enumerate(uploaded_files_tab4):
            # 每當到達新的一行，就建立新的 columns 列
            if i % cols_per_row == 0:
                cols = st.columns(cols_per_row)
                
            # 將內容分配到對應的欄位中
            with cols[i % cols_per_row]:
                with st.spinner(f"正在分析音頻 ({i+1}/{num_files}): {file.name} ..."):
                    file.seek(0)
                    file_bytes = file.read() 
                    
                    single_counts = extract_pitch_distribution_fast(file_bytes, file.name)
                    total_counts += single_counts
                    song_data.append((file.name, single_counts))
                    
                    # 繪製單曲圖表
                    fig = plot_piano_roll_distribution(single_counts, f"Vocal Range: {file.name}")
                    st.pyplot(fig)
                    
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=150)
                    img_buf.seek(0)
                    safe_name = file.name.rsplit('.', 1)[0]
                    images_dict[f"Vocal_Range_{safe_name}.png"] = img_buf.getvalue()
                    
                    plt.close(fig) 
                    
            # 更新進度條
            progress_bar.progress((i + 1) / num_files)
            
        if len(uploaded_files_tab4) > 1:
            st.markdown("---")
            st.markdown("### 📈 所有歌曲累積音域分析 (Aggregated Range)")
            with st.spinner("正在生成累積數據可視化..."):
                # 這裡沒有放在 columns 裡，所以依然會佔據一個大格
                fig_total = plot_piano_roll_distribution(total_counts, f"Cumulative Vocal Range ({len(uploaded_files_tab4)} Tracks)")
                st.pyplot(fig_total)
                
                img_buf = io.BytesIO()
                fig_total.savefig(img_buf, format='png', bbox_inches='tight', dpi=150)
                img_buf.seek(0)
                images_dict["Cumulative_Vocal_Range.png"] = img_buf.getvalue()
                
                plt.close(fig_total)
                
        st.success("✅ 音頻分析與數據可視化完成！")
        
        # --- ✨ 新增功能：智能音域模型推薦分析 (當歌曲 >= 5 首時開放) ✨ ---
        if len(uploaded_files_tab4) >= 5:
            st.markdown("---")
            st.markdown("### 🔮 智能模型數據優化")
    
            # 若尚未計算過，則顯示執行按鈕
            if not st.session_state.recom_computed:
                if st.button("🔮 執行智能音域模型推薦分析 (挑選最佳 5 首訓練集)", use_container_width=True, type="secondary"):
                    with st.spinner("正在利用數據科學演算法遍歷最優組合，排除極端異常值..."):
                        import scipy.sparse as sp
                        
                        # 1. 🛡️ 核心修復：對資料進行絕對排序，確保無論上傳順序如何，計算基準完全一致
                        song_data_sorted = sorted(song_data, key=lambda x: x[0])
                        names = [item[0] for item in song_data_sorted]
                        counts_matrix = sp.csr_matrix([item[1] for item in song_data_sorted])
                        
                        best_density = -1
                        best_total_frames = -1  # 用於平局時的第二決策指標
                        best_indices = None
                        best_counts = None      # 快取最優結果，避免迴圈外重複計算
                        
                        # 遍歷所有選 5 首的組合
                        for idxs in itertools.combinations(range(len(song_data_sorted)), 5):
                            # 透過 sparse matrix 疊加這 5 首的頻次
                            comb_counts_sparse = counts_matrix[list(idxs)].sum(axis=0)
                            # 轉回 1D NumPy array
                            comb_counts = np.asarray(comb_counts_sparse).flatten()
                            
                            # 計算「音域密度」(Pitch Density)
                            density_score = np.count_nonzero(comb_counts >= OUTLIER_THRESHOLD)
                            
                            # 計算該組合的總有效音訊幀數（作為第二指標）
                            total_frames = comb_counts.sum()
                            
                            # 2. ⚖️ 雙重機制判定：
                            # 條件 A: 發現了音域密度更高的組合
                            # 條件 B: 密度平局，但該組合包含的總幀數（訊號量）更多，代表資料更紮實、更具代表性
                            if (density_score > best_density) or (density_score == best_density and total_frames > best_total_frames):
                                best_density = density_score
                                best_total_frames = total_frames
                                best_indices = idxs
                                best_counts = comb_counts  # 直接接住數據
                                    
                        # 3. 💾 儲存最優結果至 Session State
                        st.session_state.best_5_songs = [names[i] for i in best_indices]
                        st.session_state.best_5_counts = best_counts  # 不再重複進行矩陣運算
                        st.session_state.recom_computed = True
                        st.rerun() # 強制刷新以渲染新生成的圖表
                        
            # 若已計算完成，直接渲染推薦結果與圖表
            if st.session_state.recom_computed:
                st.markdown("#### 🌟 最佳訓練集組合推薦結果")
                st.info("💡 演算法已為你精選出以下 5 首歌曲。這個組合在**排除低於閾值的瞬間雜訊**後，擁有最密集的有效音域分佈（Pitch Density），最能代表該人聲的完整潛力，強烈建議作為 AI 語音訓練的核心 Dataset：")
                
                for idx, song_name in enumerate(st.session_state.best_5_songs, 1):
                    st.markdown(f"**{idx}.** ` {song_name} `")
                    
                # 生成並顯示這 5 首的累積圖
                fig_recom = plot_piano_roll_distribution(st.session_state.best_5_counts, "Smart Recommended Dataset: Cumulative Vocal Range (Top 5)")
                st.pyplot(fig_recom)
                
                # 自動將這張精選圖轉成 bytes 塞進 images_dict 字典中
                img_buf = io.BytesIO()
                fig_recom.savefig(img_buf, format='png', bbox_inches='tight', dpi=150)
                img_buf.seek(0)
                images_dict["Smart_Recommended_Top5_Vocal_Range.png"] = img_buf.getvalue()
                plt.close(fig_recom)
                
        # --- 💾 唯一的統一匯出按鈕（維持一鍵下載原則） ---
        if images_dict:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for img_name, img_bytes in images_dict.items():
                    zip_file.writestr(img_name, img_bytes)
            
            zip_buffer.seek(0)
            
            st.markdown("---")
            st.markdown("### 💾 匯出所有分析圖表")
            
            # 根據是否執行了智能推薦，動態更改按鈕提示文字
            btn_label = "📥 一鍵下載所有音域圖 (含智能推薦圖) (ZIP 檔)" if st.session_state.recom_computed else "📥 一鍵下載所有音域圖 (ZIP 檔)"
            
            st.download_button(
                label=btn_label,
                data=zip_buffer.getvalue(),
                file_name="vocal_range_plots_package.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True
            )