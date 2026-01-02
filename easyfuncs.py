import os
import subprocess
import shutil
import time
import torch
import gc
from mega import Mega
from datetime import datetime
import pandas as pd
import numpy as np
from pydub import AudioSegment
import gradio as gr
from typing import Optional, Tuple, Dict, List, Any
import requests
from pathlib import Path

# Try to import whisper speech with better error handling
try: 
    from whisperspeech.pipeline import Pipeline as TTS
    whisperspeak_on = True
except ImportError:
    whisperspeak_on = False
    TTS = None

class CachedModels:
    """Class to handle caching model URLs from a spreadsheet"""
    def __init__(self):
        self.csv_url = "https://huggingface.co/NeoPy/Ultimate-Models/resolve/main/spreadsheet.csv"
        self.cache_file = "spreadsheet.csv"
        self.models = {}
        self._load_or_download_data()
    
    def _load_or_download_data(self) -> None:
        """Load cached data or download from URL"""
        try:
            if Path(self.cache_file).exists():
                self.cached_data = pd.read_csv(self.cache_file)
            else:
                self.cached_data = pd.read_csv(self.csv_url)
                self.cached_data.to_csv(self.cache_file, index=False)
            
            # Cache model URLs
            for _, row in self.cached_data.iterrows():
                filename = row['Filename']
                url = None
                for value in row.values:
                    if isinstance(value, str) and "huggingface" in value:
                        url = value
                        break
                if url:
                    self.models[filename] = url
        except Exception as e:
            print(f"Error loading model data: {e}")
            self.models = {}
    
    def get_models(self) -> Dict[str, str]:
        """Get cached model URLs"""
        return self.models

def show(path: str, ext: str, on_error: Optional[List[str]] = None) -> List[str]:
    """List files with specific extension in directory"""
    try:
        return [f for f in os.listdir(path) if f.endswith(ext)]
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error listing files in {path}: {e}")
        return on_error or []

def run_subprocess(command: List[str], shell: bool = False) -> Tuple[bool, Optional[str]]:
    """Run subprocess command with error handling"""
    try:
        result = subprocess.run(
            command, 
            shell=shell, 
            check=True, 
            capture_output=True, 
            text=True
        )
        return True, None
    except subprocess.CalledProcessError as e:
        return False, f"Command failed with exit code {e.returncode}: {e.stderr}"
    except Exception as e:
        return False, str(e)

def download_file(url: str, dest_path: str) -> bool:
    """Download file from URL with progress tracking"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(dest_path, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_from_url(url: Optional[str] = None, model_name: Optional[str] = None) -> str:
    """Download model from URL or cached models"""
    models_cache = CachedModels()
    
    if not url and model_name:
        url = models_cache.models.get(model_name)
        if not url:
            gr.Warning(f"Model '{model_name}' not found in cache")
            return "Failed: Model not found"
    
    if not url:
        gr.Warning("No URL provided")
        return "Failed: No URL"
    
    # Clean model name
    if not model_name:
        model_name = Path(url).stem
    model_name = model_name.replace('.pth', '').replace('.index', '').replace('.zip', '')
    
    # Fix HuggingFace URL format
    url = url.replace('/blob/main/', '/resolve/main/').strip()
    
    # Create necessary directories
    for directory in ["downloads", "unzips", "assets/weights", "logs"]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    try:
        if url.endswith('.pth'):
            dest_path = f'assets/weights/{model_name}.pth'
            if download_file(url, dest_path):
                gr.Info(f"Downloaded {model_name}.pth to assets/weights/")
                
        elif url.endswith('.index'):
            model_log_dir = f'logs/{model_name}'
            Path(model_log_dir).mkdir(parents=True, exist_ok=True)
            dest_path = f'{model_log_dir}/added_{model_name}.index'
            if download_file(url, dest_path):
                gr.Info(f"Downloaded index to logs/{model_name}/")
                
        elif url.endswith('.zip'):
            dest_path = f'downloads/{model_name}.zip'
            if download_file(url, dest_path):
                # Extract zip file
                shutil.unpack_archive(dest_path, "unzips", 'zip')
                
                # Move files to appropriate locations
                for root, _, files in os.walk('unzips'):
                    for file in files:
                        file_path = Path(root) / file
                        if file.endswith(".index"):
                            model_log_dir = f'logs/{model_name}'
                            Path(model_log_dir).mkdir(parents=True, exist_ok=True)
                            shutil.copy2(file_path, f'{model_log_dir}/')
                        elif file.endswith(".pth") and "G_" not in file and "D_" not in file:
                            shutil.copy(file_path, f'assets/weights/{model_name}.pth')
        
        else:
            # Handle other URLs
            if "drive.google.com" in url:
                success, error = run_subprocess([
                    "gdown", url, "--fuzzy", "-O", f'downloads/{model_name}'
                ])
            elif "mega.nz" in url:
                mega = Mega()
                m = mega.login()
                m.download_url(url, 'downloads')
                success = True
            else:
                success, error = download_file(url, f'downloads/{model_name}')
                
            if not success:
                gr.Warning(f"Download failed: {error}")
                return "Failed"
                
        gr.Info("Download completed successfully")
        return "Done"
        
    except Exception as e:
        gr.Warning(f"Error during download: {str(e)}")
        return f"Failed: {str(e)}"
    finally:
        # Cleanup temporary directories
        for temp_dir in ["downloads", "unzips", "zip"]:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

def speak(audio: str, text: str) -> Optional[str]:
    """Generate speech from text using GPT-Sovits"""
    print(f"Processing audio: {audio}, text: {text}")
    current_dir = os.getcwd()
    
    try:
        # Ensure necessary directories exist
        Path("gpt_sovits_demo").mkdir(exist_ok=True)
        
        os.chdir('./gpt_sovits_demo')
        
        process = subprocess.Popen([
            sys.executable, "zero.py",
            "--input_file", audio,
            "--audio_lang", "English", 
            "--text", text,
            "--text_lang", "English"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        output_path = None
        for line in process.stdout:
            line = line.strip()
            if "All keys matched successfully" in line:
                continue
            if line.startswith("(") and line.endswith(")"):
                try:
                    path, finished = line[1:-1].split(", ")
                    if finished.lower() == "true":
                        output_path = path
                        break
                except ValueError:
                    continue
        
        process.wait()
        
        if output_path and Path(output_path).exists():
            return output_path
            
    except Exception as e:
        print(f"Error in speak function: {e}")
    finally:
        os.chdir(current_dir)
    
    return None

def whisperspeak(text: str, tts_lang: str = "en", cps: float = 10.5) -> Optional[str]:
    """Generate speech using WhisperSpeech"""
    if not whisperspeak_on:
        return None
    
    try:
        # Initialize TTS pipeline once
        if 'tts_pipe' not in globals():
            global tts_pipe
            tts_pipe = TTS(
                t2s_ref='whisperspeech/whisperspeech:t2s-v1.95-small-8lang.model',
                s2a_ref='whisperspeech/whisperspeech:s2a-v1.95-medium-7lang.model'
            )
        
        # Create output directory
        output_dir = "audios"
        Path(output_dir).mkdir(exist_ok=True)
        
        # Generate output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"{output_dir}/tts_audio_{timestamp}.wav"
        
        # Generate speech
        tts_pipe.generate_to_file(output_file, text, cps=cps, lang=tts_lang)
        
        return os.path.abspath(output_file)
        
    except Exception as e:
        print(f"Error in whisperspeak: {e}")
        return None

def stereo_process(audio1: Tuple[int, np.ndarray], 
                   audio2: Tuple[int, np.ndarray], 
                   choice: str) -> Tuple[int, np.ndarray]:
    """Process stereo audio with delay effect"""
    audio = audio1 if choice == "Input" else audio2
    
    try:
        sample_rate, audio_array = audio
        
        if len(audio_array.shape) == 1:  # Mono audio
            # Convert to stereo with delay effect
            delay_samples = int(sample_rate * (0.6 / 1000.0))
            
            # Ensure audio_array is int16
            if audio_array.dtype != np.int16:
                audio_array = audio_array.astype(np.int16)
            
            # Create stereo channels
            left_channel = np.zeros_like(audio_array)
            right_channel = audio_array
            
            # Apply delay to left channel
            if delay_samples < len(audio_array):
                left_channel[delay_samples:] = audio_array[:-delay_samples]
            
            # Combine channels
            stereo_array = np.column_stack((left_channel, right_channel))
            return (sample_rate, stereo_array)
        else:
            return audio
            
    except Exception as e:
        print(f"Error in stereo_process: {e}")
        return audio

def sr_process(audio1: Tuple[int, np.ndarray], 
               audio2: Tuple[int, np.ndarray], 
               choice: str) -> Optional[Tuple[int, np.ndarray]]:
    """Super-resolution audio processing"""
    # Clean up GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Clean up TTS pipe if exists
    if 'tts_pipe' in globals():
        del globals()['tts_pipe']
    
    # Select audio source
    audio = audio1 if choice == "Input" else audio2
    sample_rate, audio_array = audio
    
    try:
        # Create temporary directory
        temp_dir = Path("TEMP")
        temp_dir.mkdir(exist_ok=True)
        
        # Save audio to temporary file
        audio_segment = AudioSegment(
            audio_array.tobytes(),
            frame_rate=sample_rate,
            sample_width=audio_array.dtype.itemsize,
            channels=1 if len(audio_array.shape) == 1 else 2
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file = temp_dir / f'{choice}_{timestamp}.wav'
        audio_segment.export(str(temp_file), format="wav")
        
        # Prepare for super-resolution
        output_folder = Path("SR")
        if output_folder.exists():
            shutil.rmtree(output_folder)
        output_folder.mkdir(exist_ok=True)
        
        # Split audio into chunks
        chunk_duration = 5.12
        ffmpeg_command = [
            "ffmpeg", "-i", str(temp_file),
            "-f", "segment", "-segment_time", str(chunk_duration),
            "-c:a", "pcm_s16le",
            f"{output_folder}/out%03d.wav"
        ]
        
        success, error = run_subprocess(ffmpeg_command)
        if not success:
            gr.Warning(f"Failed to split audio: {error}")
            return None
        
        # Create file list for processing
        file_list = output_folder / "file_list.txt"
        with open(file_list, "w") as f:
            for file in sorted(output_folder.glob("out*.wav")):
                f.write(str(file) + "\n")
        
        # Run AudioSR if available
        venv_dir = Path("audiosr")
        if venv_dir.exists() and (venv_dir / "bin" / "python").exists():
            command = [
                str(venv_dir / "bin" / "python"),
                "-m", "audiosr",
                "--input_file_list", str(file_list),
                "--model_name", "speech",
                "--suffix", "_ldm",
                "--guidance_scale", "2.7",
                "--ddim_steps", "50",
                "--save_path", str(output_folder)
            ]
            
            success, error = run_subprocess(command)
            if not success:
                gr.Warning(f"AudioSR processing failed: {error}")
        
        # Wait for processing to complete
        time.sleep(2)
        
        # Find and merge processed chunks
        processed_chunks = []
        for file in sorted(output_folder.glob("*_ldm.wav")):
            chunk = AudioSegment.from_wav(str(file))
            processed_chunks.append(chunk)
        
        if processed_chunks:
            merged_audio = sum(processed_chunks)
            output_file = output_folder / f"{choice}_merged_ldm.wav"
            merged_audio.export(str(output_file), format="wav")
            
            # Load and return the processed audio
            display_audio = AudioSegment.from_file(str(output_file))
            sample_rate = display_audio.frame_rate
            audio_array = np.array(display_audio.get_array_of_samples())
            
            return (sample_rate, audio_array)
        else:
            gr.Warning("No processed audio chunks found")
            return None
            
    except Exception as e:
        gr.Warning(f"Error in super-resolution processing: {str(e)}")
        return None
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

# Initialize model cache globally
cached_models = CachedModels()
