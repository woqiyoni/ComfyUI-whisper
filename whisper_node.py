import torch
import numpy as np
import wave
import io
import scipy.signal
import whisper  # 使用OpenAI的whisper库进行语音识别
import os
import folder_paths
import uuid
import torchaudio
import datetime

class AudioToTextNode:
    # 定义语言映射为类变量
    LANGUAGE_MAP = {
        "中文": "zh",
        "英文": "en",
        "日语": "ja",
        "韩语": "ko",
        "法语": "fr",
        "德语": "de",
        "俄语": "ru",
    }
    
    @classmethod    
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",), 
                "model": (["base", "tiny", "small", "medium", "large"],),
                "device": (["cpu", "cuda"],),
            },
            "optional": {
                "language": (list(cls.LANGUAGE_MAP.keys()), {"default": "中文"}),  # 语音识别语言
                "save_to_output": ("BOOLEAN", {"default": True}),  # 是否保存到输出目录
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)  # 输出文本、srt文件路径、vtt文件路径
    RETURN_NAMES = ("text", "srt_file", "vtt_file",)    # 输出名称
    FUNCTION = "process_audio"
    CATEGORY = "audio"          # 节点分类

    def process_audio(self, audio, model, device="cpu", language="中文", save_to_output=True):
        audio_save_path = None
        try:
            # 验证音频数据
            if not audio or 'waveform' not in audio or 'sample_rate' not in audio:
                return ("错误：无效的音频数据", "", "")
            
            waveform = audio['waveform']
            sample_rate = audio['sample_rate']
            
            # 尝试获取文件名信息
            original_filename = None
            if 'filename' in audio:
                original_filename = audio['filename']
            elif 'name' in audio:
                original_filename = audio['name']
            elif 'path' in audio:
                original_filename = os.path.basename(audio['path'])
            
            # 如果获取到文件名，使用文件名；否则使用时间戳
            if original_filename and original_filename != "unknown":
                # 清理文件名（移除扩展名）
                base_filename = os.path.splitext(original_filename)[0]
                # 替换特殊字符
                safe_filename = "".join(c for c in base_filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
                if not safe_filename:
                    safe_filename = "audio"
            else:
                # 使用时间戳作为文件名
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_filename = f"whisper_{timestamp}"
            
            # 调试信息
            print(f"音频文件信息:")
            print(f"  原始文件名: {original_filename or '未获取到'}")
            print(f"  安全文件名: {safe_filename}")
            print(f"  音频数据键: {list(audio.keys())}")
            
            # 检查音频数据是否包含NaN或无穷大值
            if torch.isnan(waveform).any() or torch.isinf(waveform).any():
                return ("错误：音频数据包含NaN或无穷大值", "", "")
            
            # 确保音频数据在合理范围内
            if waveform.max() > 1.0 or waveform.min() < -1.0:
                waveform = torch.clamp(waveform, -1.0, 1.0)
            
            # 检查音频长度
            if waveform.numel() == 0:
                return ("错误：音频数据为空", "", "")
            
            # 将音频数据保存为临时文件
            temp_dir = folder_paths.get_temp_directory()
            os.makedirs(temp_dir, exist_ok=True)
            audio_save_path = os.path.join(temp_dir, f"{uuid.uuid1()}.wav")
            
            # 确保音频格式正确
            waveform_clean = waveform.squeeze(0)  # 移除多余的维度
            if waveform_clean.dim() == 1:
                waveform_clean = waveform_clean.unsqueeze(0)  # 确保是2D张量
            
            # 额外的音频预处理
            # 1. 确保数据类型正确
            if waveform_clean.dtype != torch.float32:
                waveform_clean = waveform_clean.float()
            
            # 2. 处理静音或过小的音频
            if torch.abs(waveform_clean).max() < 1e-6:
                return ("错误：音频信号过小或为静音", "", "")
            
            # 3. 标准化音频（可选）
            if waveform_clean.max() > 0:
                waveform_clean = waveform_clean / waveform_clean.abs().max()
            
            # 保存音频文件
            torchaudio.save(audio_save_path, waveform_clean, sample_rate)

            # 使用Whisper进行语音识别
            try:
                # 加载Whisper模型
                whisper_model = whisper.load_model(model, device=device)
                
                # 使用类变量进行语言映射
                whisper_language = self.LANGUAGE_MAP.get(language, "zh")
                
                # 检查音频文件是否存在且可读
                if not os.path.exists(audio_save_path):
                    return ("错误：临时音频文件创建失败", "", "")
                
                # 获取音频文件大小
                file_size = os.path.getsize(audio_save_path)
                if file_size == 0:
                    return ("错误：音频文件为空", "", "")
                
                # 使用Whisper进行转录，添加更多参数以提高稳定性
                result = whisper_model.transcribe(
                    audio_save_path, 
                    language=whisper_language,
                    fp16=False,  # 禁用半精度以避免数值问题
                    condition_on_previous_text=False,  # 不依赖之前的文本
                    temperature=0.0,  # 使用确定性解码
                    compression_ratio_threshold=2.4,  # 压缩比阈值
                    logprob_threshold=-1.0,  # 对数概率阈值
                    no_speech_threshold=0.6  # 无语音阈值
                )
                text = result["text"].strip()
                
                if not text:
                    text = "无法识别音频内容"
                
                # 生成字幕文件
                srt_content = self.generate_srt(result["segments"])
                vtt_content = self.generate_vtt(result["segments"])
                
                # 根据设置选择保存位置
                if save_to_output:
                    # 保存到输出目录
                    output_dir = folder_paths.get_output_directory()
                    srt_file_path = self.save_srt_file(srt_content, output_dir, safe_filename)
                    vtt_file_path = self.save_vtt_file(vtt_content, output_dir, safe_filename)
                else:
                    # 保存到临时目录
                    srt_file_path = self.save_srt_file(srt_content, temp_dir, safe_filename)
                    vtt_file_path = self.save_vtt_file(vtt_content, temp_dir, safe_filename)
                    
            except Exception as e:
                # 提供更详细的错误信息
                error_msg = str(e)
                if "nan" in error_msg.lower() or "inf" in error_msg.lower():
                    text = "语音识别失败：音频数据包含无效值（NaN或无穷大）"
                elif "cuda" in error_msg.lower():
                    text = "语音识别失败：GPU内存不足或CUDA错误"
                else:
                    text = f"语音识别失败: {error_msg}"

            return (text, srt_file_path, vtt_file_path)
            
        except Exception as e:
            return (f"处理音频时出错: {str(e)}", "", "")
        finally:
            # 清理临时文件
            if audio_save_path and os.path.exists(audio_save_path):
                try:
                    os.remove(audio_save_path)
                except Exception as e:
                    print(f"删除临时文件失败: {e}")
    
    def generate_srt(self, segments):
        """生成.srt格式的字幕内容"""
        srt_content = ""
        for i, segment in enumerate(segments, 1):
            start_time = self.format_time(segment["start"])
            end_time = self.format_time(segment["end"])
            text = segment["text"].strip()
            
            srt_content += f"{i}\n"
            srt_content += f"{start_time} --> {end_time}\n"
            srt_content += f"{text}\n\n"
        
        return srt_content
    
    def format_time(self, seconds):
        """将秒数转换为HH:MM:SS,mmm格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def save_srt_file(self, srt_content, save_dir, base_filename="whisper_subtitle"):
        """保存.srt文件并返回文件路径"""
        if not srt_content.strip():
            return ""
        
        srt_filename = f"{base_filename}.srt"
        srt_file_path = os.path.join(save_dir, srt_filename)
        
        # 如果文件已存在，添加数字后缀
        counter = 1
        original_path = srt_file_path
        while os.path.exists(srt_file_path):
            srt_filename = f"{base_filename}_{counter}.srt"
            srt_file_path = os.path.join(save_dir, srt_filename)
            counter += 1
        
        try:
            with open(srt_file_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            return srt_file_path
        except Exception as e:
            print(f"保存.srt文件失败: {e}")
            return ""
    
    def generate_vtt(self, segments):
        """生成.vtt格式的字幕内容"""
        vtt_content = "WEBVTT\n\n"  # VTT文件头
        
        for i, segment in enumerate(segments, 1):
            start_time = self.format_vtt_time(segment["start"])
            end_time = self.format_vtt_time(segment["end"])
            text = segment["text"].strip()
            
            vtt_content += f"{start_time} --> {end_time}\n"
            vtt_content += f"{text}\n\n"
        
        return vtt_content
    
    def format_vtt_time(self, seconds):
        """将秒数转换为HH:MM:SS.mmm格式（VTT格式）"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"
    
    def save_vtt_file(self, vtt_content, save_dir, base_filename="whisper_subtitle"):
        """保存.vtt文件并返回文件路径"""
        if not vtt_content.strip():
            return ""
        
        vtt_filename = f"{base_filename}.vtt"
        vtt_file_path = os.path.join(save_dir, vtt_filename)
        
        # 如果文件已存在，添加数字后缀
        counter = 1
        original_path = vtt_file_path
        while os.path.exists(vtt_file_path):
            vtt_filename = f"{base_filename}_{counter}.vtt"
            vtt_file_path = os.path.join(save_dir, vtt_filename)
            counter += 1
        
        try:
            with open(vtt_file_path, 'w', encoding='utf-8') as f:
                f.write(vtt_content)
            return vtt_file_path
        except Exception as e:
            print(f"保存.vtt文件失败: {e}")
            return ""

   

