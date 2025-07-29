"""
ComfyUI Whisper 音频转录节点
支持单文件和批量音频转录功能
"""

# 导入节点类
from .whisper_node import (
    AudioToTextNode
)

NODE_CLASS_MAPPINGS={
    "Whisper音频转录":AudioToTextNode
}

NODE_DISPLAY_NAME_MAPPINGS={
    "Whisper音频转录":"Whisper音频转录"
}

# 导出节点映射
__all__ = [
    "AudioToTextNode"
]

