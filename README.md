# ComfyUI Whisper 音频转录节点

这是一个用于ComfyUI的音频转录节点，使用OpenAI的Whisper模型将音频转换为文本。

## 安装步骤

### 1. 安装依赖
在ComfyUI的Python环境中安装必要的依赖：
需安装好ffmpeg

```bash
pip install torch numpy scipy openai-whisper
```

或者使用requirements.txt：
```bash
pip install -r requirements.txt
```
安装whisper
pip install git+https://github.com/openai/whisper.git
pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git


### 2. 放置节点文件
将整个`whisper`文件夹复制到ComfyUI的`custom_nodes`目录中：
```
ComfyUI/
└── custom_nodes/
    └── whisper/
        ├── __init__.py
        ├── whisper_node.py
        ├── requirements.txt
        └── README.md
```

### 3. 重启ComfyUI
重启ComfyUI后，节点应该会自动加载。

## 使用方法

1. 在ComfyUI中搜索"音频转文字"
2. 连接音频输入到节点的"audio"端口
3. 选择Whisper模型大小（tiny, base, small, medium, large）
4. 选择设备（cpu或cuda）
5. 可选：设置语言（默认"中文"）
6. 输出包含：
   - 转录的文本内容
   - .srt字幕文件路径
   - .vtt字幕文件路径

## 节点参数

- **audio_tensor**: 输入的音频张量
- **model**: Whisper模型大小选择
  - tiny: 最快，准确度较低
  - base: 平衡速度和准确度
  - small: 更好的准确度
  - medium: 高准确度
  - large: 最高准确度，但最慢
- **sample_rate**: 音频采样率（默认16000Hz）
- **language**: 语言代码（默认"zh"）

## 输出

- **text**: 转录的文本内容
- **srt_file**: 生成的.srt字幕文件路径
- **vtt_file**: 生成的.vtt字幕文件路径

## 故障排除

如果节点加载失败，请检查：

1. 所有依赖是否正确安装
2. 文件路径是否正确
3. Python版本兼容性（建议Python 3.8+）
4. ComfyUI版本兼容性

## 示例

查看`example`文件夹中的示例工作流和音频文件。 
