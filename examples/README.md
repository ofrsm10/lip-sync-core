# OpenAI Lip-Sync Examples

This directory contains example scripts demonstrating how to use the OpenAI lip-sync integration.

## Example: OpenAI TTS Integration

The `example_openai_usage.py` script shows how to:
1. Load a trained lip-sync model
2. Process a video to detect words from lip movements
3. Send detected words to OpenAI TTS API
4. Generate audio and text files
5. Play audio using hotkeys

### Usage

1. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **Update the script with your paths:**
   Edit `example_openai_usage.py` and set:
   - `VIDEO_PATH` - path to your video file
   - `MODEL_PATH` - path to your trained model (.pth file)

3. **Run the example:**
   ```bash
   python examples/example_openai_usage.py
   ```

4. **Use hotkeys to control audio:**
   - **SPACE** - Play the generated audio
   - **S** - Stop audio playback
   - **Q** - Quit the application

### Output Files

The script will create files in the `openai_outputs/` directory:
- `speech_YYYYMMDD_HHMMSS.mp3` - Generated audio file
- `speech_YYYYMMDD_HHMMSS.txt` - Text content of the speech

### Direct Usage (Alternative)

You can also use the command-line interface directly:

```bash
export OPENAI_API_KEY="your-api-key"
python run/openai_lipsync.py --video path/to/video.mp4 --model path/to/model.pth
```

## Requirements

Make sure you have installed all dependencies:
```bash
pip install -r requirements.txt
```

Required for OpenAI integration:
- openai
- keyboard
- pydub
