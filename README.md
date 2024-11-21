# Advanced-Detection-System-with-YOLO-model

An advanced real-time object detection application using YOLO model with a modern GUI interface on your PC. This application supports both webcam and video file input, offering comprehensive controls and real-time statistics for object detection.

## Features
- Dual input support:
  - Real-time webcam detection
  - Video file detection
- Modern dark-themed GUI interface
- Video playback controls:
  - Progress bar for video files
  - Time tracking
  - Video restart capability
- Adjustable detection parameters:
  - Multiple resolution options (224x224 to 1920x1080)
  - Confidence threshold slider (0.0 to 1.0)
- Real-time statistics:
  - FPS (Frames Per Second) counter
  - Object count
  - Confidence values for each detection
- Live video feed display
- Comprehensive playback controls

## Prerequisites
```python
# Required packages
opencv-python
ultralytics
customtkinter
pillow
numpy
tkinter
```

## Installation
1. Clone the repository:
```bash
git clone [repository-url]
cd 
```

2. Install required packages:
```bash
pip install opencv-python ultralytics customtkinter pillow numpy
```

3. Place your YOLOv8 model file in the appropriate directory:
```
best.pt
```
Note: Update the model path in the code according to your setup.

## Usage
1. Run the application:
```bash
python yolo_detection_app.py
```

2. Using the Interface:
   - Select input source (Camera or Video File)
   - If using video file:
     - Click "Select Video File" to choose your video
     - Use the progress bar to track video progress
     - Use the restart button to replay the video
   - Select desired resolution from the dropdown menu
   - Adjust confidence threshold using the slider
   - Click "Start Detection" to begin
   - Click "Stop Detection" to end the detection process

## Configuration
### Input Sources:
- Webcam (default)
- Video file (supports .mp4, .avi, .mov)

### Available Resolutions:
- 224x224
- 320x320
- 640x480 (default)
- 640x640
- 800x600
- 1280x720
- 1920x1080

### Confidence Threshold:
- Range: 0.0 to 1.0
- Default: 0.5
- Higher values result in more selective detection
- Lower values detect more objects but may include false positives

## Technical Details
### Components:
- **YOLO**: Object detection model
- **CustomTkinter**: Modern GUI framework
- **OpenCV**: Video capture and processing
- **Threading**: Separate thread for detection process
- **PIL**: Image processing for GUI display

### Performance Considerations:
- Higher resolutions may impact frame rate
- Video file size and format affect performance
- CPU vs GPU processing affects performance
- Adjust confidence threshold based on use case

## Troubleshooting
Common issues and solutions:

1. Camera not detected:
   - Ensure webcam is properly connected
   - Check if another application is using the camera
   - Verify camera permissions

2. Video file issues:
   - Ensure file format is supported
   - Check if file path contains special characters
   - Verify file isn't corrupted

3. Performance issues:
   - Lower the resolution
   - Use smaller video files
   - Check system resources
   - Consider GPU acceleration if available

4. Detection issues:
   - Adjust confidence threshold
   - Ensure proper lighting
   - Verify model path is correct

## Version
2.0.0

## Future Improvements (version 3.0.0)
- Multiple camera support
- Recording capability
- Custom model loading interface
- Detection history logging
- Export detection results
- Multiple detection class filtering
- Batch processing for multiple videos
- Advanced video controls (pause, frame-by-frame)
- Detection result export
