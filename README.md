# Smart People Counting System

People detection and directional counting through a Streamlit web interface or the original OpenCV command-line interface. It uses Python, OpenCV, Ultralytics YOLO11, and ByteTrack.

The application uses the **official pretrained YOLO11n model** by default. It does not train a model and does not require a custom dataset. Ultralytics downloads the official weights on first use if they are not cached.

## Features

- **Live Camera:** Start and stop webcam inference from Streamlit. Frames show person boxes, tracking IDs, confidence, current people, observed track IDs, directional crossings, and FPS. Stopping releases the camera.
- **Upload Video:** Upload MP4, AVI, MOV, or MKV files, process them frame by frame with a progress indicator, preview the annotated result, and download it as MP4. The original upload is not modified.
- Both modes share the same YOLO/ByteTrack and hysteresis-based counting logic. Statistics and tracker state reset for each new run.
- Adjustable confidence, line height, and inference image size.
- Existing CLI workflow remains available.
- Processed previews, metrics, and download controls persist across Streamlit reruns; download does not rerun inference.

## Architecture

OpenCV reads video frames. The pretrained YOLO11 model detects only the COCO `person` class, then Ultralytics ByteTrack assigns IDs between frames using the project's `bytetrack_people.yaml` tracker settings. A per-track hysteresis state registers directional line crossings after the bounding-box center moves beyond a margin on the other side. The shared `PeopleCounter` draws the annotated frame and maintains per-run statistics. Streamlit uses the same processor for the camera and uploaded video modes.

## Installation (Windows / VS Code)

Open the project in VS Code and choose **Terminal → New Terminal**. In PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:

```bat
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Choose `.venv` with **Python: Select Interpreter** in VS Code. On first launch, allow internet access so Ultralytics can obtain official `yolo11n.pt` weights if needed.

## Launch the Streamlit interface

From the project directory with the virtual environment activated:

```powershell
streamlit run app.py
```

Streamlit opens the interface in a browser, usually at `http://localhost:8501`. The YOLO model is loaded once and cached for the Streamlit process.

### Live Camera

1. Select **Live Camera**.
2. Adjust detection confidence, counting-line height, and inference size in the sidebar if desired.
3. Click **Start camera**. The computer running Streamlit must have an accessible webcam.
4. Watch the annotated live frame and people/crossing/FPS metrics.
5. Click **Stop camera** to stop inference and release the device.

Detection defaults are confidence `0.10` and image size `960`. The low confidence cutoff is intentional: it allows ByteTrack's configured low-score matching stage (`track_low_thresh: 0.10`) to recover detections. Raising confidence above 0.10 filters those detections before ByteTrack sees them. Larger image sizes help with small/distant people but reduce CPU FPS; choose a lower size for speed.

The line is horizontal, drawn in blue, and labeled with its pixel row and height percentage. **Live Camera** starts at 50% frame height; **Upload Video** starts at 62%, which intersects the people path in the local `people.mp4` sample. Adjust **Counting line height** until the line passes through the path that people actually cross; a line above or below all tracked centers will correctly report zero crossings. Downward and upward describe image movement; whether those mean enter or exit depends on camera placement. The hysteresis band is adaptive: the maximum of 0.5% frame height, 10% of the tracked box height, and 3 pixels.

Enable **Show crossing debug details** to inspect each detected person's confidence, box, center, hysteresis width, previous stable side, current side (`above`, `deadband`, or `below`), and the last 50 counted crossing events. `--debug` in CLI mode logs the same detection/tracking details. Missing IDs are reported as untracked and are not assigned guessed crossings. A track keeps its side through gaps of up to 90 processed frames; after a longer interruption its next position initializes state without inventing a crossing.

The dashboard's **Track IDs seen** is the number of different tracker IDs observed during the session, not the number of distinct real people. It can increase when pedestrians enter the scene or when a tracker ID is fragmented. The configured ByteTrack buffer is 90 frames to tolerate brief missed detections; increasing it too far can associate stale tracks to other people in crowded scenes.

### Upload and process a video

1. Select **Upload Video**.
2. Choose an MP4, AVI, MOV, or MKV file.
3. Click **Process video** and wait for the frame progress indicator to complete.
4. Review the annotated preview and session statistics.
5. Click **Download annotated video** to save the processed MP4.

Processing uses a temporary input file that is deleted when processing completes. The annotated output is kept in a randomly named, per-session temporary directory so reruns and downloads can reuse it without storing the video bytes repeatedly in session state. The result directory is removed when a different upload replaces the current result or **Reset video results** is clicked. It does not write to or alter the selected original video.

After processing, the annotated preview, saved metrics, and download control remain visible when Streamlit reruns. The Download button uses Streamlit's `on_click="ignore"` behavior, so downloading does not trigger another script run or inference pass. Changing confidence/resolution controls also does not process the video; click **Process video again** to explicitly start another run.

## Command-line interface

Webcam:

```powershell
python main.py --source 0
```

Video file:

```powershell
python main.py --source "videos\people.mp4" --save
```

The CLI still supports `--model`, `--conf`, `--line-position`, `--imgsz`, `--save`, and `--output`. Defaults are `--conf 0.10` and `--imgsz 960`. Add `--debug` to log per-person confidence, ID, box, center, line side, and crossing event. Press **S** to save a screenshot, **Q** or **Esc** to quit.

## Testing

Run the unit tests and syntax checks:

```powershell
python -m unittest discover -v
python -m py_compile app.py main.py test_app.py test_main.py
```

The UI tests use Streamlit's `AppTest` to seed a processed session, trigger ordinary widget and download reruns, and verify that preview, metrics, output file, and download control remain. They also check that selecting a new upload removes the prior session artifact. This tests Streamlit rerun behavior, not a real browser download transfer.

Manual checks: launch with `streamlit run app.py`, test Start/Stop against a camera, upload a supported video, watch progress, and download/play the annotated MP4. For crossings, enable debug details and verify that a track's center begins on one side, enters the deadband, then reaches the opposite stable side; the relevant direction should increment once. The CLI can also be tested with `python main.py --source 0 --debug`.

On a 180-frame opening sample of the workspace's `people.mp4`, the updated settings (`conf=0.10`, `imgsz=960`, adaptive margin, 62% line) reported 22 current detections in the last frame, 54 track IDs observed over the sample, 2 downward and 2 upward crossings. A comparison over 90 frames averaged 6 people/frame with the old `conf=0.35`, `imgsz=640` settings versus about 18 people/frame with `conf=0.10`, `imgsz=960`; observed track IDs were 18 vs. 31 with the 90-frame tracker buffer. These counts are detections, not hand-labeled ground truth, and this is a short sample rather than a full-video accuracy evaluation.

## Troubleshooting

- **Webcam unavailable:** Check Windows camera privacy permissions, close applications already using the camera, and restart Streamlit. The camera must be connected to the machine running the Streamlit server.
- **Model weights cannot load:** Check that the virtual environment is active and internet access is available for the first YOLO11n download. The default weights are `yolo11n.pt`.
- **Upload rejected or no frames decoded:** Check the extension and that the file plays locally. OpenCV codec support varies by installation; try an MP4 encoded with H.264 or MPEG-4 if the source codec is unsupported.
- **Slow processing:** CPU speed, video resolution, and image size affect throughput. Select a smaller inference image size in the sidebar.
- **Upload limit:** Streamlit's configured maximum upload size may need increasing for large videos. Configure `server.maxUploadSize` in `.streamlit/config.toml` for your environment.
- **Browser can't connect:** Ensure Streamlit is still running and use the local URL it prints in the terminal.
- **`PeopleCounter.__init__()` rejects `debug`:** Stop and restart the Streamlit process using the project's selected `.venv` interpreter. The current `main.py` constructor accepts `debug`; a server started before the update may still have older Python code loaded. From the project terminal run `.\.venv\Scripts\python.exe -m streamlit run app.py`.

## Known limitations

- Unique track IDs seen is only an estimate of distinct people. Occlusion and tracker ID changes can affect it.
- Current people is the detections in the current frame, not an occupancy total.
- Crossing direction is computed from bounding-box centers and depends on camera view and line placement.
- Live camera capture is performed by the computer running the Streamlit server (local webcam use); remote users do not automatically provide a browser webcam stream.
- The app is intended for demonstration and evaluation, not safety-critical occupancy measurement.

## Resume-ready description

**Smart People Counting System | Python, Streamlit, OpenCV, YOLO11, ByteTrack**

- Built a Streamlit interface for real-time webcam and uploaded-video people detection using pretrained YOLO11 and ByteTrack.
- Implemented per-track directional line counting with hysteresis, annotated video preview/download, progress reporting, and live FPS/session metrics.
- Preserved a CLI workflow while sharing the same frame-processing and counting implementation across both interfaces.

The project integrates official pretrained model weights and does not train a custom YOLO model.
