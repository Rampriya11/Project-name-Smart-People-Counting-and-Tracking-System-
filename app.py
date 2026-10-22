"""Streamlit dashboard for live and uploaded-video people counting."""

import hashlib
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import cv2
import imageio_ffmpeg
import streamlit as st
from ultralytics import YOLO

from main import CrossingCounter, LOGGER, PeopleCounter, create_output_writer


SUPPORTED_VIDEO_TYPES = ("mp4", "avi", "mov", "mkv")
VIDEO_STATE_KEYS = (
    "video_result_path",
    "video_result_metrics",
    "video_result_frames",
    "video_result_name",
    "video_result_upload_id",
    "video_result_orientation",
    "video_result_track_ids",
    "video_debug_observations",
    "video_debug_events",
)


def _upload_identity(uploaded_file):
    file_id = getattr(uploaded_file, "file_id", None)
    size = getattr(uploaded_file, "size", None)
    name = getattr(uploaded_file, "name", None) or getattr(uploaded_file, "filename", "")
    if file_id is None:
        try:
            payload = uploaded_file.getbuffer()
        except AttributeError:
            payload = uploaded_file.getvalue()
        file_id = hashlib.sha256(payload).hexdigest()
        size = len(payload)
    if size is None:
        size = 0
    identity = f"{file_id}:{name}:{size}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def clear_video_result(state, clear_upload_id=False):
    result_dir = state.get("video_result_dir")
    if result_dir:
        directory = Path(result_dir)
        if directory.is_dir():
            shutil.rmtree(directory)

    for key in VIDEO_STATE_KEYS:
        state[key] = None
    state["video_result_dir"] = None
    state["video_status"] = "idle"
    state["video_error"] = None
    if clear_upload_id:
        state["active_upload_id"] = None


def activate_upload(state, uploaded_file):
    upload_id = _upload_identity(uploaded_file)
    if upload_id != state.get("active_upload_id"):
        clear_video_result(state)
        state["active_upload_id"] = upload_id
        state["video_status"] = "ready"
    return upload_id


def _new_result_directory(session_id):
    return Path(tempfile.mkdtemp(prefix=f"smart_people_{session_id}_"))


def _inject_styles():
    st.markdown(
        """
        <style>
        .block-container {max-width: 1440px; padding-top: 1.7rem; padding-bottom: 3rem;}
        .spc-eyebrow {font-size: .78rem; font-weight: 700; letter-spacing: .12em;
          text-transform: uppercase; color: var(--text-color); opacity: .7; margin-bottom: .35rem;}
        .spc-subtitle {font-size: 1.05rem; opacity: .75; margin-top: -.5rem; margin-bottom: 1.4rem;}
        [data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22);
          border-radius: 14px; padding: 1rem 1.1rem;
          background: color-mix(in srgb, var(--secondary-background-color) 72%, transparent);}
        [data-testid="stMetricLabel"] {font-weight: 600;}
        [data-testid="stFileUploader"] {border-radius: 14px;}
        div.stButton > button, div.stDownloadButton > button {border-radius: 10px; min-height: 2.7rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _initialize_session_state():
    defaults = {
        "live_running": False,
        "live_capture": None,
        "live_counter": None,
        "session_id": uuid.uuid4().hex,
        "active_upload_id": None,
        "video_result_dir": None,
        "video_status": "idle",
        "video_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    for key in VIDEO_STATE_KEYS:
        if key not in st.session_state:
            st.session_state[key] = None


def _render_metrics(metrics, line_orientation="horizontal"):
    columns = st.columns(5)
    columns[0].metric("Current People", metrics["current_people"])
    columns[1].metric("Unique Track IDs", metrics["unique_track_ids"])
    negative_direction, positive_direction = (
        ("Leftward", "Rightward")
        if line_orientation == "vertical"
        else ("Upward", "Downward")
    )
    columns[2].metric(
        f"{positive_direction} Crossings", metrics["downward_crossings"]
    )
    columns[3].metric(
        f"{negative_direction} Crossings", metrics["upward_crossings"]
    )
    columns[4].metric("FPS", f"{metrics['fps']:.1f}")


def _render_video_result():
    result_path_value = st.session_state.video_result_path
    if not result_path_value:
        return

    result_path = Path(result_path_value)
    if not result_path.is_file():
        clear_video_result(st.session_state)
        st.error("The saved annotated video is no longer available. Process the upload again.")
        return

    st.divider()
    st.subheader("Processed result")
    st.caption(
        f"{st.session_state.video_result_name} · "
        f"{st.session_state.video_result_frames:,} frames processed"
    )
    try:
        _validate_video_preview_source(result_path)
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.warning("Saved annotated video is not previewable: %s", exc)
        st.error(f"The saved annotated video cannot be previewed: {exc}")
    else:
        st.video(str(result_path), format="video/mp4")
    result_metrics = dict(st.session_state.video_result_metrics)
    track_ids = st.session_state.video_result_track_ids
    if track_ids is not None:
        result_metrics["unique_track_ids"] = len(track_ids)
    _render_metrics(
        result_metrics,
        st.session_state.video_result_orientation or "horizontal",
    )
    st.caption(
        "Track IDs are tracker labels, not guaranteed distinct people. "
        "ID switches, occlusions, and lost tracks can increase this count."
    )
    try:
        with result_path.open("rb") as video_file:
            st.download_button(
                "Download annotated video",
                data=video_file,
                file_name=f"{Path(st.session_state.video_result_name).stem}_annotated.mp4",
                mime="video/mp4",
                key=f"download_video_{st.session_state.video_result_upload_id}",
                on_click="ignore",
                type="primary",
                use_container_width=True,
            )
    except OSError as exc:
        LOGGER.exception("Could not read saved video for download")
        st.error(f"Could not prepare the annotated video download: {exc}")

    if st.session_state.get("video_debug_observations"):
        with st.expander("Detection and crossing diagnostics"):
            st.dataframe(
                st.session_state.video_debug_observations,
                use_container_width=True,
                hide_index=True,
            )
            st.dataframe(
                st.session_state.video_debug_events,
                use_container_width=True,
                hide_index=True,
            )


@st.cache_resource(show_spinner="Loading pretrained YOLO11n weights...")
def load_model():
    try:
        return YOLO("yolo11n.pt")
    except Exception as exc:
        raise RuntimeError(
            "Could not load YOLO11n. Check that the official weights are available "
            "or that Ultralytics can download them. "
            f"Details: {exc}"
        ) from exc


def _temporary_path(suffix):
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    path = Path(handle.name)
    handle.close()
    return path


def _create_browser_video_writer(output_path, fps, frame_size):
    output_path = Path(output_path)
    intermediate_path = output_path.with_name(
        f"{output_path.stem}_intermediate.mp4"
    )
    try:
        writer = create_output_writer(intermediate_path, fps, frame_size)
    except (OSError, RuntimeError, cv2.error) as exc:
        intermediate_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not create an intermediate video for H.264 conversion. "
            f"Details: {exc}"
        ) from exc
    return writer, intermediate_path


def _ffmpeg_executable():
    return shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()


def _convert_video_to_h264(ffmpeg, input_path, output_path):
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        details = result.stderr.strip().splitlines()
        message = details[-1] if details else "FFmpeg returned an error."
        raise RuntimeError(f"Could not convert the annotated video to H.264: {message}")


def _validate_annotated_video(output_path, expected_frames):
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("The annotated output must use the .mp4 extension.")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("The annotated MP4 output is missing or empty.")
    with output_path.open("rb") as output_file:
        if b"ftyp" not in output_file.read(32):
            raise RuntimeError("The annotated output does not contain an MP4 container.")

    capture = cv2.VideoCapture(str(output_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not open the finalized annotated MP4.")

    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    decoded_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise RuntimeError("The annotated MP4 contains an unreadable frame.")
            decoded_frames += 1
    finally:
        capture.release()

    if codec.lower() not in ("h264", "avc1"):
        raise RuntimeError(
            f"The annotated MP4 uses unsupported codec '{codec or 'unknown'}'; "
            "a browser-compatible H.264 video is required."
        )
    if decoded_frames == 0:
        raise RuntimeError("OpenCV could not decode any frames from the annotated MP4.")
    if decoded_frames != expected_frames:
        raise RuntimeError(
            "The finalized annotated MP4 is incomplete: "
            f"expected {expected_frames} readable frames, got {decoded_frames}."
        )
    return codec, decoded_frames


def _validate_video_preview_source(video_path):
    video_path = Path(video_path)
    if video_path.suffix.lower() != ".mp4":
        raise ValueError("The saved preview must be an MP4 file.")
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError("The saved MP4 is missing or empty.")
    with video_path.open("rb") as video_file:
        if b"ftyp" not in video_file.read(32):
            raise RuntimeError("The saved preview does not contain an MP4 container.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not open the saved MP4.")
    try:
        fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(
            chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4)
        )
        ok, frame = capture.read()
    finally:
        capture.release()

    if codec.lower() not in ("h264", "avc1"):
        raise RuntimeError(
            f"The saved MP4 uses unsupported codec '{codec or 'unknown'}'. "
            "Process the upload again to create a browser-compatible H.264 preview."
        )
    if not ok or frame is None or frame.size == 0:
        raise RuntimeError("OpenCV could not decode a frame from the saved MP4.")


def process_uploaded_video_to_file(
    uploaded_file,
    counter,
    output_path,
    progress_callback=None,
):
    """Process an upload into an annotated file without modifying the upload."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix.lstrip(".") not in SUPPORTED_VIDEO_TYPES:
        raise ValueError("Upload an MP4, AVI, MOV, or MKV video.")

    input_path: Path | None = None
    output_path = Path(output_path)
    cap = None
    writer = None
    intermediate_path = None
    frame_count = 0

    try:
        input_path = _temporary_path(suffix)
        input_path.write_bytes(uploaded_file.getbuffer())
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(
                "OpenCV could not open this upload. The file may be corrupt or "
                "use a codec that is not supported on this system."
            )

        counter.reset()
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps <= 1 or source_fps > 120:
            source_fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated = counter.process_frame(frame)
            if writer is None:
                writer, intermediate_path = _create_browser_video_writer(
                    output_path, source_fps, (frame.shape[1], frame.shape[0])
                )
            writer.write(annotated)
            frame_count += 1
            progress_interval = max(1, total_frames // 100)
            if progress_callback is not None and (
                frame_count % progress_interval == 0 or frame_count == total_frames
            ):
                progress_callback(frame_count, total_frames)

        if frame_count == 0:
            raise ValueError("No video frames could be decoded from this upload.")

        cap.release()
        cap = None
        writer.release()
        writer = None
        if intermediate_path is not None:
            ffmpeg = _ffmpeg_executable()
            _convert_video_to_h264(ffmpeg, intermediate_path, output_path)
        codec, decoded_frames = _validate_annotated_video(output_path, frame_count)
        LOGGER.info(
            "Finalized annotated output %s: codec=%s, container=MP4, "
            "mime=video/mp4, frames=%d",
            output_path,
            codec,
            decoded_frames,
        )
        return frame_count, counter.summary()
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if intermediate_path is not None:
            intermediate_path.unlink(missing_ok=True)
        if input_path is not None:
            input_path.unlink(missing_ok=True)


def process_uploaded_video(uploaded_file, counter, progress_callback=None):
    """Compatibility helper returning annotated MP4 bytes for callers/tests."""
    output_path = _temporary_path(".mp4")
    try:
        frame_count, summary = process_uploaded_video_to_file(
            uploaded_file, counter, output_path, progress_callback
        )
        return output_path.read_bytes(), frame_count, summary
    finally:
        output_path.unlink(missing_ok=True)


def _release_camera():
    capture = st.session_state.get("live_capture")
    if capture is not None:
        capture.release()
    st.session_state.live_capture = None
    st.session_state.live_counter = None
    st.session_state.live_running = False


def _render_live_frame():
    if not st.session_state.get("live_running", False):
        return

    stop_column, status_column = st.columns([1, 4])
    if stop_column.button("Stop camera", type="secondary", key="stop_camera"):
        _release_camera()
        st.rerun()
    status_column.caption("Live detection running")

    capture = st.session_state.live_capture
    ok, frame = capture.read()
    if not ok:
        _release_camera()
        st.error(
            "The webcam stopped returning frames. Check camera permissions and "
            "whether another application is using it."
        )
        return

    try:
        annotated = st.session_state.live_counter.process_frame(frame)
    except Exception as exc:
        _release_camera()
        st.error(f"Live detection failed; the webcam has been released. Details: {exc}")
        return

    image_slot = st.empty()
    image_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
    metrics = st.session_state.live_counter.summary()
    columns = st.columns(5)
    columns[0].metric("Current people", metrics["current_people"])
    columns[1].metric("Unique Track IDs", metrics["unique_track_ids"])
    negative_direction, positive_direction = (
        ("Leftward", "Rightward")
        if st.session_state.live_counter.line_orientation == "vertical"
        else ("Upward", "Downward")
    )
    columns[2].metric(
        positive_direction, metrics["downward_crossings"]
    )
    columns[3].metric(
        negative_direction, metrics["upward_crossings"]
    )
    columns[4].metric("FPS", f"{metrics['fps']:.1f}")
    st.caption(
        "Track IDs are not guaranteed distinct people; ID switches, occlusions, "
        "and lost tracks can increase this count."
    )
    if st.session_state.live_counter.debug:
        with st.expander("Crossing debug details"):
            st.caption("Latest tracked centers and their stable-side transitions.")
            st.dataframe(
                st.session_state.live_counter.debug_observations,
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Recent counted crossings (up to the latest 50).")
            st.dataframe(
                st.session_state.live_counter.debug_events,
                use_container_width=True,
                hide_index=True,
            )


@st.fragment(run_every=0.05)
def _live_camera_fragment():
    _render_live_frame()


def main():
    st.set_page_config(page_title="Smart People Counting", page_icon="👥", layout="wide")
    _inject_styles()
    _initialize_session_state()

    st.markdown('<div class="spc-eyebrow">Computer vision · Real-time analytics</div>',
                unsafe_allow_html=True)
    st.title("Smart People Counting")
    st.markdown(
        '<div class="spc-subtitle">Detect, track, and measure pedestrian flow '
        "with YOLO11 and ByteTrack.</div>",
        unsafe_allow_html=True,
    )
    mode = st.radio(
        "Detection mode",
        ("Live Camera", "Upload Video"),
        horizontal=True,
        key="detection_mode",
    )
    st.sidebar.header("Detection settings")
    confidence = st.sidebar.slider(
        "Confidence threshold", 0.05, 0.95, 0.10, 0.05,
        help="A low threshold retains ByteTrack's low-confidence recovery detections.",
    )
    line_position = st.sidebar.slider(
        "Counting line position",
        0.10,
        0.90,
        0.62 if mode == "Upload Video" else 0.50,
        0.01,
        key=f"line_position_{mode.lower().replace(' ', '_')}",
    )
    line_orientation = st.sidebar.selectbox(
        "Counting line orientation",
        ("Horizontal", "Vertical"),
        key="line_orientation",
    ).lower()
    st.sidebar.caption(
        "Line position is measured from the top for horizontal lines "
        "and from the left for vertical lines."
    )
    image_size = st.sidebar.select_slider(
        "Inference resolution",
        options=(320, 480, 640, 800, 960, 1280),
        value=960,
        help="Larger sizes can detect distant people better, with lower FPS.",
    )
    debug = st.sidebar.checkbox("Show detection diagnostics")
    st.sidebar.caption("YOLO11n · ByteTrack · Person class only")

    if mode == "Live Camera":
        st.subheader("Live camera")
        st.caption("Analyze the camera connected to the computer running this app.")
        if st.session_state.live_running and st.session_state.live_counter is not None:
            if st.session_state.live_counter.line_orientation != line_orientation:
                st.session_state.live_counter.crossing_counter = CrossingCounter()
                st.session_state.live_counter.downward_count = 0
                st.session_state.live_counter.upward_count = 0
            st.session_state.live_counter.conf = confidence
            st.session_state.live_counter.line_position = line_position
            st.session_state.live_counter.imgsz = image_size
            st.session_state.live_counter.debug = debug
            st.session_state.live_counter.line_orientation = line_orientation

        if not st.session_state.live_running:
            if st.button("Start camera", type="primary", key="start_camera"):
                capture = cv2.VideoCapture(0)
                if not capture.isOpened():
                    capture.release()
                    st.error(
                        "Could not open webcam 0. Check camera permissions, connect "
                        "a camera, or close other applications using it."
                    )
                else:
                    try:
                        counter = PeopleCounter(
                            model=load_model(),
                            conf=confidence,
                            line_position=line_position,
                            imgsz=image_size,
                            debug=debug,
                            line_orientation=line_orientation,
                        )
                        st.session_state.live_capture = capture
                        st.session_state.live_counter = counter
                        st.session_state.live_running = True
                    except Exception as exc:
                        capture.release()
                        LOGGER.exception("Could not start camera inference")
                        st.error(f"Could not start live detection: {exc}")
                    if st.session_state.live_running:
                        st.rerun()
        else:
            _live_camera_fragment()
    else:
        st.subheader("Analyze a video")
        st.caption("Upload a video to create a persistent annotated preview and report.")
        if st.session_state.live_running:
            _release_camera()

        if st.button("Reset video results", key="reset_video_results"):
            clear_video_result(st.session_state, clear_upload_id=True)
            st.rerun()

        uploaded_file = st.file_uploader(
            "Select a video",
            type=SUPPORTED_VIDEO_TYPES,
            help="Supported formats: MP4, AVI, MOV, and MKV.",
            key="video_upload",
        )
        if uploaded_file is not None:
            upload_id = activate_upload(st.session_state, uploaded_file)

            st.info(
                f"**{uploaded_file.name}** · "
                f"{uploaded_file.size / (1024 * 1024):.1f} MB · "
                f"{Path(uploaded_file.name).suffix[1:].upper()} video"
            )
            process_label = (
                "Process video again"
                if st.session_state.video_result_upload_id == upload_id
                else "Process video"
            )
            if st.button(
                process_label,
                type="primary",
                key=f"process_video_{upload_id}",
            ):
                clear_video_result(st.session_state)
                result_dir = _new_result_directory(st.session_state.session_id)
                result_path = result_dir / "annotated.mp4"
                st.session_state.video_result_dir = str(result_dir)
                st.session_state.video_status = "processing"
                progress = st.progress(0, text="Preparing video...")
                try:
                    counter = PeopleCounter(
                        model=load_model(),
                        conf=confidence,
                        line_position=line_position,
                        imgsz=image_size,
                        debug=debug,
                        line_orientation=line_orientation,
                    )

                    def update_progress(done, total):
                        fraction = min(done / total, 1.0) if total > 0 else 0.0
                        progress.progress(
                            fraction,
                            text=f"Processing frame {done}"
                            + (f" of {total}" if total > 0 else ""),
                        )

                    frames, summary = process_uploaded_video_to_file(
                        uploaded_file, counter, result_path, update_progress
                    )
                    st.session_state.video_result_path = str(result_path)
                    st.session_state.video_result_metrics = summary
                    st.session_state.video_result_frames = frames
                    st.session_state.video_result_name = uploaded_file.name
                    st.session_state.video_result_upload_id = upload_id
                    st.session_state.video_result_orientation = line_orientation
                    st.session_state.video_result_track_ids = set(counter.seen_ids)
                    st.session_state.video_debug_observations = (
                        counter.debug_observations if debug else None
                    )
                    st.session_state.video_debug_events = (
                        counter.debug_events if debug else None
                    )
                    st.session_state.video_status = "complete"
                    st.session_state.video_error = None
                    progress.progress(1.0, text=f"Complete · {frames:,} frames")
                    st.success("Video processing complete. Your original upload is unchanged.")
                except Exception as exc:
                    progress.empty()
                    if result_dir.is_dir():
                        shutil.rmtree(result_dir)
                    st.session_state.video_result_dir = None
                    st.session_state.video_status = "error"
                    st.session_state.video_error = str(exc)
                    LOGGER.exception("Uploaded video processing failed")

        if st.session_state.video_status == "error" and st.session_state.video_error:
            st.error(st.session_state.video_error)
        elif st.session_state.video_status == "complete":
            st.success("Results are saved for this session. Downloading will not rerun inference.")
        elif st.session_state.video_status == "processing":
            st.info("Video processing is in progress.")

        _render_video_result()


if __name__ == "__main__":
    main()
