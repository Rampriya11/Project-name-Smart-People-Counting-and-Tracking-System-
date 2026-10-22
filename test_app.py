import shutil
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from unittest.mock import Mock

import cv2
import numpy as np

from app import (
    VIDEO_STATE_KEYS,
    activate_upload,
    clear_video_result,
    process_uploaded_video,
    process_uploaded_video_to_file,
    _validate_annotated_video,
    _validate_video_preview_source,
)
from streamlit.testing.v1 import AppTest


class UploadedFile:
    def __init__(self, name, payload=b""):
        self.name = name
        self._payload = payload

    def getvalue(self):
        return self._payload

    def getbuffer(self):
        return memoryview(self._payload)

    @property
    def size(self):
        return len(self._payload)


class UploadLike:
    def __init__(self, name, payload=b""):
        self.name = name
        self._payload = payload

    def getvalue(self):
        return self._payload

    @property
    def size(self):
        return len(self._payload)


class FrameCounter:
    def reset(self):
        self.frames = 0

    def process_frame(self, frame):
        self.frames += 1
        return frame

    def summary(self):
        return {
            "current_people": 0,
            "unique_track_ids": 0,
            "downward_crossings": 0,
            "upward_crossings": 0,
            "fps": 0.0,
        }


class UploadValidationTests(unittest.TestCase):
    def test_upload_identity_works_without_getbuffer(self):
        upload = UploadLike("clip.mp4", b"video-data")
        self.assertTrue(activate_upload({"active_upload_id": None}, upload))

    def test_rejects_unsupported_extension_before_creating_files(self):
        with self.assertRaisesRegex(ValueError, "MP4, AVI, MOV, or MKV"):
            process_uploaded_video(UploadedFile("clip.wmv"), counter=None)

    def test_cleans_temporary_files_when_video_is_invalid(self):
        with TemporaryDirectory() as folder:
            temporary_paths = iter(
                (Path(folder) / "upload.mp4", Path(folder) / "annotated.mp4")
            )
            with patch("app._temporary_path", side_effect=lambda suffix: next(temporary_paths)):
                with self.assertRaisesRegex(ValueError, "could not open"):
                    process_uploaded_video(
                        UploadedFile("invalid.mp4", b"not a valid video"),
                        counter=None,
                    )

            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_processed_output_is_h264_mp4_with_readable_frames(self):
        with TemporaryDirectory() as folder:
            input_path = Path(folder) / "input.mp4"
            writer = cv2.VideoWriter(
                str(input_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                8.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            for frame_index in range(5):
                writer.write(np.full((48, 64, 3), frame_index * 20, dtype=np.uint8))
            writer.release()
            upload_payload = input_path.read_bytes()
            output_path = Path(folder) / "annotated.mp4"

            try:
                frames, summary = process_uploaded_video_to_file(
                    UploadedFile("source.mp4", upload_payload),
                    FrameCounter(),
                    output_path,
                )
            except RuntimeError as exc:
                if (
                    "OpenCV's H.264 encoder is unavailable" in str(exc)
                    and not shutil.which("ffmpeg")
                ):
                    self.skipTest("H.264 encoder and FFmpeg are unavailable.")
                raise

            self.assertEqual(frames, 5)
            self.assertEqual(summary["current_people"], 0)
            self.assertEqual(input_path.read_bytes(), upload_payload)
            codec, decoded_frames = _validate_annotated_video(output_path, frames)
            self.assertIn(codec.lower(), ("h264", "avc1"))
            self.assertEqual(decoded_frames, frames)
            self.assertGreater(output_path.stat().st_size, 0)

    def test_output_validation_rejects_empty_and_wrong_extension(self):
        with TemporaryDirectory() as folder:
            empty_mp4 = Path(folder) / "empty.mp4"
            empty_mp4.touch()
            with self.assertRaisesRegex(RuntimeError, "missing or empty"):
                _validate_annotated_video(empty_mp4, 1)

            wrong_extension = Path(folder) / "output.avi"
            wrong_extension.write_bytes(b"video")
            with self.assertRaisesRegex(ValueError, "\\.mp4"):
                _validate_annotated_video(wrong_extension, 1)

    def test_new_upload_clears_previous_result_artifact(self):
        with TemporaryDirectory() as folder:
            result_directory = Path(folder) / "session_result"
            result_directory.mkdir()
            result_file = result_directory / "annotated.mp4"
            result_file.write_bytes(b"old result")
            state = {
                "active_upload_id": "previous-upload",
                "video_result_dir": str(result_directory),
                "video_status": "complete",
                "video_error": None,
            }
            state.update({key: "previous" for key in VIDEO_STATE_KEYS})

            upload_id = activate_upload(
                state, UploadedFile("new_video.mp4", b"different content")
            )

            self.assertNotEqual(upload_id, "previous-upload")
            self.assertFalse(result_directory.exists())
            self.assertEqual(state["video_status"], "ready")
            self.assertEqual(state["active_upload_id"], upload_id)
            self.assertIsNone(state["video_result_dir"])
            self.assertTrue(all(state[key] is None for key in VIDEO_STATE_KEYS))

    def test_same_upload_does_not_clear_saved_result(self):
        with TemporaryDirectory() as folder:
            result_directory = Path(folder) / "session_result"
            result_directory.mkdir()
            result_file = result_directory / "annotated.mp4"
            result_file.write_bytes(b"keep result")
            upload = UploadedFile("same.mp4", b"same content")
            state = {
                "active_upload_id": None,
                "video_result_dir": str(result_directory),
                "video_status": "idle",
                "video_error": None,
            }
            state.update({key: None for key in VIDEO_STATE_KEYS})
            upload_id = activate_upload(state, upload)
            result_directory.mkdir(exist_ok=True)
            result_file.write_bytes(b"keep result")
            state["video_result_path"] = str(result_file)
            state["video_result_upload_id"] = upload_id
            state["video_status"] = "complete"

            self.assertEqual(activate_upload(state, upload), upload_id)
            self.assertTrue(result_file.is_file())
            self.assertEqual(state["video_status"], "complete")

    def test_clear_result_removes_only_its_session_artifact(self):
        with TemporaryDirectory() as folder:
            result_directory = Path(folder) / "session_result"
            other_session_directory = Path(folder) / "other_session"
            result_directory.mkdir()
            other_session_directory.mkdir()
            (result_directory / "annotated.mp4").write_bytes(b"result")
            other_file = other_session_directory / "annotated.mp4"
            other_file.write_bytes(b"other session")
            state = {
                "video_result_dir": str(result_directory),
                "video_status": "complete",
                "video_error": None,
            }
            state.update({key: None for key in VIDEO_STATE_KEYS})

            clear_video_result(state, clear_upload_id=True)

            self.assertFalse(result_directory.exists())
            self.assertTrue(other_file.is_file())
            self.assertIsNone(state["active_upload_id"])
            self.assertEqual(state["video_status"], "idle")

    def test_saved_result_and_metrics_survive_download_rerun(self):
        with TemporaryDirectory() as folder:
            result_directory = Path(folder) / f"session_{uuid.uuid4().hex}"
            result_directory.mkdir()
            video_path = result_directory / "annotated.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"avc1"),
                5.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()
            try:
                _validate_video_preview_source(video_path)
            except RuntimeError as exc:
                if "H.264" in str(exc) or "h264" in str(exc).lower():
                    self.skipTest("H.264 encoder is unavailable.")
                raise
            upload_id = "test-upload"
            app = AppTest.from_file(str(Path(__file__).with_name("app.py")))
            app.session_state["detection_mode"] = "Upload Video"
            app.session_state["session_id"] = uuid.uuid4().hex
            app.session_state["active_upload_id"] = upload_id
            app.session_state["video_result_dir"] = str(result_directory)
            app.session_state["video_result_path"] = str(video_path)
            app.session_state["video_result_metrics"] = {
                "current_people": 4,
                "unique_track_ids": 9,
                "downward_crossings": 2,
                "upward_crossings": 1,
                "fps": 12.5,
            }
            app.session_state["video_result_frames"] = 42
            app.session_state["video_result_name"] = "sample.mp4"
            app.session_state["video_result_upload_id"] = upload_id
            app.session_state["video_result_track_ids"] = set(range(9))
            app.session_state["video_status"] = "complete"
            app.session_state["video_error"] = None
            app.session_state["video_debug_observations"] = None
            app.session_state["video_debug_events"] = None

            with patch("app.load_model", Mock(side_effect=AssertionError(
                "Rendering a saved result must not rerun inference"
            ))):
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(
                    app.sidebar.selectbox[0].options, ["Horizontal", "Vertical"]
                )
                self.assertEqual(len(app.get("video")), 1)
                self.assertEqual(len(app.metric), 5)
                self.assertEqual([metric.value for metric in app.metric], [
                    "4", "9", "2", "1", "12.5",
                ])
                self.assertEqual(app.session_state["video_result_track_ids"], set(range(9)))
                self.assertEqual(len(app.get("download_button")), 1)

                app.sidebar.slider[0].set_value(0.20).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.get("video")), 1)
                self.assertEqual(len(app.metric), 5)
                self.assertEqual(app.session_state["video_result_track_ids"], set(range(9)))
                self.assertEqual(len(app.get("download_button")), 1)

                app.sidebar.selectbox[0].set_value("Vertical").run()
                self.assertFalse(app.exception)
                self.assertIn("Upward Crossings", [metric.label for metric in app.metric])
                self.assertIn("Downward Crossings", [metric.label for metric in app.metric])

                app.get("download_button")[0].click().run()

                self.assertFalse(app.exception)
                self.assertTrue(video_path.is_file())
                self.assertEqual(app.session_state["video_result_track_ids"], set(range(9)))
                self.assertEqual(len(app.get("video")), 1)
                self.assertEqual(len(app.metric), 5)
                self.assertEqual(len(app.get("download_button")), 1)
                self.assertEqual(app.session_state["video_result_frames"], 42)
                self.assertEqual(app.session_state["video_status"], "complete")


if __name__ == "__main__":
    unittest.main()
