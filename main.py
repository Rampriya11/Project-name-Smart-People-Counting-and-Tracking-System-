"""
Shared people detection and counting logic for the CLI and Streamlit interfaces.

Examples:
    python main.py --source 0
    python main.py --source videos\\people.mp4 --save
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


LOGGER = logging.getLogger(__name__)
TRACKER_CONFIG = Path(__file__).with_name("bytetrack_people.yaml")


class CrossingCounter:
    """Count tracked center points crossing a horizontal line with a deadband."""

    def __init__(self, max_missing_frames=90):
        self.max_missing_frames = max_missing_frames
        self._tracks = {}

    @staticmethod
    def classify(center_y, line_y, hysteresis):
        if center_y < line_y - hysteresis:
            return "above"
        if center_y > line_y + hysteresis:
            return "below"
        return "deadband"

    def update(self, track_id, center_y, line_y, hysteresis, frame_index):
        current_side = self.classify(center_y, line_y, hysteresis)
        track = self._tracks.get(track_id)
        previous_side = None if track is None else track["stable_side"]
        if track is not None and frame_index - track["last_frame"] > self.max_missing_frames:
            previous_side = None

        event = None
        if current_side != "deadband":
            if previous_side == "above" and current_side == "below":
                event = "downward"
            elif previous_side == "below" and current_side == "above":
                event = "upward"
            stable_side = current_side
        else:
            stable_side = previous_side

        self._tracks[track_id] = {
            "stable_side": stable_side,
            "last_frame": frame_index,
        }
        return {
            "track_id": track_id,
            "previous_side": previous_side or "unknown",
            "current_side": current_side,
            "stable_side": stable_side or "unknown",
            "event": event or "-",
        }


class PeopleCounter:
    """Per-session metrics and frame annotation backed by an Ultralytics model."""

    def __init__(self, model, conf=0.10, line_position=0.5, imgsz=960, debug=False):
        self.model = model
        self.conf = conf
        self.line_position = line_position
        self.imgsz = imgsz
        self.debug = debug
        self.reset()

    @staticmethod
    def crossing_hysteresis(frame_height, person_height):
        return max(
            3,
            int(frame_height * 0.005),
            int(person_height * 0.10),
        )

    def reset(self):
        self.seen_ids = set()
        self.crossing_counter = CrossingCounter()
        self.downward_count = 0
        self.upward_count = 0
        self.current_people = 0
        self.smoothed_fps = 0.0
        self._previous_time = None
        self.frame_index = 0
        self.debug_observations = []
        self.debug_events = []

        predictor = getattr(self.model, "predictor", None)
        if predictor is not None:
            for tracker in getattr(predictor, "trackers", []):
                tracker.reset()
            if hasattr(predictor, "vid_path"):
                predictor.vid_path = [None] * len(predictor.vid_path)

    def process_frame(self, frame):
        self.frame_index += 1
        height, width = frame.shape[:2]
        line_y = int(height * self.line_position)
        results = self.model.track(
            frame,
            persist=True,
            tracker=str(TRACKER_CONFIG),
            classes=[0],
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )
        result = results[0]
        annotated = frame.copy()
        self.current_people = 0
        self.debug_observations = []

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            ids = (
                result.boxes.id.int().cpu().tolist()
                if result.boxes.id is not None
                else [None] * len(boxes)
            )

            for box, track_id, class_id, confidence in zip(
                boxes, ids, classes, confidences
            ):
                if class_id != 0:
                    continue

                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                hysteresis = self.crossing_hysteresis(height, y2 - y1)
                self.current_people += 1
                if track_id is not None:
                    self.seen_ids.add(track_id)
                    observation = self.crossing_counter.update(
                        track_id, center_y, line_y, hysteresis, self.frame_index
                    )
                    if observation["event"] == "downward":
                        self.downward_count += 1
                    elif observation["event"] == "upward":
                        self.upward_count += 1
                else:
                    observation = {
                        "track_id": "missing",
                        "previous_side": "unknown",
                        "current_side": CrossingCounter.classify(
                            center_y, line_y, hysteresis
                        ),
                        "stable_side": "unknown",
                        "event": "untracked",
                    }

                observation["center_x"] = center_x
                observation["center_y"] = center_y
                observation["confidence"] = float(confidence)
                observation["bbox"] = (x1, y1, x2, y2)
                observation["hysteresis_px"] = hysteresis
                self.debug_observations.append(observation)
                if observation["event"] in ("downward", "upward"):
                    self.debug_events.append({
                        **observation,
                        "frame_index": self.frame_index,
                    })
                    self.debug_events = self.debug_events[-50:]
                if self.debug:
                    LOGGER.debug(
                        "Detection class=person id=%s confidence=%.3f "
                        "bbox=(%d,%d,%d,%d) center=(%d,%d) "
                        "hysteresis=%d previous_side=%s current_side=%s event=%s",
                        track_id,
                        confidence,
                        x1,
                        y1,
                        x2,
                        y2,
                        center_x,
                        center_y,
                        hysteresis,
                        observation["previous_side"],
                        observation["current_side"],
                        observation["event"],
                    )

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
                label = (
                    f"Person ID {track_id} | {confidence:.2f}"
                    if track_id is not None
                    else f"Person | {confidence:.2f}"
                )
                if self.debug:
                    label += (
                        f" c=({center_x},{center_y}) "
                        f"{observation['previous_side']}->{observation['current_side']}"
                    )
                    if observation["event"] != "-":
                        label += f" {observation['event']}"
                cv2.putText(
                    annotated, label, (max(0, x1), max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2,
                    cv2.LINE_AA,
                )
                cv2.circle(annotated, (center_x, center_y), 4, (0, 0, 255), -1)

        cv2.line(annotated, (0, line_y), (width, line_y), (255, 180, 0), 2)
        cv2.putText(
            annotated, "COUNTING LINE", (10, max(22, line_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2,
            cv2.LINE_AA,
        )

        now = time.perf_counter()
        instant_fps = (
            0.0 if self._previous_time is None
            else 1.0 / max(now - self._previous_time, 1e-6)
        )
        self._previous_time = now
        if instant_fps:
            self.smoothed_fps = instant_fps if self.smoothed_fps == 0 else (
                0.85 * self.smoothed_fps + 0.15 * instant_fps
            )

        overlay = annotated.copy()
        cv2.rectangle(overlay, (8, 8), (380, 188), (20, 20, 20), -1)
        dashboard = [
            f"Current people: {self.current_people}",
            f"Track IDs observed: {len(self.seen_ids)}",
            f"Downward crossings: {self.downward_count}",
            f"Upward crossings: {self.upward_count}",
            f"FPS: {self.smoothed_fps:.1f}",
            f"Line: y={line_y}px ({self.line_position:.0%})",
        ]
        annotated = cv2.addWeighted(overlay, 0.72, annotated, 0.28, 0)
        for index, text in enumerate(dashboard):
            cv2.putText(
                annotated, text, (18, 32 + index * 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1,
                cv2.LINE_AA,
            )

        return annotated

    def summary(self):
        return {
            "current_people": self.current_people,
            "unique_track_ids": len(self.seen_ids),
            "downward_crossings": self.downward_count,
            "upward_crossings": self.upward_count,
            "fps": self.smoothed_fps,
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Detect, track, and count people from a webcam or local video."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (for example 0) or local video file path.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Ultralytics model weights; pretrained yolo11n.pt is used by default.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.10,
        help="Detection confidence threshold (0 to 1); 0.10 preserves ByteTrack low-score matches.",
    )
    parser.add_argument(
        "--line-position", type=float, default=0.50,
        help="Horizontal counting-line position as a frame-height fraction (0 to 1).",
    )
    parser.add_argument("--save", action="store_true", help="Save annotated output video.")
    parser.add_argument(
        "--output", default="outputs/people_counting_output.mp4",
        help="Output video path when --save is enabled.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=960,
        help="YOLO inference image size; larger sizes help with distant people but run slower.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Show per-track center, line side, and crossing events.",
    )
    args = parser.parse_args(argv)

    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1.")
    if not 0.0 < args.line_position < 1.0:
        parser.error("--line-position must be between 0 and 1.")
    if args.imgsz <= 0:
        parser.error("--imgsz must be a positive integer.")
    return args


def parse_source(source):
    if source.isdecimal():
        return int(source)
    if source.lstrip("-").isdecimal():
        raise ValueError("Webcam index must be zero or greater.")
    video_path = Path(source)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    return str(video_path)


def create_output_writer(output_path, fps, frame_size):
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            frame_size,
        )
    except (OSError, cv2.error) as exc:
        raise RuntimeError(
            f"Could not create output video at {output_path}: {exc}"
        ) from exc

    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            f"Could not create output video at {output_path}. "
            "Check the destination folder and MP4 codec support."
        )
    return writer


def main(argv=None):
    args = parse_args(argv)
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    try:
        source = parse_source(args.source)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise SystemExit(
            f"Could not open source '{args.source}'. Check the webcam index, "
            "video file, camera permissions, and whether another app is using the camera."
        )

    output_writer = None
    output_path = Path(args.output)
    frame_count = 0
    try:
        try:
            model = YOLO(args.model)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load model '{args.model}'. Check the model path and, "
                "for an official model name, network access for Ultralytics' "
                f"automatic weight download. Details: {exc}"
            ) from exc

        counter = PeopleCounter(
            model=model,
            conf=args.conf,
            line_position=args.line_position,
            imgsz=args.imgsz,
            debug=args.debug,
        )
        print("Smart People Counting System started.")
        print("Press Q or Esc to quit; press S to save a screenshot.")
        print(f"Source: {args.source}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            annotated = counter.process_frame(frame)

            if args.save:
                if output_writer is None:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if not fps or fps <= 1 or fps > 120:
                        fps = 25.0
                    output_writer = create_output_writer(
                        output_path, fps, (frame.shape[1], frame.shape[0])
                    )
                output_writer.write(annotated)

            cv2.imshow("Smart People Counting System", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                screenshot_path = Path("screenshots") / (
                    f"detection_{int(time.time())}.jpg"
                )
                screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(screenshot_path), annotated):
                    raise RuntimeError(f"Could not save screenshot to {screenshot_path}.")
                print(f"Screenshot saved: {screenshot_path}")
            if key in (ord("q"), 27):
                break

        if frame_count == 0:
            raise RuntimeError(
                f"No frames could be read from '{args.source}'. "
                "The video may be empty, corrupt, or unsupported."
            )
    finally:
        cap.release()
        if output_writer is not None:
            output_writer.release()
        cv2.destroyAllWindows()

    summary = counter.summary()
    print("\nSession summary")
    print(f"Frames processed: {frame_count}")
    print(f"Track IDs observed: {summary['unique_track_ids']}")
    print(f"Downward crossings: {summary['downward_crossings']}")
    print(f"Upward crossings: {summary['upward_crossings']}")
    if args.save:
        print(f"Annotated video saved to: {output_path}")


if __name__ == "__main__":
    main()
