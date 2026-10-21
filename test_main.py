import unittest
from pathlib import Path
import tempfile

import numpy as np

from main import (
    CrossingCounter,
    PeopleCounter,
    create_output_writer,
    parse_args,
    parse_source,
)


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def tolist(self):
        return self.value

    def int(self):
        return self


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[10, 20, 30, 60], [40, 20, 60, 60]])
        self.cls = FakeTensor([0, 2])
        self.conf = FakeTensor([0.83, 0.99])
        self.id = FakeTensor([17, 99])


class FakeResult:
    boxes = FakeBoxes()


class FakeModel:
    def __init__(self):
        self.track_options = None

    def track(self, frame, **kwargs):
        self.track_options = kwargs
        return [FakeResult()]


class CrossingCounterTests(unittest.TestCase):
    def test_counts_downward_crossing(self):
        counter = CrossingCounter()

        self.assertEqual(
            counter.update(7, 40, 50, 5, 1)["event"],
            "-",
        )
        crossing = counter.update(7, 56, 50, 5, 2)

        self.assertEqual(crossing["event"], "downward")
        self.assertEqual(crossing["previous_side"], "above")
        self.assertEqual(crossing["current_side"], "below")

    def test_counts_upward_crossing(self):
        counter = CrossingCounter()

        counter.update(3, 60, 50, 5, 1)
        crossing = counter.update(3, 40, 50, 5, 2)

        self.assertEqual(crossing["event"], "upward")
        self.assertEqual(crossing["previous_side"], "below")
        self.assertEqual(crossing["current_side"], "above")

    def test_counts_repeated_crossings_for_same_id(self):
        counter = CrossingCounter()

        counter.update(1, 40, 50, 5, 1)
        first = counter.update(1, 60, 50, 5, 2)
        second = counter.update(1, 40, 50, 5, 3)
        third = counter.update(1, 60, 50, 5, 4)

        self.assertEqual(
            [first["event"], second["event"], third["event"]],
            ["downward", "upward", "downward"],
        )

    def test_no_crossing_when_track_stays_on_same_side(self):
        counter = CrossingCounter()

        counter.update(5, 30, 50, 5, 1)
        result = counter.update(5, 35, 50, 5, 2)

        self.assertEqual(result["event"], "-")
        self.assertEqual(result["stable_side"], "above")

    def test_deadband_jitter_does_not_count_or_change_stable_side(self):
        counter = CrossingCounter()

        counter.update(5, 40, 50, 5, 1)
        for frame_index, center_y in enumerate((46, 50, 54, 48), start=2):
            result = counter.update(5, center_y, 50, 5, frame_index)
            self.assertEqual(result["current_side"], "deadband")
            self.assertEqual(result["stable_side"], "above")
            self.assertEqual(result["event"], "-")

        self.assertEqual(
            counter.update(5, 56, 50, 5, 6)["event"],
            "downward",
        )

    def test_short_tracking_interruption_preserves_crossing_state(self):
        counter = CrossingCounter(max_missing_frames=10)

        counter.update(9, 40, 50, 5, 1)
        crossing = counter.update(9, 60, 50, 5, 8)

        self.assertEqual(crossing["event"], "downward")

    def test_long_tracking_interruption_reinitializes_without_false_crossing(self):
        counter = CrossingCounter(max_missing_frames=10)

        counter.update(9, 40, 50, 5, 1)
        resumed = counter.update(9, 60, 50, 5, 20)

        self.assertEqual(resumed["previous_side"], "unknown")
        self.assertEqual(resumed["event"], "-")

    def test_new_id_after_switch_does_not_invent_crossing(self):
        counter = CrossingCounter()

        counter.update(10, 40, 50, 5, 1)
        changed_id = counter.update(11, 60, 50, 5, 2)

        self.assertEqual(changed_id["previous_side"], "unknown")
        self.assertEqual(changed_id["event"], "-")

    def test_person_sized_deadband_suppresses_center_jitter_then_allows_crossing(self):
        counter = CrossingCounter()
        line_y = 1000
        person_height = 280
        margin = PeopleCounter.crossing_hysteresis(1994, person_height)

        self.assertEqual(margin, 28)
        counter.update(21, 950, line_y, margin, 1)
        for frame_index, center_y in enumerate((1012, 980, 1018, 990), start=2):
            observation = counter.update(21, center_y, line_y, margin, frame_index)
            self.assertEqual(observation["event"], "-")
            self.assertEqual(observation["stable_side"], "above")

        crossing = counter.update(21, 1035, line_y, margin, 6)
        self.assertEqual(crossing["event"], "downward")


class PeopleCounterTests(unittest.TestCase):
    def test_constructor_accepts_all_application_options(self):
        counter = PeopleCounter(
            model=None,
            conf=0.45,
            line_position=0.62,
            imgsz=480,
            debug=True,
        )

        self.assertIsNone(counter.model)
        self.assertEqual(counter.conf, 0.45)
        self.assertEqual(counter.line_position, 0.62)
        self.assertEqual(counter.imgsz, 480)
        self.assertTrue(counter.debug)

    def test_process_frame_tracks_only_person_and_records_real_box_data(self):
        model = FakeModel()
        counter = PeopleCounter(model=model, debug=True)
        annotated = counter.process_frame(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertEqual(model.track_options["classes"], [0])
        self.assertEqual(model.track_options["tracker"], str(
            Path(__file__).with_name("bytetrack_people.yaml")
        ))
        self.assertEqual(counter.current_people, 1)
        self.assertEqual(counter.seen_ids, {17})
        self.assertEqual(counter.debug_observations[0]["confidence"], 0.83)
        self.assertEqual(counter.debug_observations[0]["bbox"], (10, 20, 30, 60))
        self.assertEqual(counter.debug_observations[0]["center_y"], 40)
        self.assertEqual(counter.debug_observations[0]["hysteresis_px"], 4)
        self.assertEqual(annotated.shape, (100, 100, 3))

    def test_reset_clears_session_metrics(self):
        counter = PeopleCounter(model=None)
        counter.seen_ids.add(42)
        counter.downward_count = 3
        counter.upward_count = 2
        counter.debug_events.append({"event": "downward"})

        counter.reset()

        self.assertEqual(counter.summary(), {
            "current_people": 0,
            "unique_track_ids": 0,
            "downward_crossings": 0,
            "upward_crossings": 0,
            "fps": 0.0,
        })
        self.assertEqual(counter.debug_events, [])


class ArgumentTests(unittest.TestCase):
    def test_parses_webcam_and_video_sources(self):
        self.assertEqual(parse_source("0"), 0)
        self.assertEqual(parse_source("12"), 12)

    def test_rejects_negative_camera_index(self):
        with self.assertRaisesRegex(ValueError, "zero or greater"):
            parse_source("-1")

    def test_rejects_missing_video_file(self):
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            parse_source(str(Path("missing-test-video.mp4")))

    def test_argument_defaults_and_custom_values(self):
        args = parse_args(["--source", "2", "--conf", "0.1", "--line-position", "0.6"])

        self.assertEqual(args.source, "2")
        self.assertEqual(args.conf, 0.1)
        self.assertEqual(args.line_position, 0.6)
        self.assertEqual(args.imgsz, 960)

    def test_detection_defaults_allow_bytetrack_low_score_association(self):
        args = parse_args([])

        self.assertEqual(args.conf, 0.10)
        self.assertEqual(args.imgsz, 960)
        self.assertFalse(args.save)


class OutputWriterTests(unittest.TestCase):
    def test_reports_unwritable_output_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            blocker = Path(folder) / "not-a-directory"
            blocker.write_text("block")

            with self.assertRaisesRegex(RuntimeError, "Could not create output video"):
                create_output_writer(blocker / "output.mp4", 25.0, (640, 480))


if __name__ == "__main__":
    unittest.main()
