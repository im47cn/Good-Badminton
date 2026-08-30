from collections import deque

import numpy as np

from ..court.mapper import CourtMapper


class PlayerTracker:
    """
    Player tracking system.

    Tracks player positions, court coordinates, movement statistics, and writes
    one structured detection record per processed court frame.
    """

    def __init__(self, corners, threshold=680, history_size=50, detection_writer=None, fps=30):
        self.threshold = threshold
        self.fps = fps
        self.detection_writer = detection_writer
        self.max_frame_distance = 8.0 / self.fps

        self.players = {
            "upper": None,
            "lower": None,
        }
        self.history = {
            "upper": deque(maxlen=history_size),
            "lower": deque(maxlen=history_size),
        }
        self.court_history = {
            "upper": deque(maxlen=history_size),
            "lower": deque(maxlen=history_size),
        }

        self.match_stats = {
            "upper": {"total_distance": 0, "max_speed": 0, "total_frames": 0},
            "lower": {"total_distance": 0, "max_speed": 0, "total_frames": 0},
        }
        self.rally_stats = {
            "upper": {"total_distance": 0, "max_speed": 0, "total_frames": 0},
            "lower": {"total_distance": 0, "max_speed": 0, "total_frames": 0},
        }
        self.current_speed = {
            "upper": 0,
            "lower": 0,
        }

        self.court_mapper = CourtMapper(corners)

    def _empty_player_record(self):
        return {
            "image": None,
            "court": None,
            "speed": None,
            "hands": {
                "left": None,
                "right": None,
            },
        }

    def _initialize_player_record(self):
        return {
            "upper": self._empty_player_record(),
            "lower": self._empty_player_record(),
        }

    def _point_or_none(self, point, zero_is_none=False):
        if point is None:
            return None
        try:
            x, y = point[0], point[1]
        except (TypeError, IndexError):
            return None
        if x is None or y is None:
            return None
        if zero_is_none and float(x) == 0.0 and float(y) == 0.0:
            return None
        return [float(x), float(y)]

    def write_detection_record(self, frame_index, players_record, ball_image_position, detect_frame_count):
        if self.detection_writer is None:
            return

        record = {
            "schema_version": "1.0",
            "frame": int(frame_index),
            "time_sec": round(frame_index / self.fps, 6) if self.fps else None,
            "detect_frame": int(detect_frame_count),
            "players": players_record,
            "shuttlecock": {
                "image": self._point_or_none(ball_image_position, zero_is_none=True),
            },
        }
        self.detection_writer.write(record)

    def update(self, frame_index, centroids, ball_image_position, left_hand_positions, right_hand_positions, detect_frame_count):
        players_record = self._initialize_player_record()

        for region in ["upper", "lower"]:
            if self.players[region] is not None:
                self.match_stats[region]["total_frames"] += 1
                self.rally_stats[region]["total_frames"] += 1

        upper_court_centroids = []
        lower_court_centroids = []
        for centroid in centroids:
            if centroid[1] < self.threshold:
                upper_court_centroids.append(centroid)
            else:
                lower_court_centroids.append(centroid)

        if len(upper_court_centroids) > 1:
            upper_court_centroids.sort(key=lambda p: -p[1])
            upper_court_centroids = [upper_court_centroids[0]]

        filtered_centroids = upper_court_centroids + lower_court_centroids

        for centroid in filtered_centroids:
            try:
                region = "upper" if centroid[1] < self.threshold else "lower"
                left_hand = left_hand_positions.get(centroid[1])
                right_hand = right_hand_positions.get(centroid[1])
                self._update_player_position(region, centroid, left_hand, right_hand, players_record)
            except Exception as exc:
                print(f"Error processing player position: {exc}")
                import traceback
                traceback.print_exc()

        self.write_detection_record(frame_index, players_record, ball_image_position, detect_frame_count)
        return self.players

    def _update_player_position(self, region, centroid, left_hand_pos, right_hand_pos, players_record):
        self.players[region] = centroid
        self.history[region].append(centroid)

        court_position = self.court_mapper.image_to_court(centroid)
        self.court_history[region].append(court_position)

        player_record = players_record[region]
        player_record["image"] = self._point_or_none(centroid)
        player_record["court"] = self._point_or_none(court_position)
        player_record["speed"] = float(self.current_speed[region])
        if left_hand_pos:
            player_record["hands"]["left"] = self._point_or_none(left_hand_pos)
        if right_hand_pos:
            player_record["hands"]["right"] = self._point_or_none(right_hand_pos)

    def _update_rally_and_match_stats(self, region, distance, speed):
        capped_speed = round(min(speed, 8.0), 2)

        self.rally_stats[region]["total_distance"] += distance
        self.rally_stats[region]["max_speed"] = max(self.rally_stats[region]["max_speed"], capped_speed)
        self.current_speed[region] = capped_speed

        self.match_stats[region]["total_distance"] += distance
        self.match_stats[region]["max_speed"] = max(self.match_stats[region]["max_speed"], capped_speed)
        self.current_speed[region] = capped_speed

    def start_new_rally(self):
        for region in ["upper", "lower"]:
            self.rally_stats[region]["total_distance"] = 0
            self.rally_stats[region]["max_speed"] = 0
            self.rally_stats[region]["total_frames"] = 0

    def get_player_movement_stats(self):
        stats = {}
        for region in ["upper", "lower"]:
            history = [pos for pos in list(self.court_history[region]) if pos is not None]
            stats[region] = self._compute_region_stats(region, history)
        return stats

    def _compute_region_stats(self, region, history):
        """Compute movement stats for one court region."""
        region_stats = {
            "current_speed": 0,
            "rally_avg_speed": 0,
            "rally_max_speed": 0,
            "rally_distance": 0,
            "match_avg_speed": 0,
            "match_max_speed": 0,
            "match_distance": 0,
            "position_count": len(history),
        }
        if len(history) < 2:
            return region_stats

        rally_stats = self.rally_stats[region]
        match_stats = self.match_stats[region]
        region_stats["current_speed"] = round(self._sample_current_speed(region, history), 2)
        region_stats["rally_avg_speed"] = round(self._average_speed(rally_stats), 2)
        region_stats["rally_max_speed"] = round(rally_stats["max_speed"], 2)
        region_stats["rally_distance"] = round(rally_stats["total_distance"], 2)
        region_stats["match_avg_speed"] = round(self._average_speed(match_stats), 2)
        region_stats["match_max_speed"] = round(match_stats["max_speed"], 2)
        region_stats["match_distance"] = round(match_stats["total_distance"], 2)
        return region_stats

    def _sample_current_speed(self, region, history):
        """Estimate current speed from recent positions and update rally/match stats."""
        current_time = len(history) - 1
        window_start = max(0, current_time - int(self.fps / 2))
        sample_interval = 5

        if current_time - window_start < sample_interval:
            sample_points = [window_start, current_time]
        else:
            sample_points = list(range(window_start, current_time + 1, sample_interval))
            if current_time not in sample_points:
                sample_points.append(current_time)

        half_second_total_distance = 0
        valid_frames = 0
        actual_time_span = 0
        for i in range(len(sample_points) - 1):
            idx1 = sample_points[i]
            idx2 = sample_points[i + 1]
            p1 = np.array(history[idx1])
            p2 = np.array(history[idx2])
            distance = np.linalg.norm(p2 - p1)
            time_span = (idx2 - idx1) / self.fps
            max_possible_distance = self.max_frame_distance * (idx2 - idx1)

            if distance > 0.05 and distance < max_possible_distance:
                half_second_total_distance += distance
                valid_frames += 1
                actual_time_span += time_span

        current_speed = 0
        if valid_frames > 0 and actual_time_span > 0:
            current_speed = half_second_total_distance / actual_time_span
            self._update_rally_and_match_stats(region, half_second_total_distance, current_speed)

        return min(current_speed, 8.0)

    def _average_speed(self, region_stats):
        """Average speed over the accumulated frames of a stat bucket."""
        frames = region_stats["total_frames"]
        if frames > 1 and self.fps > 0:
            elapsed = frames / self.fps
            return region_stats["total_distance"] / elapsed if elapsed > 0 else 0
        return 0
    def get_player_trajectories(self):
        return {region: list(history) for region, history in self.history.items()}

    def close(self):
        if self.detection_writer is not None:
            self.detection_writer.close()
