# ============================================================
# CCTV SYSTEM ENGINE — FIXED & IMPROVED
# Save this as: cctv_engine.py
# ============================================================

import cv2
import numpy as np
import cctv_engine
import pickle
import time
from collections import defaultdict
from insightface.app import FaceAnalysis


# ─────────────────────────────────────────────────────────────
# DEFAULT CONFIGURATION
# ─────────────────────────────────────────────────────────────
DEFAULT_SIMILARITY_THRESHOLD = 0.40
DEFAULT_COOLDOWN_SECONDS     = 4
PROCESS_EVERY_N              = 3
MAX_TRACK_DISTANCE           = 100
MIN_FACE_SIZE                = 40
# ─────────────────────────────────────────────────────────────


class SimpleCentroidTracker:
    """Tracks faces frame-to-frame by matching centroids."""

    def __init__(self, max_distance=MAX_TRACK_DISTANCE, max_lost=15):
        self.next_id = 0
        self.tracks = {}
        self.max_distance = max_distance
        self.max_lost = max_lost

    def update(self, detections):
        """
        detections: list of (cx, cy, name, conf, bbox)
        returns: list of (track_id, cx, cy, name, conf, bbox)
        """
        if not detections:
            to_delete = []
            for tid in self.tracks:
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.max_lost:
                    to_delete.append(tid)
            for tid in to_delete:
                del self.tracks[tid]
            return []

        # Build distance pairs and sort (greedy nearest matching)
        pairs = []
        for tid, track in self.tracks.items():
            tx, ty = track['center']
            for i, (cx, cy, name, conf, bbox) in enumerate(detections):
                dist = np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
                if dist < self.max_distance:
                    pairs.append((dist, tid, i))
        pairs.sort(key=lambda x: x[0])

        assigned_tracks = {}
        used_detections = set()
        used_tracks = set()

        for dist, tid, det_idx in pairs:
            if tid in used_tracks or det_idx in used_detections:
                continue
            cx, cy, name, conf, bbox = detections[det_idx]
            assigned_tracks[tid] = (cx, cy, name, conf, bbox)
            used_detections.add(det_idx)
            used_tracks.add(tid)
            self.tracks[tid]['center'] = (cx, cy)
            self.tracks[tid]['lost'] = 0
            self.tracks[tid]['bbox'] = bbox
            if name != 'Unknown':
                self.tracks[tid]['name'] = name
                self.tracks[tid]['conf'] = conf

        # Age unmatched tracks
        for tid in self.tracks:
            if tid not in used_tracks:
                self.tracks[tid]['lost'] += 1

        # Create new tracks for unmatched detections
        for i, (cx, cy, name, conf, bbox) in enumerate(detections):
            if i not in used_detections:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'center': (cx, cy),
                    'lost': 0,
                    'name': name,
                    'conf': conf,
                    'bbox': bbox
                }
                assigned_tracks[tid] = (cx, cy, name, conf, bbox)

        # Remove stale tracks
        to_delete = [tid for tid, t in self.tracks.items() if t['lost'] > self.max_lost]
        for tid in to_delete:
            del self.tracks[tid]

        # Return active tracks with best known identity
        result = []
        for tid, (cx, cy, name, conf, bbox) in assigned_tracks.items():
            if tid in self.tracks:
                best_name = self.tracks[tid]['name']
                best_conf = self.tracks[tid]['conf']
                best_bbox = self.tracks[tid]['bbox']
            else:
                best_name = name
                best_conf = conf
                best_bbox = bbox
            result.append((tid, cx, cy, best_name, best_conf, best_bbox))

        return result


class CCTVEngine:
    """Core engine: face detection, recognition, line crossing, counting."""

    def __init__(self, embeddings_path, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
                 cooldown_seconds=DEFAULT_COOLDOWN_SECONDS):
        self.similarity_threshold = similarity_threshold
        self.cooldown_seconds = cooldown_seconds

        print("Loading ArcFace model...")
        self.face_app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        print("ArcFace loaded ✓")

        print(f"Loading embeddings from: {embeddings_path}")
        with open(embeddings_path, 'rb') as f:
            raw = pickle.load(f)

        # Normalize: handle {name: array}, {name: [[...]]} and {name: [emb1, emb2, ...]}
        self.embeddings_db = {}
        for name, emb in raw.items():
            arr = np.array(emb, dtype=np.float32).squeeze()
            if arr.ndim == 1:
                arr = arr / (np.linalg.norm(arr) + 1e-8)
                self.embeddings_db[name] = arr
            elif arr.ndim == 2:
                # Multiple embeddings per person — average them
                arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
                avg = arr.mean(axis=0)
                avg = avg / (np.linalg.norm(avg) + 1e-8)
                self.embeddings_db[name] = avg
                print(f"  {name}: averaged {arr.shape[0]} embeddings")
            else:
                print(f"  ⚠ Skipping {name}: unexpected shape {arr.shape}")

        if not self.embeddings_db:
            raise ValueError("No valid embeddings found in the database!")

        self.known_names = list(self.embeddings_db.keys())
        self.known_embeddings = np.stack(list(self.embeddings_db.values()))
        print(f"Enrolled persons ({len(self.known_names)}): {self.known_names}")

        # Counting state
        self.tracker = SimpleCentroidTracker()
        self.prev_y = {}
        self.cooldown_log = {}
        self.in_count = 0
        self.out_count = 0
        self.crossing_events = []
        self.line_y = None

    def recognize(self, embedding):
        """Cosine similarity match. Returns (name, confidence)."""
        if len(self.known_embeddings) == 0:
            return 'Unknown', 0.0

        emb = embedding / (np.linalg.norm(embedding) + 1e-8)
        similarities = self.known_embeddings @ emb
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= self.similarity_threshold:
            return self.known_names[best_idx], best_score
        return 'Unknown', best_score

    def check_crossing(self, track_id, cy, line_y):
        """Returns 'IN', 'OUT', or None based on crossing direction."""
        if track_id not in self.prev_y:
            self.prev_y[track_id] = cy
            return None

        prev = self.prev_y[track_id]
        self.prev_y[track_id] = cy

        if prev < line_y <= cy:
            return 'IN'
        if prev > line_y >= cy:
            return 'OUT'
        return None

    def draw_hud(self, frame, line_y, line_set):
        """Draw counting line and stats overlay."""
        h, w = frame.shape[:2]

        if line_set:
            cv2.line(frame, (0, line_y), (w, line_y), (0, 220, 255), 2)
            cv2.putText(frame, "COUNTING LINE", (10, line_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
            cv2.putText(frame, "IN", (w - 80, line_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)
            cv2.putText(frame, "OUT", (w - 80, line_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
        else:
            cv2.putText(frame, "Click to set counting line", (w // 2 - 160, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

        inside = max(0, self.in_count - self.out_count)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (200, 100), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, f"IN   : {self.in_count}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 100), 2)
        cv2.putText(frame, f"OUT  : {self.out_count}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 100, 255), 2)
        cv2.putText(frame, f"INSIDE: {inside}", (10, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)

    def draw_face(self, frame, x1, y1, x2, y2, cx, cy, name, conf, direction=None):
        """Draw bounding box, label, and crossing flash."""
        known = name != 'Unknown'
        color = (0, 255, 100) if known else (60, 60, 255)

        if direction == 'IN':
            color = (0, 255, 0)
        elif direction == 'OUT':
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 5, color, -1)

        label = f"{name}  {conf:.2f}" if known else f"Unknown  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        if direction:
            arrow = f"{'↓ IN' if direction == 'IN' else '↑ OUT'}"
            cv2.putText(frame, arrow, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if direction == 'IN' else (0, 0, 255), 2)

    def process_video(self, video_path, line_y_fraction, output_path=None, progress_callback=None):
        """
        Main processing loop.
        line_y_fraction: float 0–1, position of counting line from top.
        Returns report dict.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = cap.get(cv2.CAP_PROP_FPS) or 25
        TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.line_y = int(H * line_y_fraction)
        print(f"Video: {W}x{H} @ {FPS:.1f}fps  |  Total frames: {TOTAL}")
        print(f"Counting line at y={self.line_y} ({line_y_fraction*100:.0f}% from top)")

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, FPS, (W, H))

        frame_idx = 0
        last_detections = []      # FIX: initialized before loop
        flash_frames = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            current_time = time.time()

            # ── Face detection (every Nth frame) ────────────
            if frame_idx % PROCESS_EVERY_N == 0:
                faces = self.face_app.get(frame)
                detections = []

                for face in faces:
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    face_w = x2 - x1
                    face_h = y2 - y1
                    if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
                        continue

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    name, conf = self.recognize(face.normed_embedding)
                    # FIX: bbox stored with detection
                    detections.append((cx, cy, name, conf, (x1, y1, x2, y2)))

                last_detections = detections
            else:
                detections = last_detections

            # ── Tracking + line crossing ─────────────────────
            tracked = self.tracker.update(detections)

            for tid, cx, cy, name, conf, bbox in tracked:
                # FIX: bbox comes from tracker, correctly associated
                x1, y1, x2, y2 = bbox

                direction = None
                crossing = self.check_crossing(tid, cy, self.line_y)

                if crossing:
                    # FIX: cooldown for both known (by name) and unknown (by track_id)
                    can_count = True
                    cooldown_key = name if name != 'Unknown' else f"_track_{tid}"
                    last_t = self.cooldown_log.get(cooldown_key, 0)
                    if current_time - last_t < self.cooldown_seconds:
                        can_count = False
                    else:
                        self.cooldown_log[cooldown_key] = current_time

                    if can_count:
                        direction = crossing
                        if crossing == 'IN':
                            self.in_count += 1
                        else:
                            self.out_count += 1

                        event = {
                            'direction': crossing,
                            'name': name,
                            'confidence': round(conf, 3),
                            'time': time.strftime('%H:%M:%S'),
                            'frame': frame_idx
                        }
                        self.crossing_events.append(event)
                        flash_frames[tid] = (crossing, frame_idx + int(FPS * 0.5))
                        print(f"  [{crossing}] {name} (conf={conf:.3f}) @ frame {frame_idx}")

                # Check flash
                flash_dir = None
                if tid in flash_frames:
                    fdir, fexpire = flash_frames[tid]
                    if frame_idx <= fexpire:
                        flash_dir = fdir
                    else:
                        del flash_frames[tid]

                self.draw_face(frame, x1, y1, x2, y2, cx, cy, name, conf, flash_dir)

            # ── HUD ──────────────────────────────────────────
            self.draw_hud(frame, self.line_y, True)

            if writer:
                writer.write(frame)

            if progress_callback and TOTAL > 0:
                progress_callback(frame_idx / TOTAL)

        cap.release()
        if writer:
            writer.release()

        return self.build_report()

    def build_report(self):
        """Summarize all counting events."""
        attendance = defaultdict(list)
        for ev in self.crossing_events:
            attendance[ev['name']].append(f"{ev['direction']} @ {ev['time']}")

        return {
            'total_in': self.in_count,
            'total_out': self.out_count,
            'current_inside': max(0, self.in_count - self.out_count),
            'events': self.crossing_events,
            'attendance': dict(attendance)
        }
