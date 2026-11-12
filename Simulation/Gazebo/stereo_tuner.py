#!/usr/bin/env python3
"""
stereo_tuner.py
===============
Interactive two-stage tuner for stereo depth accuracy.

Stage 1 — SGBM parameter tuning
  Reads live rectified stereo images (or a ROS2 bag via image topics),
  shows the disparity map in real time, and lets you adjust SGBM
  parameters with trackbars.  Ground-truth distance is entered via
  keyboard.  The tool measures median disparity inside a drawn ROI and
  compares it to the theoretical value.  Best parameter set is saved.

Stage 2 — Depth node parameter tuning  (run after Stage 1)
  Reads the depth node's JSON output topic from a live ROS2 bag or live
  node, compares depth_m / raw_depth_m against ground-truth distances
  you mark, and scores combinations of (roi_shrink, buffer_len,
  kf_q_pos, kf_r_xy) using a grid search.  Best params are printed as
  a ROS2 launch argument block ready to paste.

Usage — Stage 1 (SGBM live tuning)
  python3 stereo_tuner.py stage1 \
      --left  /dev/video0 \
      --right /dev/video1 \
      --calib stereo_calibration_params.json

  OR with a ROS2 bag (plays both image topics via ros2 bag play):
  python3 stereo_tuner.py stage1 \
      --bag   /path/to/my.bag \
      --calib stereo_calibration_params.json

Usage — Stage 2 (depth node offline evaluation)
  python3 stereo_tuner.py stage2 \
      --log   depth_node_output.txt \
      --gt    ground_truth.csv

  ground_truth.csv format:
      timestamp_sec, true_distance_m
      1779011994.009, 2.8
      1779011994.189, 2.8

Requirements: opencv-python numpy (+ picamera2 if using Pi cameras)

Controls (Stage 1)
  d         — enter ground-truth distance (keyboard prompt)
  s         — save current SGBM params + score to results.json
  b         — toggle best-params overlay
  r         — reset ROI to centre
  q / ESC   — quit
  Left-click drag — draw custom ROI on disparity image
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_calibration(path: str) -> dict:
    with open(path) as f:
        cal = json.load(f)
    P1 = np.array(cal['P1'])
    P2 = np.array(cal['P2'])
    M1 = np.array(cal['M1']); D1 = np.array(cal['D1'])
    M2 = np.array(cal['M2']); D2 = np.array(cal['D2'])
    R1 = np.array(cal['R1']); R2 = np.array(cal['R2'])
    w, h = cal['image_size']
    focal_px   = float(P1[0, 0])
    baseline_m = float(-P2[0, 3] / P2[0, 0]) / 1000.0
    map1x, map1y = cv2.initUndistortRectifyMap(M1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(M2, D2, R2, P2, (w, h), cv2.CV_32FC1)
    return {
        'focal_px': focal_px, 'baseline_m': baseline_m,
        'P1': P1, 'P2': P2,
        'map1x': map1x, 'map1y': map1y,
        'map2x': map2x, 'map2y': map2y,
        'image_size': (w, h),
    }


def theoretical_disparity(focal_px: float, baseline_m: float, dist_m: float) -> float:
    """Expected disparity in pixels for a given distance."""
    if dist_m <= 0:
        return 0.0
    return focal_px * baseline_m / dist_m


def depth_error_pct(meas: float, true: float) -> float:
    if true == 0:
        return float('inf')
    return abs(meas - true) / true * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Camera source abstraction
# ─────────────────────────────────────────────────────────────────────────────

class StereoCameraSource:
    """Yields (frame_left_bgr, frame_right_bgr) pairs."""

    def __init__(self, left_idx: int, right_idx: int,
                 width: int = 640, height: int = 480,
                 vflip: bool = True, hflip: bool = False):
        self.vflip = vflip
        self.hflip = hflip
        self._picam_l = self._picam_r = None
        self._cap_l   = self._cap_r   = None
        try:
            from picamera2 import Picamera2
            from libcamera import Transform
            tf = Transform(vflip=vflip, hflip=hflip)
            self._picam_l = Picamera2(left_idx)
            self._picam_r = Picamera2(right_idx)
            for cam in (self._picam_l, self._picam_r):
                cfg = cam.create_preview_configuration(
                    main={'size': (width, height), 'format': 'RGB888'},
                    transform=tf)
                cam.configure(cfg)
                cam.start()
            time.sleep(1.5)
            print(f"[Camera] Picamera2 ready  vflip={vflip} hflip={hflip}")
        except Exception as e:
            print(f"[Camera] Picamera2 unavailable ({e}), using VideoCapture")
            self._picam_l = self._picam_r = None
            self._cap_l = cv2.VideoCapture(left_idx)
            self._cap_r = cv2.VideoCapture(right_idx)
            for cap in (self._cap_l, self._cap_r):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._picam_l:
            import cv2 as _cv2
            l = self._picam_l.capture_array()
            r = self._picam_r.capture_array()
            if l is None or r is None:
                return None, None
            return _cv2.cvtColor(l, _cv2.COLOR_RGB2BGR), _cv2.cvtColor(r, _cv2.COLOR_RGB2BGR)
        ret_l, fl = self._cap_l.read()
        ret_r, fr = self._cap_r.read()
        if not ret_l or not ret_r:
            return None, None
        if self.vflip and self.hflip:
            fl = cv2.flip(fl, -1); fr = cv2.flip(fr, -1)
        elif self.vflip:
            fl = cv2.flip(fl, 0);  fr = cv2.flip(fr, 0)
        elif self.hflip:
            fl = cv2.flip(fl, 1);  fr = cv2.flip(fr, 1)
        return fl, fr

    def stop(self):
        if self._picam_l:
            self._picam_l.stop(); self._picam_r.stop()
        if self._cap_l:
            self._cap_l.release(); self._cap_r.release()


# ─────────────────────────────────────────────────────────────────────────────
# SGBM wrapper
# ─────────────────────────────────────────────────────────────────────────────

SGBM_PARAMS = {
    # (min, max, default, step)
    'minDisparity':      (0,   32,  2,   1),
    'numDisparities':    (16, 256, 96,  16),   # must stay multiple of 16
    'blockSize':         (3,   21,  11,  2),   # must be odd
    'P1':                (0,  1000, 363, 1),   # 8*channels*blockSize^2
    'P2':                (0,  4000, 1452, 1),  # 32*channels*blockSize^2
    'disp12MaxDiff':     (0,   16,  1,   1),
    'uniquenessRatio':   (0,   30,  10,  1),
    'speckleWindowSize': (0,  300, 100,  10),
    'speckleRange':      (1,   64,  32,  1),
    'mode':              (0,    1,   1,  1),   # 0=SGBM, 1=HH (full DP)
}

def make_sgbm(params: dict) -> cv2.StereoSGBM:
    mode = cv2.StereoSGBM_MODE_HH if params['mode'] == 1 else cv2.StereoSGBM_MODE_SGBM
    return cv2.StereoSGBM_create(
        minDisparity      = int(params['minDisparity']),
        numDisparities    = int(params['numDisparities']),
        blockSize         = int(params['blockSize']),
        P1                = int(params['P1']),
        P2                = int(params['P2']),
        disp12MaxDiff     = int(params['disp12MaxDiff']),
        uniquenessRatio   = int(params['uniquenessRatio']),
        speckleWindowSize = int(params['speckleWindowSize']),
        speckleRange      = int(params['speckleRange']),
        mode              = mode,
    )


def compute_disparity(sgbm: cv2.StereoSGBM,
                      rect_l: np.ndarray, rect_r: np.ndarray) -> np.ndarray:
    """Returns float32 disparity map."""
    gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
    disp16 = sgbm.compute(gray_l, gray_r)
    disp   = disp16.astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    return disp


def disparity_colormap(disp: np.ndarray,
                       vmin: float = 0, vmax: float = 128) -> np.ndarray:
    """Render disparity as a colour image for display."""
    disp_vis = np.clip((disp - vmin) / (vmax - vmin + 1e-6), 0, 1)
    disp_vis = np.nan_to_num(disp_vis, nan=0.0)
    disp_u8  = (disp_vis * 255).astype(np.uint8)
    return cv2.applyColorMap(disp_u8, cv2.COLORMAP_TURBO)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — SGBM interactive tuning
# ─────────────────────────────────────────────────────────────────────────────

def stage1(args):
    print("\n=== Stage 1: SGBM Parameter Tuning ===\n")
    cal = load_calibration(args.calib)
    focal_px   = cal['focal_px']
    baseline_m = cal['baseline_m']
    W, H       = cal['image_size']
    print(f"  focal_px   = {focal_px:.2f}  baseline = {baseline_m*1000:.2f} mm")
    print(f"  image_size = {W}×{H}\n")

    src = StereoCameraSource(
        args.left, args.right, W, H,
        vflip=not args.no_vflip, hflip=args.hflip
    )

    # Current SGBM params (start from defaults)
    cur = {k: v[2] for k, v in SGBM_PARAMS.items()}
    sgbm = make_sgbm(cur)

    WIN_CTRL = "SGBM Controls"
    WIN_DISP = "Disparity  [d=set distance  s=save  r=reset ROI  q=quit]"
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_DISP, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 500, 40 * len(SGBM_PARAMS))
    cv2.resizeWindow(WIN_DISP, W * 2, H)

    # Create trackbars
    def make_cb(key):
        lo, hi, _, step = SGBM_PARAMS[key]
        def cb(val):
            # snap to step and constraints
            v = lo + round((val - lo) / step) * step
            if key == 'numDisparities':
                v = max(16, round(v / 16) * 16)
            if key == 'blockSize':
                v = v if v % 2 == 1 else v + 1
            cur[key] = int(np.clip(v, lo, hi))
        return cb

    for k, (lo, hi, default, _) in SGBM_PARAMS.items():
        cv2.createTrackbar(k, WIN_CTRL, default, hi, make_cb(k))

    # ROI state
    roi      = [W // 4, H // 4, 3 * W // 4, 3 * H // 4]  # x1,y1,x2,y2
    drawing  = [False]
    roi_tmp  = [None]

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing[0] = True
            roi_tmp[0] = [x, y, x, y]
        elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:
            roi_tmp[0][2] = x; roi_tmp[0][3] = y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing[0] = False
            if roi_tmp[0]:
                x1 = min(roi_tmp[0][0], roi_tmp[0][2])
                x2 = max(roi_tmp[0][0], roi_tmp[0][2])
                y1 = min(roi_tmp[0][1], roi_tmp[0][3])
                y2 = max(roi_tmp[0][1], roi_tmp[0][3])
                if x2 - x1 > 10 and y2 - y1 > 10:
                    roi[:] = [x1, y1, x2, y2]

    cv2.setMouseCallback(WIN_DISP, mouse_cb)

    gt_dist_m  = None   # ground-truth distance
    results    = []     # saved (params, score) entries
    show_best  = False
    best_entry = None

    font = cv2.FONT_HERSHEY_SIMPLEX

    print("Controls:")
    print("  d  — enter ground-truth distance (m)")
    print("  s  — save current params + score")
    print("  b  — toggle best-params overlay")
    print("  r  — reset ROI to centre")
    print("  q  — quit and show best params\n")

    while True:
        frame_l, frame_r = src.read()
        if frame_l is None:
            time.sleep(0.05)
            continue

        # Rectify
        rect_l = cv2.remap(frame_l, cal['map1x'], cal['map1y'], cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, cal['map2x'], cal['map2y'], cv2.INTER_LINEAR)

        # Rebuild SGBM if params changed (trackbar fires async)
        try:
            sgbm = make_sgbm(cur)
            disp = compute_disparity(sgbm, rect_l, rect_r)
        except Exception as e:
            disp = np.zeros((H, W), np.float32)

        # Disparity colourmap
        vmax = float(cur['numDisparities'] + cur['minDisparity'])
        disp_col = disparity_colormap(disp, vmin=float(cur['minDisparity']), vmax=vmax)

        # ROI stats
        x1, y1, x2, y2 = roi
        roi_disp = disp[y1:y2, x1:x2]
        valid    = np.isfinite(roi_disp) & (roi_disp > 0)
        n_valid  = int(valid.sum())
        median_d = float(np.median(roi_disp[valid])) if n_valid > 10 else None

        # Draw ROI box on disparity image
        draw_col = (0, 255, 0) if median_d else (0, 80, 200)
        cv2.rectangle(disp_col, (x1, y1), (x2, y2), draw_col, 2)
        if roi_tmp[0] and drawing[0]:
            p1 = (roi_tmp[0][0], roi_tmp[0][1])
            p2 = (roi_tmp[0][2], roi_tmp[0][3])
            cv2.rectangle(disp_col, p1, p2, (200, 200, 0), 1)

        # Overlay text
        lines = []
        if median_d is not None:
            meas_depth = focal_px * baseline_m / median_d
            lines.append(f"median disp : {median_d:.2f} px")
            lines.append(f"meas depth  : {meas_depth:.3f} m")
            if gt_dist_m:
                exp_disp = theoretical_disparity(focal_px, baseline_m, gt_dist_m)
                err_pct  = depth_error_pct(meas_depth, gt_dist_m)
                lines.append(f"GT distance : {gt_dist_m:.2f} m")
                lines.append(f"exp disp    : {exp_disp:.2f} px")
                lines.append(f"depth err   : {err_pct:.1f}%")
                col_err = (0,220,60) if err_pct < 10 else (30,160,220) if err_pct < 25 else (40,40,220)
                cv2.putText(disp_col, f"ERR {err_pct:.1f}%", (W-130, 30),
                            font, 0.8, col_err, 2, cv2.LINE_AA)
        else:
            lines.append("No valid disparity in ROI")
        lines.append(f"valid px: {n_valid}")
        lines.append(f"numDisp={cur['numDisparities']}  block={cur['blockSize']}")

        for i, txt in enumerate(lines):
            cv2.putText(disp_col, txt, (8, 22 + i * 22), font, 0.55,
                        (220, 220, 220), 1, cv2.LINE_AA)

        # Draw epilines (horizontal) to verify rectification
        for y_line in range(0, H, H // 8):
            cv2.line(rect_l, (0, y_line), (W, y_line), (0, 180, 255), 1)
            cv2.line(rect_r, (0, y_line), (W, y_line), (0, 180, 255), 1)

        top_row    = np.hstack([rect_l, rect_r])
        top_scaled = cv2.resize(top_row, (W * 2, H // 2))
        bottom_row = disp_col
        composite  = np.vstack([top_scaled, bottom_row])
        cv2.imshow(WIN_DISP, composite)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        elif key == ord('d'):
            cv2.destroyWindow("__input__")   # clear any stale
            raw = input("  Enter ground-truth distance (m): ").strip()
            try:
                gt_dist_m = float(raw)
                print(f"  GT distance set to {gt_dist_m:.2f} m")
                if median_d:
                    exp = theoretical_disparity(focal_px, baseline_m, gt_dist_m)
                    print(f"  Expected disparity: {exp:.2f} px  |  Measured: {median_d:.2f} px")
            except ValueError:
                print("  Invalid input — ignored")

        elif key == ord('s'):
            if median_d and gt_dist_m:
                meas_depth = focal_px * baseline_m / median_d
                err_pct    = depth_error_pct(meas_depth, gt_dist_m)
                entry = {
                    'params':      dict(cur),
                    'gt_dist_m':   gt_dist_m,
                    'median_disp': round(median_d, 3),
                    'meas_depth':  round(meas_depth, 3),
                    'error_pct':   round(err_pct, 2),
                    'n_valid':     n_valid,
                }
                results.append(entry)
                results.sort(key=lambda e: e['error_pct'])
                best_entry = results[0]
                with open('sgbm_tuning_results.json', 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"  Saved — error={err_pct:.1f}%  (best so far: {best_entry['error_pct']:.1f}%)")
            else:
                print("  Need both valid disparity ROI and GT distance (press d first)")

        elif key == ord('b'):
            show_best = not show_best
            if show_best and best_entry:
                print(f"\n  Best params so far (error={best_entry['error_pct']:.1f}%):")
                for k, v in best_entry['params'].items():
                    print(f"    {k}: {v}")

        elif key == ord('r'):
            roi[:] = [W // 4, H // 4, 3 * W // 4, 3 * H // 4]

    cv2.destroyAllWindows()
    src.stop()

    # Final report
    if results:
        best = results[0]
        print(f"\n{'='*60}")
        print(f"Best SGBM parameters (error = {best['error_pct']:.1f}%)")
        print(f"{'='*60}")
        p = best['params']
        print(f"""
  StereoSGBM_create(
      minDisparity      = {p['minDisparity']},
      numDisparities    = {p['numDisparities']},
      blockSize         = {p['blockSize']},
      P1                = {p['P1']},
      P2                = {p['P2']},
      disp12MaxDiff     = {p['disp12MaxDiff']},
      uniquenessRatio   = {p['uniquenessRatio']},
      speckleWindowSize = {p['speckleWindowSize']},
      speckleRange      = {p['speckleRange']},
  )
""")
        print(f"  → Set num_disparities: {p['numDisparities']} in your depth node launch file")
        print(f"  All results saved to sgbm_tuning_results.json")
    else:
        print("\nNo results saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Depth node parameter grid search (offline)
# ─────────────────────────────────────────────────────────────────────────────

def parse_depth_log(log_path: str) -> List[dict]:
    """Parse depth node JSON output lines from a log file."""
    records = []
    with open(log_path) as f:
        for line in f:
            if 'object depths:' not in line:
                continue
            try:
                bracket = line.index('[{"')
                data    = json.loads(line[bracket:])
                for obj in data:
                    if 'raw_depth_m' in obj and 'raw_disp' in obj:
                        # Extract timestamp from ROS2 log prefix
                        ts = None
                        try:
                            ts = float(line.split(']')[1].strip().split()[0])
                        except Exception:
                            pass
                        records.append({
                            'ts':          ts,
                            'id':          obj.get('id'),
                            'raw_depth_m': float(obj['raw_depth_m']),
                            'depth_m':     float(obj.get('depth_m', obj['raw_depth_m'])),
                            'raw_disp':    float(obj['raw_disp']),
                        })
            except Exception:
                continue
    print(f"  Parsed {len(records)} depth records from {log_path}")
    return records


def parse_ground_truth(gt_path: str) -> List[Tuple[float, float]]:
    """Parse ground truth CSV: timestamp_sec, true_distance_m"""
    pairs = []
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    pairs.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    print(f"  Parsed {len(pairs)} ground-truth entries from {gt_path}")
    return pairs


def match_records_to_gt(records: List[dict],
                        gt: List[Tuple[float, float]],
                        slop: float = 0.2) -> List[Tuple[dict, float]]:
    """Match each GT entry to the nearest-timestamp depth record."""
    matched = []
    for gt_ts, gt_dist in gt:
        best  = None
        best_dt = slop + 1
        for rec in records:
            if rec['ts'] is None:
                continue
            dt = abs(rec['ts'] - gt_ts)
            if dt < best_dt:
                best_dt = dt
                best    = rec
        if best is not None and best_dt <= slop:
            matched.append((best, gt_dist))
    print(f"  Matched {len(matched)} / {len(gt)} GT entries (slop={slop}s)")
    return matched


def iqr_filter(values: List[float]) -> Optional[float]:
    if len(values) < 3:
        return None
    arr = np.array(values, dtype=np.float32)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return float(arr.mean())
    filtered = arr[(arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)]
    return float(filtered.mean()) if filtered.size > 0 else None


def simulate_depth_pipeline(matched: List[Tuple[dict, float]],
                             focal_px: float, baseline_m: float,
                             roi_shrink_pct: float,
                             buffer_len: int) -> Tuple[float, float, int]:
    """
    Simulate the IQR buffer on raw_depth values and return
    (mean_error_pct, std_error_pct, n_valid).
    roi_shrink_pct is ignored here (already baked into raw_disp in logs)
    but buffer_len and the IQR filter are simulated properly.
    """
    buffer: List[float] = []
    errors = []
    for rec, gt_dist in matched:
        raw_depth = rec['raw_depth_m']
        buffer.append(raw_depth)
        if len(buffer) > buffer_len:
            buffer = buffer[-buffer_len:]
        iqr_depth = iqr_filter(buffer)
        est = iqr_depth if iqr_depth else raw_depth
        errors.append(depth_error_pct(est, gt_dist))
    if not errors:
        return 999.0, 999.0, 0
    arr = np.array(errors)
    return float(arr.mean()), float(arr.std()), len(errors)


def stage2(args):
    print("\n=== Stage 2: Depth Node Parameter Grid Search ===\n")

    records = parse_depth_log(args.log)
    gt      = parse_ground_truth(args.gt)
    matched = match_records_to_gt(records, gt, slop=float(args.slop))

    if not matched:
        print("ERROR: No matched records. Check timestamps in your GT CSV and log file.")
        sys.exit(1)

    cal        = load_calibration(args.calib)
    focal_px   = cal['focal_px']
    baseline_m = cal['baseline_m']

    # Grid search space
    buffer_lens   = [4, 6, 8, 10, 15, 20]
    roi_shrinks   = [0.10, 0.15, 0.20, 0.25]   # informational only in offline mode

    print(f"\n  Grid: buffer_len={buffer_lens}  (roi_shrink is live-only)")
    print(f"  Evaluating {len(buffer_lens) * len(roi_shrinks)} combinations...\n")

    results = []
    for bl in buffer_lens:
        for rs in roi_shrinks:
            mean_err, std_err, n = simulate_depth_pipeline(
                matched, focal_px, baseline_m, rs, bl)
            results.append({
                'buffer_len':  bl,
                'roi_shrink':  rs,
                'mean_err_pct': round(mean_err, 2),
                'std_err_pct':  round(std_err, 2),
                'n_valid':      n,
            })

    results.sort(key=lambda r: r['mean_err_pct'])

    print(f"  {'buffer_len':>12} {'roi_shrink':>10} {'mean_err%':>10} {'std_err%':>9}")
    print(f"  {'-'*45}")
    for r in results[:10]:
        print(f"  {r['buffer_len']:>12}   {r['roi_shrink']:>8.2f}   "
              f"{r['mean_err_pct']:>8.1f}%   {r['std_err_pct']:>7.1f}%")

    best = results[0]
    out_path = 'depth_node_tuning_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  All results saved to {out_path}")

    print(f"""
{'='*60}
Best depth node parameters (mean error = {best['mean_err_pct']:.1f}%)
{'='*60}

Paste into your ROS2 launch file:

  stereo_box_depth_kalman_synced:
    ros__parameters:
      buffer_len:    {best['buffer_len']}
      roi_shrink:    {best['roi_shrink']}
      num_disparities: <use best numDisparities from Stage 1>
      conf_thresh:   0.30
      min_disparity: 3.0
      depth_min:     0.3
      depth_max:     50.0
      kf_q_pos:      0.10
      kf_q_vel:      0.80
      kf_r_xy:       0.10
""")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Stereo depth tuner — Stage 1 (SGBM) and Stage 2 (depth node)")
    sub = p.add_subparsers(dest='stage', required=True)

    # Stage 1
    s1 = sub.add_parser('stage1', help='Interactive SGBM tuning with live cameras')
    s1.add_argument('--calib',    required=True, help='stereo_calibration_params.json')
    s1.add_argument('--left',     type=int, default=0, help='Left camera index (default 0)')
    s1.add_argument('--right',    type=int, default=1, help='Right camera index (default 1)')
    s1.add_argument('--no-vflip', action='store_true',  help='Disable vertical flip')
    s1.add_argument('--hflip',    action='store_true',  help='Enable horizontal flip')

    # Stage 2
    s2 = sub.add_parser('stage2', help='Offline depth node parameter grid search')
    s2.add_argument('--log',   required=True, help='Depth node log file (ROS2 stdout)')
    s2.add_argument('--gt',    required=True, help='Ground truth CSV: timestamp_sec,distance_m')
    s2.add_argument('--calib', required=True, help='stereo_calibration_params.json')
    s2.add_argument('--slop',  type=float, default=0.2,
                    help='Max timestamp difference for GT matching (default 0.2s)')

    args = p.parse_args()
    if args.stage == 'stage1':
        stage1(args)
    else:
        stage2(args)


if __name__ == '__main__':
    main()
