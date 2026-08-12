#!/usr/bin/env python3
"""
diagnose_rectification.py
=========================
Standalone diagnostic tool — no ROS required.

Reads one saved left/right frame pair (or grabs live from TCP), applies
rectification, and draws horizontal epipolar lines. If rectification is
correct, every feature in the left image must sit on the SAME horizontal
line in the right image.

Also tries both camera orderings to help detect a left/right swap.

Usage
-----
  # Save a frame pair first (run on the laptop while the Pi is streaming):
  python3 diagnose_rectification.py --calib stereo_calibration_params.json

  # Or point at saved images:
  python3 diagnose_rectification.py --calib stereo_calibration_params.json \
      --left left.png --right right.png
"""

import argparse
import json
import socket
import struct

import cv2
import numpy as np


def load_calibration(path: str):
    with open(path, "r") as f:
        cal = json.load(f)

    K1 = np.array(cal["M1"], dtype=np.float64)
    D1 = np.array(cal["D1"], dtype=np.float64)
    K2 = np.array(cal["M2"], dtype=np.float64)
    D2 = np.array(cal["D2"], dtype=np.float64)
    R1 = np.array(cal["R1"], dtype=np.float64)
    R2 = np.array(cal["R2"], dtype=np.float64)
    P1 = np.array(cal["P1"], dtype=np.float64)
    P2 = np.array(cal["P2"], dtype=np.float64)
    w, h = int(cal["image_size"][0]), int(cal["image_size"][1])

    map1L, map2L = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_16SC2)
    map1R, map2R = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_16SC2)

    print(f"  Calibration loaded: {w}×{h}")
    print(f"  RMS         : {cal.get('rms_stereo', 'n/a')}")
    print(f"  Baseline    : {cal.get('baseline_mm', 'n/a')} mm")
    print(f"  Focal (rect): {P1[0,0]:.2f} px")

    return map1L, map2L, map1R, map2R


def recv_frame_pair(host: str, port: int):
    """Connect to the Pi streamer and grab one stereo pair."""
    buf = b""

    def recv_exactly(sock, n):
        nonlocal buf
        while len(buf) < n:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("disconnected")
            buf += chunk
        data, buf = buf[:n], buf[n:]
        return data

    def recv_frame(sock):
        raw_len = recv_exactly(sock, 4)
        size = struct.unpack("!I", raw_len)[0]
        raw = recv_exactly(sock, size)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("imdecode failed")
        return img

    print(f"Connecting to Pi at {host}:{port} …")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    conn, addr = srv.accept()
    print(f"Pi connected from {addr} — grabbing one frame pair …")
    left  = recv_frame(conn)
    right = recv_frame(conn)
    conn.close()
    srv.close()
    print("Frame pair captured.")
    return left, right


def draw_epipolar(left_raw, right_raw, map1L, map2L, map1R, map2R,
                  label="original", n_lines=12):
    """
    Rectify a frame pair, draw horizontal epipolar lines, and display.
    If rectification is correct every feature aligns on the same scanline
    in both images.
    """
    left_rect  = cv2.remap(left_raw,  map1L, map2L, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_raw, map1R, map2R, cv2.INTER_LINEAR)

    h, w = left_rect.shape[:2]
    canvas = np.hstack([left_rect, right_rect])

    # Draw evenly-spaced horizontal lines across the stereo pair
    step = h // (n_lines + 1)
    for i in range(1, n_lines + 1):
        y = i * step
        color = (0, 255, 0) if i % 2 == 0 else (0, 200, 255)
        cv2.line(canvas, (0, y), (w * 2, y), color, 1)

    # Label
    cv2.putText(canvas, f"Epipolar lines — {label}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "LEFT (rectified)", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(canvas, "RIGHT (rectified)", (w + 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return canvas, left_rect, right_rect


def disparity_quick(left_rect, right_rect, label="") -> np.ndarray:
    """Compute a quick disparity map to assess quality.
    Parameters tuned for this rig: fx=1200.18px, baseline=59.88mm.
      minDisparity=16, numDisparities=176 → covers 0.4m … 4.5m depth.
    """
    gray_l = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    sgbm = cv2.StereoSGBM_create(
        minDisparity=16,       # start above 0 to avoid left-edge black strip
        numDisparities=176,    # covers disparity 16..192 → depth 0.4m..4.5m
        blockSize=7,           # larger block handles JPEG noise better than 5
        P1=8  * 3 * 7 * 7,    # = 1176
        P2=32 * 3 * 7 * 7,    # = 4704
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=150,
        speckleRange=16,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = sgbm.compute(gray_l, gray_r).astype(np.float32) / 16.0
    disp[disp < 0] = float("nan")

    # Normalise for display (ignore NaNs)
    valid = disp[np.isfinite(disp)]
    if valid.size == 0:
        disp_vis = np.zeros(disp.shape, dtype=np.uint8)
    else:
        disp_norm = np.where(np.isfinite(disp), disp, 0)
        disp_vis = cv2.normalize(disp_norm, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)
    # Mark NaN pixels black so invalid regions are obvious
    disp_color[~np.isfinite(disp)] = [0, 0, 0]

    if label:
        cv2.putText(disp_color, f"Disparity — {label}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(disp_color, "black=invalid  warm=near  cool=far", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return disp_color


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", required=True, help="Path to stereo_calibration_params.json")
    parser.add_argument("--left",  default=None,  help="Path to saved left image (skip TCP capture)")
    parser.add_argument("--right", default=None,  help="Path to saved right image")
    parser.add_argument("--host",  default="0.0.0.0", help="Listen address for Pi TCP stream")
    parser.add_argument("--port",  type=int, default=8000)
    parser.add_argument("--save",  action="store_true", help="Save output images to disk")
    args = parser.parse_args()

    print("=== Stereo rectification diagnostics ===")
    print(f"Calibration: {args.calib}")
    map1L, map2L, map1R, map2R = load_calibration(args.calib)

    # ── Acquire frames ─────────────────────────────────────────────────
    if args.left and args.right:
        left_orig  = cv2.imread(args.left)
        right_orig = cv2.imread(args.right)
        if left_orig is None or right_orig is None:
            raise FileNotFoundError("Could not read saved images.")
        print(f"Loaded images: {args.left}, {args.right}")
    else:
        left_orig, right_orig = recv_frame_pair(args.host, args.port)

    # ── Test 1: original camera order ─────────────────────────────────
    print("\n[Test 1] Original camera order (left=cam0, right=cam1)")
    canvas1, l_rect1, r_rect1 = draw_epipolar(
        left_orig, right_orig, map1L, map2L, map1R, map2R,
        label="cam0=left, cam1=right"
    )
    disp1 = disparity_quick(l_rect1, r_rect1, label="cam0=left, cam1=right")

    # ── Test 2: swapped camera order ──────────────────────────────────
    print("[Test 2] Swapped camera order (left=cam1, right=cam0)")
    canvas2, l_rect2, r_rect2 = draw_epipolar(
        right_orig, left_orig, map1L, map2L, map1R, map2R,
        label="cam1=left, cam0=right"
    )
    disp2 = disparity_quick(l_rect2, r_rect2, label="cam1=left, cam0=right")

    # ── Display ────────────────────────────────────────────────────────
    print("\nShowing results. Press any key to cycle through windows.")
    print("  Window 1/2 : epipolar lines (green lines should pass through same features on both sides)")
    print("  Window 3/4 : disparity maps (good = smooth colour gradients following scene depth)")
    print("  → Whichever test shows aligned epipolar lines AND a coherent disparity is the correct camera order.")

    for title, img in [
        ("Test1 — epipolar (cam0=L, cam1=R)",  canvas1),
        ("Test2 — epipolar (cam1=L, cam0=R)",  canvas2),
        ("Test1 — disparity (cam0=L, cam1=R)", disp1),
        ("Test2 — disparity (cam1=L, cam0=R)", disp2),
    ]:
        cv2.imshow(title, img)
        print(f"  [{title}]")

    if args.save:
        cv2.imwrite("epipolar_test1.png",  canvas1)
        cv2.imwrite("epipolar_test2.png",  canvas2)
        cv2.imwrite("disparity_test1.png", disp1)
        cv2.imwrite("disparity_test2.png", disp2)
        print("\nSaved: epipolar_test1/2.png, disparity_test1/2.png")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n=== What to look for ===")
    print("Good epipolar alignment : a point on a green line in the LEFT")
    print("                          image sits on the same green line in RIGHT.")
    print("Good disparity map      : near objects = warm (red/yellow),")
    print("                          far objects  = cool (blue/purple),")
    print("                          smooth transitions, edges follow real object boundaries.")
    print("If only Test 2 looks good → swap cam0/cam1 in camera_streamer_fixed.py")


if __name__ == "__main__":
    main()
