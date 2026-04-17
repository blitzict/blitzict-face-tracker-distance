"""
collect_my_face.py  —  Capture your own face for training
==========================================================

WHAT THIS DOES (step by step)
──────────────────────────────
 Step 1  Opens your webcam and shows a live preview.
 Step 2  Prompts you to stand at a series of distances (1m → 8m).
 Step 3  At each distance: press SPACE to snap a photo (8 shots per distance).
 Step 4  Saves photos to  DnHFaces/open_data_set/photos_all_faces/
         using the DnHFaces filename format so they slot straight into
         the existing training pipeline.

FILENAME FORMAT
───────────────
    <subject>_webcam_<shot>_ef_<dist_id>.jpg
    e.g.   l_webcam_0_ef_05.jpg
           └ subject 'l', shot 0, distance ID 05 = 2.17m

AFTER COLLECTING
────────────────
 1. Add your subject letter to  dataset.py:
        SUBJECT_IDS = ['a', 'b', ..., 'k', 'l']   # append your letter
        NUM_SUBJECTS = len(SUBJECT_IDS)             # updates automatically

 2. Retrain the recognition model:
        python train.py

USAGE
─────
    python collect_my_face.py --subject l --camera 1

CONTROLS
────────
    SPACE   take a photo at the current distance
    N       skip to the next distance
    Q/ESC   quit early
"""

import argparse
from pathlib import Path

import cv2

# ── Which distances to collect ────────────────────────────────────────────────
# Every 3rd distance ID out of 0-30 gives ~11 distinct distances.
# The recogniser interpolates well between them.
COLLECT_IDS     = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
SHOTS_PER_DIST  = 8     # photos to take at each distance

# Physical distance mapping (must match dataset.py constants)
DIST_MIN_M  = 1.0
DIST_STEP_M = 0.2333

# Where to save (same directory the recogniser trains on)
SAVE_DIR = Path('DnHFaces/open_data_set/photos_all_faces')

FONT = cv2.FONT_HERSHEY_DUPLEX
GREEN  = (0, 230, 118)
GREY   = (200, 200, 200)
WHITE  = (255, 255, 255)
ORANGE = (0, 140, 255)


def dist_metres(dist_id: int) -> float:
    return DIST_MIN_M + dist_id * DIST_STEP_M


def collect(subject: str, camera_idx: int):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera {camera_idx}')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    total_saved = 0

    print(f"\nCollecting face data for subject '{subject}'")
    print(f"Save directory : {SAVE_DIR.resolve()}")
    print(f"Distances      : {[f'{dist_metres(d):.1f}m' for d in COLLECT_IDS]}")
    print(f"Photos per dist: {SHOTS_PER_DIST}")
    print(f"Total target   : {len(COLLECT_IDS) * SHOTS_PER_DIST} photos\n")

    for dist_id in COLLECT_IDS:
        metres   = dist_metres(dist_id)
        captured = 0

        print(f"── {metres:.2f}m  (dist ID {dist_id:02d}) ──  stand {metres:.1f}m from camera")

        while captured < SHOTS_PER_DIST:
            ret, frame = cap.read()
            if not ret:
                print('Camera read failed.')
                break

            h, w   = frame.shape[:2]
            disp   = frame.copy()

            # Instruction overlay
            cv2.putText(disp,
                        f"Subject: {subject.upper()}   Distance: {metres:.2f} m  (ID {dist_id:02d})",
                        (10, 32), FONT, 0.65, GREEN, 1, cv2.LINE_AA)
            cv2.putText(disp,
                        f"Shots: {captured}/{SHOTS_PER_DIST}   SPACE=capture   N=next dist   Q=quit",
                        (10, 62), FONT, 0.52, GREY, 1, cv2.LINE_AA)
            cv2.putText(disp,
                        f"Stand {metres:.1f} m from the camera, then press SPACE",
                        (10, 92), FONT, 0.52, ORANGE, 1, cv2.LINE_AA)

            # Progress bar at the bottom
            bar_x2 = 10 + int((w - 20) * captured / SHOTS_PER_DIST)
            cv2.rectangle(disp, (10, h-18), (w-10, h-5), (50, 50, 50), -1)
            if bar_x2 > 10:
                cv2.rectangle(disp, (10, h-18), (bar_x2, h-5), GREEN, -1)

            cv2.imshow('Collect Face Data', disp)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                cap.release()
                cv2.destroyAllWindows()
                print(f"\nStopped early.  Saved {total_saved} photos total.")
                _print_next_steps(subject)
                return

            if key == ord('n'):
                print(f"  Skipped (captured {captured}/{SHOTS_PER_DIST})")
                break

            if key == ord(' '):
                # Filename: subject_webcam_shot_ef_distID.jpg
                # Using shot index in the height field so multiple shots at
                # the same distance get unique filenames that the parser accepts.
                fname = SAVE_DIR / f"{subject}_webcam_{captured}_ef_{dist_id:02d}.jpg"
                cv2.imwrite(str(fname), frame)
                captured   += 1
                total_saved += 1
                print(f"  [{captured}/{SHOTS_PER_DIST}]  Saved  {fname.name}")

                # White flash to confirm capture
                flash = frame.copy()
                cv2.rectangle(flash, (0, 0), (w, h), WHITE, 25)
                cv2.imshow('Collect Face Data', flash)
                cv2.waitKey(60)

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nDone!  Saved {total_saved} photos for subject '{subject}'.")
    _print_next_steps(subject)


def _print_next_steps(subject: str):
    print(f"""
Next steps:
  1. Open dataset.py and add '{subject}' to SUBJECT_IDS:
         SUBJECT_IDS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', '{subject}']

  2. Retrain the recognition model:
         python train.py

  3. Run inference:
         python inference.py --camera 1
""")


def main():
    parser = argparse.ArgumentParser(
        description='Capture your face at various distances to add yourself to training data')
    parser.add_argument('--subject', type=str, required=True,
                        help="Single letter subject ID, e.g. 'l'  (must not already be in SUBJECT_IDS)")
    parser.add_argument('--camera',  type=int, default=0,
                        help='Camera index (default: 0)')
    args = parser.parse_args()

    if len(args.subject) != 1 or not args.subject.isalpha():
        print("Error: --subject must be a single letter, e.g.   --subject l")
        return

    collect(args.subject.lower(), args.camera)


if __name__ == '__main__':
    main()
