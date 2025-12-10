# extract_echonet_frames.py
# Convert EchoNet-Dynamic videos into grayscale 256x256 PNG frames (flat folder).
# New: --video_frac to randomly select a fraction of videos without needing FileList.

import os, cv2, argparse, pathlib, random
import numpy as np

def center_crop_to_square(img):
    h, w = img.shape[:2]
    if h == w:
        return img
    if h > w:
        m = (h - w) // 2
        return img[m:m+w, :]
    else:
        m = (w - h) // 2
        return img[:, m:m+h]

def process_video(vpath, out_dir, stride=4, max_frames=None, size=256):
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        print(f"[WARN] cannot open: {vpath}")
        return 0
    vid_id = pathlib.Path(vpath).stem
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    fidx  = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (fidx % stride) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sq = center_crop_to_square(gray)
            resized = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
            out_name = f"{vid_id}_f{count+1:04d}.png"
            cv2.imwrite(str(out_dir / out_name), resized)
            count += 1
            if max_frames and count >= max_frames:
                break
        fidx += 1
    cap.release()
    return count

def maybe_read_filelist(filelist_path, splits_csv):
    """Return a set of allowed stems from a FileList (xlsx/csv), restricted to given splits.
       If file cannot be read, return None (so caller can proceed without it)."""
    if not (filelist_path and os.path.exists(filelist_path)):
        return None
    try:
        import pandas as pd
        if str(filelist_path).lower().endswith(".xlsx"):
            df = pd.read_excel(filelist_path)
        else:
            df = pd.read_csv(filelist_path)
        # find split column
        split_col = None
        for cand in ["Split","split","SET","Set"]:
            if cand in df.columns:
                split_col = df[cand].astype(str).str.upper()
                break
        if split_col is None:
            split_col = None  # allow all
        # find filename column
        name_col = None
        for cand in ["Filename","FileName","File","NAME","video","Video"]:
            if cand in df.columns:
                name_col = df[cand].astype(str)
                break
        if name_col is None:
            name_col = df[df.columns[0]].astype(str)
        # normalize names -> stems
        stems = name_col.str.replace(".avi","", regex=False)\
                        .str.replace(".AVI","", regex=False)\
                        .str.replace(".mp4","", regex=False)\
                        .str.replace(".MP4","", regex=False)
        if split_col is not None:
            keep_splits = {s.strip().upper() for s in splits_csv.split(",")}
            stems = [s for s, sp in zip(stems.tolist(), split_col.tolist()) if str(sp).upper() in keep_splits]
        else:
            stems = stems.tolist()
        stems = set(stems)
        print(f"[INFO] FileList filter active: {len(stems)} ids in {{{splits_csv}}}")
        return stems
    except Exception as e:
        print(f"[WARN] Could not read/parse filelist ({e}). Proceeding over ALL videos.")
        return None

def main():
    ap = argparse.ArgumentParser(description="Extract EchoNet-Dynamic frames to a flat PNG folder")
    ap.add_argument("--videos_dir", required=True, help="Path to EchoNet-Dynamic/Videos (top-level)")
    ap.add_argument("--out_dir",    required=True, help="Output folder for frames (flat)")
    ap.add_argument("--stride",     type=int, default=4, help="Keep every Nth frame (larger=fewer frames)")
    ap.add_argument("--max_frames", type=int, default=32, help="Max frames per video (0/None=all)")
    ap.add_argument("--size",       type=int, default=256, help="Output image size")

    # Optional: filter by official list (TRAIN/VAL), if available
    ap.add_argument("--filelist",   type=str, default=None, help="Optional FileList.xlsx/.csv to filter videos")
    ap.add_argument("--splits",     type=str, default="TRAIN,VAL", help="Comma list of splits if using --filelist")

    # NEW: random subsample of videos even without FileList
    ap.add_argument("--video_frac", type=float, default=1.0, help="Fraction of videos to process (0<frac<=1.0)")
    ap.add_argument("--seed",       type=int, default=42, help="Random seed for --video_frac selection")

    # Utility
    ap.add_argument("--dryrun",     type=int, default=0, help="Process only first N videos after filtering (0=all)")
    args = ap.parse_args()

    videos_dir = pathlib.Path(args.videos_dir)
    out_dir    = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional FileList filter
    keep_ids = maybe_read_filelist(args.filelist, args.splits)

    # Robust recursive discovery
    all_files = [p for p in videos_dir.rglob("*") if p.is_file()]
    print(f"[INFO] Found {len(all_files)} filesystem entries under {videos_dir}")

    common_ext = {".mp4",".MP4",".avi",".AVI",".mov",".MOV",".m4v",".M4V",".mpeg",".MPEG",".mpg",".MPG"}
    candidates = [p for p in all_files if (p.suffix in common_ext) or (p.suffix == "")]
    print(f"[INFO] Candidate files after extension filter: {len(candidates)}")

    # Apply FileList filter on stem if present
    if keep_ids is not None:
        before = len(candidates)
        candidates = [p for p in candidates if p.stem in keep_ids]
        print(f"[INFO] After FileList filter: {len(candidates)} / {before} remain")
    else:
        print(f"[INFO] No filelist filter: using {len(candidates)} candidates")

    # Verify readable by OpenCV
    readable = []
    for p in candidates:
        cap = cv2.VideoCapture(str(p))
        if cap.isOpened():
            readable.append(p)
        cap.release()
    print(f"[INFO] Readable videos (cv2): {len(readable)}")

    # Randomly subsample videos if requested
    if args.video_frac < 1.0:
        random.seed(args.seed)
        k = max(1, int(len(readable) * args.video_frac))
        readable = random.sample(readable, k)
        print(f"[INFO] Subsampled by --video_frac={args.video_frac}: keeping {len(readable)} videos")

    # Dry-run
    if args.dryrun and len(readable) > args.dryrun:
        readable = readable[:args.dryrun]
        print(f"[INFO] Dry-run: processing only first {len(readable)} videos")

    # Extract
    total_frames = 0
    max_frames_arg = None if args.max_frames in (0, None) else args.max_frames
    for i, v in enumerate(readable, 1):
        n = process_video(v, out_dir, stride=args.stride, max_frames=max_frames_arg, size=args.size)
        total_frames += n
        print(f"[{i:05d}/{len(readable):05d}] {v.name} -> {n} frames")

    print(f"[DONE] Wrote {total_frames} frames to: {out_dir}")

if __name__ == "__main__":
    main()
