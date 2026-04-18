"""
download_datasets.py  —  Download face datasets for detector training
======================================================================

DATASETS
────────
 LFW  (Labeled Faces in the Wild)
   13,233 tight face crops — diverse people, good general positives.
   Size  : ~173 MB

 WIDER FACE
   32,203 images with 393,703 annotated faces at wildly varying scales,
   occlusions and distances.
   Size  : ~1.5 GB (train split only). Requires Kaggle CLI.

 UTKFace
   23k face crops with explicit age / gender / ethnicity labels. Good
   demographic spread — critical for a detector that has to work on
   arbitrary people, not just the celebrity distribution in CelebA.
   Size  : ~100 MB. Requires Kaggle CLI.

 FairFace
   108k faces balanced across 7 race groups × age × gender. The best
   single dataset for defeating demographic bias. Authors publish on
   GitHub / Google Drive — no clean Kaggle mirror, so we print manual
   instructions instead of auto-downloading.
   Size  : ~550 MB.

USAGE
─────
    python download_datasets.py                  # download LFW only
    python download_datasets.py --wider          # also download WIDER FACE
    python download_datasets.py --utk            # also download UTKFace
    python download_datasets.py --fairface       # show FairFace instructions
    python download_datasets.py --all            # LFW + WIDER + UTKFace
    python download_datasets.py --dest datasets
"""

import argparse
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path


# ── LFW mirrors ───────────────────────────────────────────────────────────────
LFW_URLS = [
    'http://vis-www.cs.umass.edu/lfw/lfw.tgz',
    'https://ndownloader.figshare.com/files/5976018',
    'https://github.com/italojs/facial-landmarks-recognition/raw/master/lfw.tgz',
]


def _progress(count, block_size, total):
    done = count * block_size
    pct  = min(100, done * 100 // max(total, 1))
    mb   = done / 1_000_000
    print(f'\r  {pct:3d}%  {mb:6.1f} MB', end='', flush=True)


# ── LFW ───────────────────────────────────────────────────────────────────────

def download_lfw(dest_dir: str = 'datasets') -> Path:
    """Download and extract LFW into <dest_dir>/lfw/."""
    dest     = Path(dest_dir)
    lfw_dir  = dest / 'lfw'
    tgz_path = dest / 'lfw.tgz'

    if lfw_dir.exists():
        n = sum(1 for _ in lfw_dir.glob('**/*.jpg'))
        print(f'LFW already present at {lfw_dir} ({n:,} images) — skipping.')
        return lfw_dir

    dest.mkdir(parents=True, exist_ok=True)
    downloaded = False
    for url in LFW_URLS:
        try:
            print(f'Trying {url}')
            urllib.request.urlretrieve(url, str(tgz_path), reporthook=_progress)
            print()
            downloaded = True
            break
        except Exception as e:
            print(f'\n  Failed: {e}')
            if tgz_path.exists():
                tgz_path.unlink()

    if not downloaded:
        print('\nAll LFW mirrors failed. Manual download:')
        print('  https://www.kaggle.com/datasets/jessicali9530/lfw-dataset')
        print('  Extract so that  datasets/lfw/<person>/<image>.jpg  exists')
        return None

    print(f'Extracting to {lfw_dir} ...')
    with tarfile.open(tgz_path, 'r:gz') as tar:
        tar.extractall(dest)
    tgz_path.unlink()

    n = sum(1 for _ in lfw_dir.glob('**/*.jpg'))
    print(f'Done. {n:,} LFW images saved to {lfw_dir}')
    return lfw_dir


# ── WIDER FACE ────────────────────────────────────────────────────────────────

WIDER_KAGGLE_DATASET = 'wardaddy24/wider-face-dataset-for-object-detection'

def download_wider_face(dest_dir: str = 'datasets') -> Path:
    """
    Download WIDER FACE train split via Kaggle API.

    Expected result:
        datasets/wider_face/images/<category>/<image>.jpg
        datasets/wider_face/wider_face_train_bbx_gt.txt

    Requires: pip install kaggle
    And:      ~/.kaggle/kaggle.json  (API key from kaggle.com → Account → API)
    """
    dest      = Path(dest_dir)
    wider_dir = dest / 'wider_face'

    if wider_dir.exists() and any(wider_dir.rglob('*.jpg')):
        n = sum(1 for _ in wider_dir.rglob('*.jpg'))
        print(f'WIDER FACE already present ({n:,} images) — skipping.')
        return wider_dir

    # Check for kaggle CLI
    if shutil.which('kaggle') is None:
        print('\nKaggle CLI not found. Install it:')
        print('  pip install kaggle')
        print('  Then get your API key from https://www.kaggle.com/settings')
        print('  Save it to  ~/.kaggle/kaggle.json')
        print('  Then re-run:  python download_datasets.py --wider')
        return None

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / 'wider_face.zip'

    print(f'Downloading WIDER FACE via Kaggle ({WIDER_KAGGLE_DATASET}) ...')
    try:
        subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', WIDER_KAGGLE_DATASET,
             '-p', str(dest), '--unzip'],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f'Kaggle download failed: {e}')
        _print_wider_manual()
        return None

    # Locate and normalise extracted structure
    _normalise_wider(dest, wider_dir)

    n = sum(1 for _ in wider_dir.rglob('*.jpg'))
    print(f'Done. {n:,} WIDER FACE images in {wider_dir}')
    return wider_dir


def _normalise_wider(dest: Path, wider_dir: Path):
    """Move images + annotation file into a consistent location."""
    wider_dir.mkdir(exist_ok=True)

    # Find annotation file wherever it landed
    for candidate in ['wider_face_train_bbx_gt.txt',
                       'wider_face_split/wider_face_train_bbx_gt.txt']:
        src = dest / candidate
        if src.exists():
            dst = wider_dir / 'wider_face_train_bbx_gt.txt'
            if not dst.exists():
                shutil.copy(src, dst)
            break

    # Find images folder (varies by Kaggle dataset version)
    for images_src in [dest / 'WIDER_train' / 'images',
                        dest / 'images',
                        dest / 'train' / 'images']:
        if images_src.exists():
            dst_images = wider_dir / 'images'
            if not dst_images.exists():
                shutil.copytree(images_src, dst_images)
            break


def _print_wider_manual():
    print('\nManual WIDER FACE download:')
    print('  1. Go to  http://shuoyang1213.me/WIDERFACE/')
    print('  2. Download  WIDER Face Training Images  (~1.5 GB)')
    print('  3. Download  Face annotations  (wider_face_split.zip)')
    print('  4. Extract so that:')
    print('       datasets/wider_face/images/<category>/<img>.jpg')
    print('       datasets/wider_face/wider_face_train_bbx_gt.txt')


# ── UTKFace ───────────────────────────────────────────────────────────────────

UTKFACE_KAGGLE_DATASET = 'jangedoo/utkface-new'

def download_utkface(dest_dir: str = 'datasets') -> Path:
    """
    Download UTKFace via Kaggle API.

    Expected result:
        datasets/utkface/*.jpg     (filenames encode age_gender_race_date.jpg)

    Demographic spread matters here — UTKFace has explicit race/age/gender
    labels across 23k images. Adding it to the training pool is the single
    biggest fix for "detector only fires on one demographic".
    """
    dest    = Path(dest_dir)
    utk_dir = dest / 'utkface'

    if utk_dir.exists() and any(utk_dir.glob('*.jpg')):
        n = sum(1 for _ in utk_dir.glob('*.jpg'))
        print(f'UTKFace already present ({n:,} images) — skipping.')
        return utk_dir

    if shutil.which('kaggle') is None:
        print('\nKaggle CLI not found. Install it:')
        print('  pip install kaggle')
        print('  Then re-run:  python download_datasets.py --utk')
        return None

    dest.mkdir(parents=True, exist_ok=True)
    tmp_dir = dest / 'tmp_utkface'
    print(f'Downloading UTKFace via Kaggle ({UTKFACE_KAGGLE_DATASET}) ...')
    try:
        subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', UTKFACE_KAGGLE_DATASET,
             '-p', str(tmp_dir), '--unzip'],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f'Kaggle download failed: {e}')
        return None

    # Kaggle mirror nests images under UTKFace/ or crop_part1/ depending on
    # the version — flatten to a single folder.
    utk_dir.mkdir(exist_ok=True)
    flattened = 0
    for p in tmp_dir.rglob('*.jpg'):
        dst = utk_dir / p.name
        if not dst.exists():
            shutil.move(str(p), str(dst))
            flattened += 1
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f'Done. {flattened:,} UTKFace images in {utk_dir}')
    return utk_dir


# ── FairFace ──────────────────────────────────────────────────────────────────

def print_fairface_instructions(dest_dir: str = 'datasets') -> None:
    """FairFace has no clean Kaggle mirror — print the manual recipe."""
    dest = Path(dest_dir)
    print('\nFairFace is hosted on the authors\' GitHub / Google Drive:')
    print('  1. Go to  https://github.com/joojs/fairface')
    print('  2. Download one of the "padding" zips linked from the README:')
    print('       fairface-img-margin025-trainval.zip  (~550 MB, recommended)')
    print('  3. Extract so that:')
    print(f'       {dest}/fairface/train/<image>.jpg')
    print(f'       {dest}/fairface/val/<image>.jpg')
    print('  Training will automatically pick up anything in  datasets/fairface/**/*.jpg.')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Download face datasets')
    parser.add_argument('--dest',     default='datasets')
    parser.add_argument('--wider',    action='store_true',
                        help='Also download WIDER FACE (needs Kaggle CLI)')
    parser.add_argument('--utk',      action='store_true',
                        help='Also download UTKFace (needs Kaggle CLI)')
    parser.add_argument('--fairface', action='store_true',
                        help='Print FairFace manual download instructions')
    parser.add_argument('--all',      action='store_true',
                        help='LFW + WIDER + UTKFace + FairFace instructions')
    args = parser.parse_args()

    print('=== Face dataset downloader ===\n')
    download_lfw(args.dest)

    if args.wider or args.all:
        print()
        download_wider_face(args.dest)

    if args.utk or args.all:
        print()
        download_utkface(args.dest)

    if args.fairface or args.all:
        print()
        print_fairface_instructions(args.dest)

    print('\nAll done.')
    print('Next step:  python train_detector.py')


if __name__ == '__main__':
    main()
