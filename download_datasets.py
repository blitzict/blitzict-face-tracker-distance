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
   occlusions and distances.  This is the key dataset for teaching the
   detector to find faces far away (small) as well as close up (large).
   Size  : ~1.5 GB (train split only)
   Requires the Kaggle CLI:  pip install kaggle
   Then place your kaggle.json in  ~/.kaggle/kaggle.json

USAGE
─────
    python download_datasets.py            # download LFW only
    python download_datasets.py --wider    # also download WIDER FACE
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Download face datasets')
    parser.add_argument('--dest',   default='datasets')
    parser.add_argument('--wider',  action='store_true',
                        help='Also download WIDER FACE (needs Kaggle CLI)')
    args = parser.parse_args()

    print('=== Face dataset downloader ===\n')
    download_lfw(args.dest)

    if args.wider:
        print()
        download_wider_face(args.dest)

    print('\nAll done.')
    print('Next step:  python train_face_detector.py')


if __name__ == '__main__':
    main()
