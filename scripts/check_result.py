from pathlib import Path
import argparse
import requests
import tarfile
import cv2
import numpy as np

OUT_DIR = Path(__file__).cwd() / 'out'

def main(args):
    response = requests.get(args.url, stream=True)
    assert response.status_code == 200, 'Error status code'
    name = args.url.split('/')[-1]
    uuid = name.split('.')[0][6:]

    tar_path = OUT_DIR / name
    if response.status_code == 200:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(tar_path, 'wb') as f:
            f.write(response.raw.read())
    
    with tarfile.open(tar_path) as f:
        f.extractall(OUT_DIR)
    tar_path.unlink()
    
    out_path = OUT_DIR / 'build'
    out_path = out_path.rename(OUT_DIR / f'build-{uuid}')
    print(f"Downloaded build {uuid} to {out_path.absolute()}")

    showed_images = None
    for i, img_path in enumerate(out_path.glob('*.png')):
        img = cv2.imread(str(img_path))
        if showed_images is None:
            showed_images = img
        else:
            showed_images = np.hstack((showed_images, img))
    showed_images = showed_images.astype(np.uint8)
    if args.show:
        cv2.imshow(f'build-{uuid}', showed_images)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, required=True)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    main(args)