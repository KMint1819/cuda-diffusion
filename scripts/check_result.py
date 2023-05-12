from pathlib import Path
import argparse
import tarfile
import requests

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
    out_path.rename(OUT_DIR / f'build-{uuid}')
    print(f"Downloaded build {uuid} to {out_path.absolute()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, required=True)
    args = parser.parse_args()
    main(args)