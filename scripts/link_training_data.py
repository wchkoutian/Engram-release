"""Link an existing training cache into a new, separate stage output root."""
import argparse
from pathlib import Path
import shutil

FILES = ('manifest.json', 'DATA_READY.json', 'caption_text.pt', 'captions.json')

def link(source, target):
    source, target = source.resolve(), target.resolve()
    for name in (*FILES, 'data'):
        if not (source / name).exists():
            raise FileNotFoundError(source / name)
    if target.exists():
        raise FileExistsError('Use a new stage root; refusing to replace files')
    target.mkdir(parents=True)
    for name in FILES:
        shutil.copy2(source / name, target / name)
    (target / 'data').symlink_to(source / 'data', target_is_directory=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('target', type=Path)
    a = p.parse_args()
    link(a.source, a.target)
