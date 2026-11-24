"""Validate user paths before allocating models or writing output."""
import json
from pathlib import Path
import re

def parse_shard(value):
    parts = value.split('/')
    if len(parts) != 2:
        raise ValueError('Shard must be k/n, with 0 <= k < n')
    k, n = map(int, parts)
    if not 0 <= k < n:
        raise ValueError('Shard must be k/n, with 0 <= k < n')
    return k, n

def validate_worldscore_inputs(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'))
    rows = manifest.get('cases')
    if not isinstance(rows, list) or not rows:
        raise ValueError('WorldScore manifest must contain a nonempty cases list')
    names, result = set(), []
    for row in rows:
        name = row['name']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in names:
            raise ValueError('Case names must be unique safe directory names')
        names.add(name)
        prompt = row['positive_prompt']
        second = row.get('positive_prompt_2', prompt)
        if any(not isinstance(x, str) or not x.strip() for x in (prompt, second)):
            raise ValueError('WorldScore prompts must be nonempty strings')
        folder = (path.parent / row['input_directory']).resolve()
        for file in ('input_image.png', 'camera_data.json', 'image_data.json'):
            if not (folder / file).is_file():
                raise FileNotFoundError(folder / file)
        result.append({**row, 'directory': folder, 'positive_prompt_2': second})
    return result
