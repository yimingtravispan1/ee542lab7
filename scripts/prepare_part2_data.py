"""Download the lab's Parquet dataset and make a reproducible held-out split."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.packages'))
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

REVISION = 'b5ae3589204019bc2cc97e99e4914a54589333ef'
SOURCE = 'https://huggingface.co/datasets/microsoft/cats_vs_dogs'
SHARDS = [
    ('train-00000-of-00002.parquet', 330245704,
     'd2a40bc460fc9ac12a4a0edbe44476cbed5dd0829a8cbc08af9bcbbea5cc9620'),
    ('train-00001-of-00002.parquet', 391396716,
     '568276cea0e7ee4173847ab3db33d3b64c3ec8fd061e2f3b336d0f48d791421e'),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(raw):
    raw.mkdir(parents=True, exist_ok=True)
    for name, size, checksum in SHARDS:
        path = raw / name
        if path.exists() and path.stat().st_size == size and sha256(path) == checksum:
            print('Verified existing shard:', name, flush=True)
            continue
        temporary = path.with_suffix('.parquet.partial')
        url = f'{SOURCE}/resolve/{REVISION}/data/{name}?download=true'
        for attempt in range(3):
            try:
                offset = temporary.stat().st_size if temporary.exists() else 0
                request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
                with urllib.request.urlopen(request, timeout=90) as response:
                    resume = offset > 0 and response.status == 206
                    done = offset if resume else 0
                    with temporary.open('ab' if resume else 'wb') as handle:
                        last = time.perf_counter()
                        while chunk := response.read(4 * 1024 * 1024):
                            handle.write(chunk)
                            done += len(chunk)
                            if time.perf_counter() - last > 15:
                                print(f'{name}: {done / size:.1%}', flush=True)
                                last = time.perf_counter()
                if temporary.stat().st_size != size or sha256(temporary) != checksum:
                    temporary.unlink()
                    raise ValueError(f'Integrity check failed: {name}')
                temporary.replace(path)
                print('Downloaded and SHA-256 verified:', name, flush=True)
                break
            except Exception:
                if attempt == 2:
                    raise
                print(f'Retrying download: {name}', flush=True)


def prepare(data, seed):
    start = time.perf_counter()
    download(data / 'raw')
    records, skipped, duplicates = [], [], []
    seen = set()
    staging = data / 'images'
    staging.mkdir(parents=True, exist_ok=True)
    for shard, _, _ in SHARDS:
        rows = 0
        parquet = pq.ParquetFile(data / 'raw' / shard)
        for batch in parquet.iter_batches(batch_size=128, columns=['image', 'labels']):
            for item in batch.to_pylist():
                row = rows
                rows += 1
                label = int(item['labels'])
                if label not in (0, 1):
                    raise ValueError(f'Unexpected label: {label}')
                payload = item['image']['bytes']
                digest = hashlib.sha256(payload).hexdigest()
                source_id = f'{shard}:{row}'
                if digest in seen:
                    duplicates.append(source_id)
                    continue
                try:
                    with Image.open(io.BytesIO(payload)) as image:
                        image.load()
                        image.convert('RGB')
                except (OSError, ValueError) as error:
                    skipped.append({'source': source_id, 'error': str(error)})
                    continue
                seen.add(digest)
                path = staging / f'{digest}.jpg'
                path.write_bytes(payload)
                records.append({'source': source_id, 'label': label, 'sha256': digest})
        print(f'Decoded {rows} rows from {shard}', flush=True)
    rng = np.random.default_rng(seed)
    counts = {split: {'cat': 0, 'dog': 0} for split in ('train', 'test')}
    for label, name in enumerate(('cat', 'dog')):
        group = [r for r in records if r['label'] == label]
        order = rng.permutation(len(group))
        split_at = int(0.8 * len(group))
        for split in ('train', 'test'):
            (data / split / name).mkdir(parents=True, exist_ok=True)
        for position, index in enumerate(order):
            record = group[index]
            split = 'train' if position < split_at else 'test'
            relative = Path(split) / name / f"{record['sha256']}.jpg"
            (staging / relative.name).replace(data / relative)
            record['path'] = relative.as_posix()
            counts[split][name] += 1
    staging.rmdir()
    summary = {
        'dataset': 'microsoft/cats_vs_dogs', 'source_url': SOURCE,
        'revision': REVISION, 'label_mapping': {'0': 'cat', '1': 'dog'},
        'split_seed': seed, 'train_fraction_per_class': 0.8,
        'split_method': 'seeded stratified split after exact-byte deduplication',
        'counts': counts, 'usable_unique_images': len(records),
        'duplicate_rows_removed': len(duplicates), 'duplicate_sources': duplicates,
        'skipped_images': skipped, 'preparation_seconds': time.perf_counter() - start,
        'shards': [{'file': n, 'bytes': s, 'sha256': h} for n, s, h in SHARDS],
    }
    manifest = {**summary, 'images': records}
    (data / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    output = ROOT / 'results' / 'part2'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'data_preparation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    data = ROOT / 'data'
    if (data / 'manifest.json').exists():
        existing = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
        if existing['split_seed'] != args.seed or existing['revision'] != REVISION:
            raise SystemExit('An existing dataset has a different seed/revision; '
                             'use a separate project copy for a different split.')
        if not all((data / row['path']).is_file() for row in existing['images']):
            raise SystemExit('Prepared dataset is incomplete; restore missing images before running.')
        print('Reusing prepared dataset:', existing['counts'])
        return
    prepare(data, args.seed)


if __name__ == '__main__':
    main()
