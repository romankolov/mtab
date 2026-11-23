#!/usr/bin/env python3
"""Verify release archives contain exactly the expected files; no extraction.

Build/CI only. This whitelist excludes model weights and auxiliary blobs.
The workflow separately runs the binary and computes archive SHA-256 sums.
"""
import argparse
from pathlib import Path
import tarfile
import zipfile
import stat


def verify(archive):
    name = archive.name
    if name.endswith('.tar.gz'):
        stem, exe = name[:-7], 'mtab'
        with tarfile.open(archive, 'r:gz') as source:
            members = source.getmembers()
            if any(not (entry.isfile() or entry.isdir()) for entry in members):
                raise ValueError(f'{archive}: symlinks and special entries are forbidden')
            files = [entry.name for entry in members if entry.isfile()]
            dirs = [entry.name.rstrip('/') for entry in members if entry.isdir()]
            sizes = {entry.name: entry.size for entry in members if entry.isfile()}
    elif name.endswith('.zip'):
        stem, exe = name[:-4], 'mtab.exe'
        with zipfile.ZipFile(archive) as source:
            entries = source.infolist()
            if any(stat.S_ISLNK(entry.external_attr >> 16) for entry in entries):
                raise ValueError(f'{archive}: symlinks are forbidden')
            files = [entry.filename for entry in entries if not entry.is_dir()]
            dirs = [entry.filename.rstrip('/') for entry in entries if entry.is_dir()]
            sizes = {entry.filename: entry.file_size for entry in entries if not entry.is_dir()}
    else:
        raise ValueError(f'{archive}: unsupported archive type')
    docs = ['architecture.md', 'tuning-detection.md', 'techniques.md', 'weights.md']
    expected = {f'{stem}/{file}' for file in [exe, 'README.md', 'LICENSE', 'NOTICE', 'PROMPT.md']}
    expected |= {f'{stem}/docs/{doc}' for doc in docs}
    if len(files) != len(expected) or set(files) != expected:
        raise ValueError(f'{archive}: unexpected archive layout or duplicate files: {files}')
    # Directory entries may be implicit; if present, only the archive root and
    # its docs directory are allowed.
    if not set(dirs) <= {stem, f'{stem}/docs'}:
        raise ValueError(f'{archive}: unexpected directories: {dirs}')
    if any(sizes[path] == 0 for path in expected):
        raise ValueError(f'{archive}: empty required file')
    print(f'{archive}: verified {len(files)} expected files and no model assets')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', type=Path, nargs='+')
    args = parser.parse_args()
    for archive in args.archives:
        verify(archive)


if __name__ == '__main__':
    main()
