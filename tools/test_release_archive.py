#!/usr/bin/env python3
"""Offline adversarial packaging tests for release archive whitelist."""
import io
import stat
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

from verify_release_archive import verify


DOCS = ['architecture.md', 'tuning-detection.md', 'techniques.md', 'weights.md']


class ReleaseArchiveTest(unittest.TestCase):
    def test_tar_and_zip_whitelist_rejects_extra_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tar = root/'mtab-test-x86_64-unknown-linux-gnu.tar.gz'
            stem = tar.name[:-7]
            names = [f'{stem}/{x}' for x in ['mtab', 'README.md', 'LICENSE', 'NOTICE', 'PROMPT.md']]
            names += [f'{stem}/docs/{doc}' for doc in DOCS]
            def make_tar(extra=None):
                with tarfile.open(tar, 'w:gz') as out:
                    for name in names + ([extra] if extra else []):
                        entry = tarfile.TarInfo(name)
                        entry.size = 1
                        out.addfile(entry, io.BytesIO(b'x'))
            make_tar()
            verify(tar)
            make_tar(f'{stem}/model.safetensors')
            with self.assertRaises(ValueError):
                verify(tar)
            make_tar(names[0])
            with self.assertRaises(ValueError):
                verify(tar)
            # A missing document is a failure, not a silently thinner archive.
            make_tar()
            with tarfile.open(tar, 'w:gz') as out:
                for name in names[:-1]:
                    entry = tarfile.TarInfo(name)
                    entry.size = 1
                    out.addfile(entry, io.BytesIO(b'x'))
            with self.assertRaises(ValueError):
                verify(tar)
            with tarfile.open(tar, 'w:gz') as out:
                for name in names:
                    entry = tarfile.TarInfo(name)
                    entry.type = tarfile.SYMTYPE
                    entry.linkname = '/tmp/elsewhere'
                    out.addfile(entry)
            with self.assertRaises(ValueError):
                verify(tar)
            zip_path = root/'mtab-test-x86_64-pc-windows-msvc.zip'
            stem = zip_path.stem
            with zipfile.ZipFile(zip_path, 'w') as out:
                for name in ['mtab.exe', 'README.md', 'LICENSE', 'NOTICE', 'PROMPT.md']:
                    out.writestr(f'{stem}/{name}', b'x')
                for doc in DOCS:
                    out.writestr(f'{stem}/docs/{doc}', b'x')
            verify(zip_path)
            symlink = root/'mtab-link.zip'
            with zipfile.ZipFile(symlink, 'w') as out:
                for doc in DOCS:
                    out.writestr(f'mtab-link/docs/{doc}', b'x')
                for name in ['mtab.exe', 'README.md', 'LICENSE', 'NOTICE', 'PROMPT.md']:
                    info = zipfile.ZipInfo('mtab-link/' + name)
                    if name == 'mtab.exe':
                        info.create_system = 3
                        info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    out.writestr(info, b'x')
            with self.assertRaises(ValueError):
                verify(symlink)
            with zipfile.ZipFile(zip_path, 'a') as out:
                out.writestr(f'{stem}/model.onnx', b'x')
            with self.assertRaises(ValueError):
                verify(zip_path)

    def test_windows_backslash_entries_are_accepted(self):
        """Compress-Archive writes `stem\docs\weights.md`, not `stem/docs/...`.

        That is what the real Windows leg produces, and comparing it verbatim
        made every nested file look unknown — the v0.1.0 dry run failed on it.
        Normalising must accept the layout without weakening the check.
        """
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            zip_path = root/'mtab-test-x86_64-pc-windows-msvc.zip'
            stem = zip_path.stem

            def write(extra=None):
                with zipfile.ZipFile(zip_path, 'w') as out:
                    for name in ['mtab.exe', 'README.md', 'LICENSE', 'NOTICE', 'PROMPT.md']:
                        out.writestr(f'{stem}\\{name}', b'x')
                    for doc in DOCS:
                        out.writestr(f'{stem}\\docs\\{doc}', b'x')
                    if extra:
                        out.writestr(f'{stem}\\{extra}', b'x')

            write()
            verify(zip_path)
            # A backslash archive that smuggles weights in must still be caught.
            write('model.safetensors')
            with self.assertRaises(ValueError):
                verify(zip_path)


if __name__ == '__main__':
    unittest.main()
