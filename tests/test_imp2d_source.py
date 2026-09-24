"""Automatic source fetching must never publish incomplete or wrong databases."""
from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from HERA.data import imp2d_source as source


class SourceDownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'dataset/imp2d/imp2d.db'
        self.payload = b'SQLite format 3\x00' + b'verified test fixture' * 10
        constants = patch.multiple(source, IMP2D_DATABASE_BYTES=len(self.payload),
                                   IMP2D_DATABASE_SHA256=hashlib.sha256(self.payload).hexdigest())
        constants.start()
        self.addCleanup(constants.stop)

    def fetch(self):
        with redirect_stdout(io.StringIO()):
            return source.download_imp2d_database(self.path)

    def test_download_and_offline_reuse(self):
        with patch.object(source, 'urlopen', return_value=io.BytesIO(self.payload)) as request:
            self.assertEqual(self.fetch(), self.path)
            self.assertEqual(self.path.read_bytes(), self.payload)
            request.assert_called_once()
        with patch.object(source, 'urlopen', side_effect=AssertionError('network forbidden')):
            self.assertEqual(self.fetch(), self.path)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_wrong_release_never_becomes_a_database(self):
        altered = b'x' + self.payload[1:]
        with patch.object(source, 'urlopen', return_value=io.BytesIO(altered)):
            with self.assertRaisesRegex(RuntimeError, 'SHA256'):
                self.fetch()
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_truncated_download_does_not_persist(self):
        with patch.object(source, 'urlopen', return_value=io.BytesIO(self.payload[:32])):
            with self.assertRaisesRegex(RuntimeError, 'size/SHA256'):
                self.fetch()
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_network_failure_is_explicit_and_cleans_temporary_files(self):
        with patch.object(source, 'urlopen', side_effect=OSError('network unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'Physical screening was not skipped'):
                self.fetch()
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_existing_destination_is_not_overwritten(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(b'existing user database')
        with patch.object(source, 'urlopen') as request:
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                self.fetch()
            request.assert_not_called()
        self.assertEqual(self.path.read_bytes(), b'existing user database')

    def test_concurrent_download_lock_is_preserved(self):
        self.path.parent.mkdir(parents=True)
        lock = self.path.with_name(self.path.name + '.download.lock')
        lock.write_text('another process', encoding='utf-8')
        with patch.object(source, 'urlopen') as request:
            with self.assertRaisesRegex(RuntimeError, 'download lock exists'):
                self.fetch()
            request.assert_not_called()
        self.assertEqual(lock.read_text(), 'another process')


if __name__ == '__main__':
    unittest.main()
