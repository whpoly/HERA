"""Fetch the pinned public IMP2D source for reproducible physical screening."""
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


# Official download linked by https://cmr.fysik.dtu.dk/imp2d/imp2d.html.
# Pin the 2022-07-12 release used by the checked CIF/label audit.
IMP2D_DATABASE_URL = 'https://wiki.fysik.dtu.dk/cmr-files/imp2d.db'
IMP2D_DATABASE_BYTES = 71819264
IMP2D_DATABASE_SHA256 = '3a71db999b477112da248dcf762c4384e455689953679d58b3d71a91e7148fc4'


def _verify_download(path):
    checksum = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            size += len(chunk)
            checksum.update(chunk)
    if size != IMP2D_DATABASE_BYTES or checksum.hexdigest() != IMP2D_DATABASE_SHA256:
        raise ValueError('IMP2D download size/SHA256 differs from the verified 2022-07-12 release')


def download_imp2d_database(destination):
    """Download to a temporary file; publish only after size and hash verification.

The lock avoids concurrent writers. Interrupted downloads never become a usable
database, and existing destination files are never replaced by this operation.
"""
    destination = Path(destination)
    if destination.exists():
        _verify_download(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(destination.name + '.download.lock')
    try:
        lock_handle = lock.open('x', encoding='utf-8')
    except FileExistsError as exc:
        raise RuntimeError(f'IMP2D download lock exists: {lock}. Wait for the other download; '
                           'if it was interrupted, remove the stale lock before retrying.') from exc
    temporary = None
    try:
        with lock_handle:
            lock_handle.write(f'pid={os.getpid()}\n')
        if destination.exists():
            _verify_download(destination)
            return destination
        print(f'IMP2D source database missing: downloading {IMP2D_DATABASE_URL}\n'
              f'  Destination: {destination} ({IMP2D_DATABASE_BYTES / 1e6:.1f} MB)', flush=True)
        request = Request(IMP2D_DATABASE_URL, headers={'User-Agent': 'HERA-IMP2D-physical-filter'})
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.imp2d-',
                                         suffix='.part', delete=False) as stream:
            temporary = Path(stream.name)
            with urlopen(request, timeout=60) as response:
                received, next_progress = 0, 8 * 1024 * 1024
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > IMP2D_DATABASE_BYTES:
                        raise ValueError('IMP2D response exceeds the verified release size')
                    stream.write(chunk)
                    if received >= next_progress:
                        print(f'  IMP2D download: {received / 1e6:.1f}/{IMP2D_DATABASE_BYTES / 1e6:.1f} MB',
                              flush=True)
                        next_progress += 8 * 1024 * 1024
            stream.flush()
            os.fsync(stream.fileno())
        _verify_download(temporary)
        if destination.exists():
            _verify_download(destination)
        else:
            # Cooperating processes cannot publish while this lock is held.
            os.replace(temporary, destination)
            temporary = None
        print(f'IMP2D database downloaded and SHA256 verified: {destination}', flush=True)
        return destination
    except (OSError, ValueError) as exc:
        raise RuntimeError(f'Could not obtain the verified IMP2D database from {IMP2D_DATABASE_URL}: {exc}. '
                           f'Retry with network access, or copy the original database to {destination}. '
                           'Physical screening was not skipped.') from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
