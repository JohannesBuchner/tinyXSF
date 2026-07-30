"""Get simple hash of a file."""

import hashlib


def hashfile(filename):
    """Compute a hash for the content of a file.

    Parameters
    ----------
    filename: str
        file name

    Returns
    -------
    hash: str
        hash digest of file content
    """
    with open(filename, 'rb', buffering=0) as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
