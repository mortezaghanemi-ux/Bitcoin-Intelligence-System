"""Read one file out of a huge split ZIP without downloading all of it.

Kraken publishes its full OHLCVT history as one ZIP cut into five parts
of about 2 GB each. A GitHub runner cannot comfortably hold 10 GB, but a
single pair's 5-minute CSV is only tens of megabytes inside it. The ZIP
format keeps a directory at the end of the archive, so only that
directory and the bytes of the wanted member need to be fetched, using
HTTP Range requests across the parts.
"""

from __future__ import annotations

import io
import re
import sys
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

DEFAULT_BASE = "https://assets.kraken.com/marketing/institutions"
DEFAULT_PARTS = tuple(
    f"{DEFAULT_BASE}/Kraken_OHLCVT_Full.zip.part{index:02d}"
    for index in range(5)
)
DEFAULT_MEMBER_PATTERN = r"(^|/)(XBTUSD|BTCUSD)_5\.csv$"
CHUNK_BYTES = 8 * 1024 * 1024

# fetch(url, start, end_inclusive) -> bytes
Fetch = Callable[[str, int, int], bytes]
# size(url) -> total size in bytes
Size = Callable[[str], int]


class RemoteZipError(RuntimeError):
    """Raised when the remote archive cannot be read."""


def http_fetch(url: str, start: int, end: int) -> bytes:
    import requests

    response = requests.get(
        url,
        headers={"Range": f"bytes={start}-{end}"},
        timeout=120,
    )

    if response.status_code not in (200, 206):
        raise RemoteZipError(
            f"HTTP {response.status_code} for {url} "
            f"bytes {start}-{end}"
        )

    data = response.content

    if response.status_code == 200:
        # Server ignored the Range header and sent everything.
        raise RemoteZipError(
            f"{url} does not support HTTP Range requests."
        )

    return data


def http_size(url: str) -> int:
    import requests

    response = requests.get(
        url,
        headers={"Range": "bytes=0-0"},
        timeout=120,
    )

    if response.status_code != 206:
        raise RemoteZipError(
            f"HTTP {response.status_code} for {url} "
            "(expected a ranged response)."
        )

    total = response.headers.get("Content-Range", "").rpartition("/")[2]

    if not total.isdigit():
        raise RemoteZipError(f"No size reported for {url}.")

    return int(total)


class SplitRangeFile(io.RawIOBase):
    """A read-only seekable file made of several remote parts."""

    def __init__(
        self,
        urls: Sequence[str],
        fetch: Fetch = http_fetch,
        size: Size = http_size,
    ) -> None:
        if not urls:
            raise RemoteZipError("No archive parts were given.")

        self._urls = list(urls)
        self._fetch = fetch
        self._sizes = [size(url) for url in self._urls]
        self._starts = []

        offset = 0

        for part_size in self._sizes:
            self._starts.append(offset)
            offset += part_size

        self._total = offset
        self._position = 0
        self.requests_made = 0
        self.bytes_fetched = 0

    @property
    def total_size(self) -> int:
        return self._total

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self._position + offset
        elif whence == io.SEEK_END:
            target = self._total + offset
        else:
            raise ValueError("Invalid whence.")

        if target < 0:
            raise ValueError("Negative seek position.")

        self._position = target

        return target

    def readinto(self, buffer) -> int:
        wanted = len(buffer)

        if wanted == 0 or self._position >= self._total:
            return 0

        end = min(self._position + wanted, self._total)
        chunks = []
        cursor = self._position

        while cursor < end:
            index = max(
                i
                for i, start in enumerate(self._starts)
                if start <= cursor
            )
            part_start = self._starts[index]
            part_end = part_start + self._sizes[index]
            stop = min(end, part_end)

            data = self._fetch(
                self._urls[index],
                cursor - part_start,
                stop - part_start - 1,
            )

            expected = stop - cursor

            if len(data) != expected:
                raise RemoteZipError(
                    f"Short read from {self._urls[index]}: "
                    f"got {len(data)} of {expected} bytes."
                )

            self.requests_made += 1
            self.bytes_fetched += len(data)
            chunks.append(data)
            cursor = stop

        joined = b"".join(chunks)
        buffer[: len(joined)] = joined
        self._position += len(joined)

        return len(joined)


def extract_member(
    urls: Sequence[str],
    output_path: str | Path,
    *,
    member_pattern: str = DEFAULT_MEMBER_PATTERN,
    fetch: Fetch = http_fetch,
    size: Size = http_size,
    chunk_bytes: int = CHUNK_BYTES,
) -> dict:
    """Write the first archive member matching the pattern to a file."""

    raw = SplitRangeFile(urls, fetch=fetch, size=size)
    reader = io.BufferedReader(raw, buffer_size=chunk_bytes)

    try:
        archive = zipfile.ZipFile(reader)
    except zipfile.BadZipFile as error:
        raise RemoteZipError(
            f"The parts do not form a valid ZIP: {error}"
        ) from error

    pattern = re.compile(member_pattern)
    names = archive.namelist()
    matches = [name for name in names if pattern.search(name)]

    if not matches:
        candidates = [
            name for name in names if re.search(r"(?i)(XBT|BTC)", name)
        ][:20]
        raise RemoteZipError(
            "No archive member matches "
            f"{member_pattern!r}. Candidates: {candidates}"
        )

    member = matches[0]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with archive.open(member) as source, output.open("wb") as target:
        while True:
            block = source.read(1024 * 1024)

            if not block:
                break

            target.write(block)
            written += len(block)

    return {
        "member": member,
        "bytes_written": written,
        "archive_bytes": raw.total_size,
        "bytes_downloaded": raw.bytes_fetched,
        "requests": raw.requests_made,
    }


if __name__ == "__main__":
    destination = sys.argv[1] if len(sys.argv) > 1 else "XBTUSD_5.csv"
    print(extract_member(DEFAULT_PARTS, destination))
