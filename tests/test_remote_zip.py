import io
import zipfile

import pytest

from backtesting.remote_zip import (
    RemoteZipError,
    SplitRangeFile,
    extract_member,
)


def make_parts(tmp_members, part_size):
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in tmp_members.items():
            archive.writestr(name, content)

    blob = buffer.getvalue()
    parts = [
        blob[index: index + part_size]
        for index in range(0, len(blob), part_size)
    ]
    urls = [f"mem://part{index}" for index in range(len(parts))]

    return blob, dict(zip(urls, parts))


def make_io(parts):
    calls = []

    def fetch(url, start, end):
        calls.append((url, start, end))
        return parts[url][start: end + 1]

    def size(url):
        return len(parts[url])

    return fetch, size, calls


CSV = "".join(f"{1700000000 + i * 300},1,2,0.5,1.5,3,4\n" for i in range(5000))


def test_extracts_member_across_part_boundaries(tmp_path):
    members = {
        "other/ETHUSD_5.csv": "1,2,3\n" * 3000,
        "XBTUSD_5.csv": CSV,
        "other/XBTEUR_5.csv": "9,9,9\n" * 3000,
    }
    blob, parts = make_parts(members, part_size=1500)
    fetch, size, calls = make_io(parts)

    result = extract_member(
        list(parts),
        tmp_path / "out.csv",
        fetch=fetch,
        size=size,
        chunk_bytes=1024,
    )

    assert result["member"] == "XBTUSD_5.csv"
    assert (tmp_path / "out.csv").read_text() == CSV
    assert result["archive_bytes"] == len(blob)
    assert len({call[0] for call in calls}) > 1


def test_downloads_only_part_of_a_large_archive(tmp_path):
    members = {
        "BIG/ETHUSD_5.csv": ("%d,1,2\n" * 1) * 1,
        "XBTUSD_5.csv": CSV,
    }
    filler = "".join(
        f"{i},{i * 7919 % 1000003},{i * 104729 % 999983}\n"
        for i in range(300000)
    )
    members["BIG/ETHUSD_5.csv"] = filler
    blob, parts = make_parts(members, part_size=400000)
    fetch, size, _ = make_io(parts)

    result = extract_member(
        list(parts),
        tmp_path / "out.csv",
        fetch=fetch,
        size=size,
        chunk_bytes=64 * 1024,
    )

    assert (tmp_path / "out.csv").read_text() == CSV
    assert result["bytes_downloaded"] < result["archive_bytes"]


def test_missing_member_lists_candidates(tmp_path):
    _, parts = make_parts({"ETHUSD_5.csv": "1,2\n" * 100}, 500)
    fetch, size, _ = make_io(parts)

    with pytest.raises(RemoteZipError, match="No archive member"):
        extract_member(
            list(parts), tmp_path / "x.csv", fetch=fetch, size=size
        )


def test_short_read_is_an_error():
    parts = {"mem://a": b"x" * 100}

    def fetch(url, start, end):
        return b"x"

    split = SplitRangeFile(
        list(parts), fetch=fetch, size=lambda url: 100
    )

    with pytest.raises(RemoteZipError):
        split.read(50)
