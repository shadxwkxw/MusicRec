"""Обложка из тегов аудиофайла: ID3 (mp3), FLAC, MP4 (m4a), Ogg.

Для mp3 из S3 весь файл не нужен: тег ID3v2 лежит в начале, его размер записан
в первых 10 байтах (id3_tag_size), и обложку можно прочитать из одного куска
(cover_from_id3) — около 300 КБ вместо ~9 МБ трека.
"""

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
FRONT_COVER = 3  # тип картинки «обложка, лицевая сторона» в ID3 и FLAC


@dataclass(frozen=True)
class Cover:
    data: bytes
    mime: str

    @property
    def extension(self) -> str:
        return EXTENSIONS.get(self.mime.lower(), ".jpg")


def id3_tag_size(header: bytes) -> int | None:
    """Полный размер тега ID3v2 (с заголовком и подвалом) по первым 10 байтам файла."""
    if len(header) < 10 or header[:3] != b"ID3":
        return None
    size = 0
    for byte in header[6:10]:  # syncsafe: по 7 бит на байт
        size = (size << 7) | (byte & 0x7F)
    footer = 10 if header[5] & 0x10 else 0
    return 10 + size + footer


def _pick(pictures: list) -> Cover | None:
    if not pictures:
        return None
    front = [p for p in pictures if getattr(p, "type", None) == FRONT_COVER]
    picture = (front or pictures)[0]
    data = bytes(picture.data)
    return Cover(data, picture.mime or "image/jpeg") if data else None


def cover_from_id3(tag: bytes) -> Cover | None:
    from mutagen.id3 import ID3, ID3NoHeaderError

    try:
        tags = ID3(BytesIO(tag))
    except (ID3NoHeaderError, Exception):
        return None
    return _pick(tags.getall("APIC"))


def read_cover(path: str | Path) -> Cover | None:
    """Обложка из файла целиком; None, если её нет или формат не читается."""
    import mutagen

    try:
        audio = mutagen.File(str(path))
    except Exception:
        return None
    if audio is None:
        return None
    if getattr(audio, "pictures", None):  # FLAC
        return _pick(audio.pictures)
    tags = audio.tags
    if tags is None:
        return None
    if hasattr(tags, "getall"):  # ID3 (mp3, wav, aiff)
        return _pick(tags.getall("APIC"))
    if "covr" in tags:  # MP4
        from mutagen.mp4 import MP4Cover

        art = tags["covr"][0]
        mime = (
            "image/png"
            if getattr(art, "imageformat", None) == MP4Cover.FORMAT_PNG
            else "image/jpeg"
        )
        return Cover(bytes(art), mime)
    blocks = tags.get("metadata_block_picture") if hasattr(tags, "get") else None
    if blocks:  # Ogg Vorbis / Opus
        from mutagen.flac import Picture

        return _pick([Picture(base64.b64decode(b)) for b in blocks])
    return None
