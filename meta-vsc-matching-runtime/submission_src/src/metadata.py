from dataclasses import dataclass, fields
from typing import Optional
import ffmpeg


@dataclass
class Stream:
    profile: str
    width: int
    height: int
    coded_width: int
    coded_height: int
    pix_fmt: str
    level: int
    chroma_location: Optional[str]
    r_frame_rate: str
    avg_frame_rate: str
    time_base: str
    duration_ts: int
    duration: float
    bit_rate: int
    nb_frames: int
    sample_aspect_ratio: Optional[str]  # None: (probably) not augmented
    display_aspect_ratio: Optional[str]


@dataclass
class Format:
    filename: str
    nb_streams: int  # 1: video only, 2: video + audio
    duration: float
    size: int
    bit_rate: int


@dataclass
class FFProbeMetadata:
    stream: Stream
    format: Format


def get_video_metadata(path: str):
    probe = ffmpeg.probe(path)
    for stream in probe["streams"]:
        if stream["codec_type"] == "video":
            break
    format = probe["format"]

    return FFProbeMetadata(
        stream=Stream(
            profile=stream["profile"],
            width=int(stream["width"]),
            height=int(stream["height"]),
            coded_width=int(stream["coded_width"]),
            coded_height=int(stream["coded_height"]),
            pix_fmt=str(stream["pix_fmt"]),
            level=int(stream["level"]),
            chroma_location=stream.get("chroma_location"),
            r_frame_rate=str(stream["r_frame_rate"]),
            avg_frame_rate=str(stream["avg_frame_rate"]),
            time_base=str(stream["time_base"]),
            duration_ts=int(stream["duration_ts"]),
            duration=float(stream["duration"]),
            bit_rate=int(stream["bit_rate"]),
            nb_frames=int(stream["nb_frames"]),
            sample_aspect_ratio=stream.get("sample_aspect_ratio"),
            display_aspect_ratio=stream.get("display_aspect_ratio"),
        ),
        format=Format(
            filename=str(format["filename"]),
            nb_streams=int(format["nb_streams"]),
            duration=float(format["duration"]),
            size=int(format["size"]),
            bit_rate=int(format["bit_rate"]),
        )
    )
