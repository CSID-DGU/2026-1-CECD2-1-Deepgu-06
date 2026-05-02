import uuid
from datetime import datetime

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.core.config import settings

_SIGV4_CONFIG = Config(signature_version="s3v4")


def _get_client():
    kwargs = {"region_name": settings.aws_region, "config": _SIGV4_CONFIG}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    return boto3.client("s3", **kwargs)


def upload_event_video(video_bytes: bytes, camera_id: str, detected_at: datetime) -> str:
    """MP4 bytes를 S3에 업로드하고 object key를 반환한다."""
    key = f"events/{camera_id}/{detected_at.strftime('%Y/%m')}/{uuid.uuid4().hex}.mp4"
    client = _get_client()
    client.put_object(
        Bucket=settings.s3_bucket_name,
        Key=key,
        Body=video_bytes,
        ContentType="video/mp4",
    )
    return key


def generate_presigned_url(s3_key: str) -> str | None:
    if not s3_key or not settings.s3_bucket_name:
        return None
    try:
        client = _get_client()
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket_name, "Key": s3_key},
            ExpiresIn=settings.s3_presigned_expires,
        )
    except ClientError:
        return None
