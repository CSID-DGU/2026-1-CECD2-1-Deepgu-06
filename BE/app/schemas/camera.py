from typing import Annotated, Optional
from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator


CameraIdType = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=50,
        pattern=r"^[a-z0-9_-]+$",
    ),
]

NameType = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=100,
    ),
]

LocationType = Annotated[
    Optional[str],
    StringConstraints(
        strip_whitespace=True,
        max_length=100,
    ),
]

StreamKeyType = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=100,
        pattern=r"^[a-z0-9_-]+$",
    ),
]

DescriptionType = Annotated[
    Optional[str],
    StringConstraints(
        strip_whitespace=True,
        max_length=255,
    ),
]

PatchNameType = Annotated[
    Optional[str],
    StringConstraints(
        strip_whitespace=True,
        max_length=100,
    ),
]

PatchLocationType = Annotated[
    Optional[str],
    StringConstraints(
        strip_whitespace=True,
        max_length=100,
    ),
]

PatchStreamKeyType = Annotated[
    Optional[str],
    StringConstraints(
        strip_whitespace=True,
        max_length=100,
        pattern=r"^[a-z0-9_-]+$",
    ),
]

PatchDescriptionType = Annotated[
    Optional[str],
    StringConstraints(
        strip_whitespace=True,
        max_length=255,
    ),
]


class CameraCreateRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    cameraId: CameraIdType
    name: NameType
    location: LocationType = None
    streamKey: StreamKeyType
    description: DescriptionType = None

    @field_validator("cameraId", "streamKey", mode="before")
    @classmethod
    def normalize_id_fields(cls, v):
        if v is None:
            return v
        if not isinstance(v, str):
            raise TypeError("must be a string")
        return v.strip().lower()

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, v):
        if v is None:
            return v
        if not isinstance(v, str):
            raise TypeError("must be a string")
        return v.strip()

    @field_validator("location", "description", mode="before")
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        if not isinstance(v, str):
            raise TypeError("must be a string")
        v = v.strip()
        return v if v != "" else None

    @field_validator("name")
    @classmethod
    def validate_name_not_blank(cls, v):
        if v is None or v == "":
            raise ValueError("name must not be blank")
        return v


class CameraUpdateRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    name: PatchNameType = None
    location: PatchLocationType = None
    streamKey: PatchStreamKeyType = None
    description: PatchDescriptionType = None

    @field_validator("streamKey", mode="before")
    @classmethod
    def normalize_stream_key(cls, v):
        if v is None:
            return None
        if not isinstance(v, str):
            raise TypeError("must be a string")
        v = v.strip().lower()
        return v if v != "" else None

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, v):
        if v is None:
            return None
        if not isinstance(v, str):
            raise TypeError("must be a string")
        return v.strip()

    @field_validator("location", "description", mode="before")
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        if not isinstance(v, str):
            raise TypeError("must be a string")
        v = v.strip()
        return v if v != "" else None

    @field_validator("name")
    @classmethod
    def validate_name_not_blank(cls, v):
        if v == "":
            raise ValueError("name must not be blank")
        return v