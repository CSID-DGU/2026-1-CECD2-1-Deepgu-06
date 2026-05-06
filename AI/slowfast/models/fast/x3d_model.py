from pytorchvideo.models.x3d import create_x3d


def build_fast_model(
    architecture="x3d_s",
    num_classes=1,
    pretrained=False,
    input_clip_length=13,
    input_crop_size=160,
):
    if architecture != "x3d_s":
        raise ValueError(f"unsupported architecture '{architecture}'; expected 'x3d_s'")
    if pretrained:
        raise ValueError("pretrained backbone loading is not wired for the direct create_x3d path yet")

    return create_x3d(
        input_clip_length=int(input_clip_length),
        input_crop_size=int(input_crop_size),
        model_num_class=int(num_classes),
        dropout_rate=0.5,
        width_factor=2.0,
        depth_factor=2.2,
        bottleneck_factor=2.25,
        head_dim_out=2048,
        head_activation=None,
    )
