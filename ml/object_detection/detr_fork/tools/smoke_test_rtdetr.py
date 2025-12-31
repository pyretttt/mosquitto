import torch

from model.rtdetr_paper_aligned import build_rt_detr


def main():
    # Build model (mobilenet_v3_large by default)
    model = build_rt_detr(num_classes=80, bg_class_idx=0, backbone="mobilenet_v3_large")
    model.eval()

    # Random input (B=2, 3x224x224)
    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        out = model(x)

    print("Output keys:", list(out.keys()))
    detections = out.get("detections", [])
    print("Num batches:", len(detections))
    for i, d in enumerate(detections):
        print(
            f"Batch {i}: boxes={tuple(d['boxes'].shape)}, scores={tuple(d['scores'].shape)}, labels={tuple(d['labels'].shape)}"
        )


if __name__ == "__main__":
    main()
