#!/usr/bin/env python3
"""Cut the main object out of a photo and write the RGBA image generate.py needs.

Two backends, both from Meta and both usable commercially. By default a
noun phrase goes to SAM 3 and anything else to SAM 2:
  sam3  Segment Anything 3 (SAM License, gated weights: fetch with
        `fetch_weights.py --with sam3`). Finds every instance of each short
        noun phrase, whatever the background. It segments only what a phrase
        names, so give one phrase per part to keep: "farmer" "basket". A
        generic word like "object" finds nothing.
  sam2  Segment Anything 2.1 (Apache 2.0). Takes a box or a point. Without
        one it tries a grid of points and keeps the masks that stay clear of
        the frame, which finds a free-standing object with everything it
        carries, but fails when the object and the background are alike.

On the 12 example images composited over four backgrounds, SAM 3 with the
object's name scored a mean IoU of 0.85 against the true alpha and SAM 2
without hints 0.84, with SAM 2 falling to 0.2-0.5 on mid-grey backgrounds.

Examples, inside the GPU container:
  python scripts/cutout.py /input/photo.jpg /out/photo.png --prompt "coffee mug" saucer
  python scripts/cutout.py /input/photo.jpg /out/photo.png
  python scripts/cutout.py /input/photo.jpg /out/photo.png --box 120 80 900 1000
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

MODELS = {"sam3": "facebook/sam3", "sam2": "facebook/sam2.1-hiera-large"}


def device_for(name):
    import torch
    if name == "auto":
        return "xpu" if torch.xpu.is_available() else "cpu"
    return name


def group_norm_in_plain_ops(x, num_groups, weight=None, bias=None, eps=1e-5):
    """GroupNorm from reshape, mean and variance: the fused XPU kernel
    segfaults on SAM 3's mask decoder (torch 2.15.0+xpu)."""
    import torch
    n, c = x.shape[:2]
    g = x.reshape(n, num_groups, -1)
    g = (g - g.mean(-1, keepdim=True)) / torch.sqrt(g.var(-1, unbiased=False, keepdim=True) + eps)
    x = g.reshape(x.shape)
    shape = (1, c) + (1,) * (x.dim() - 2)
    if weight is not None:
        x = x * weight.reshape(shape)
    if bias is not None:
        x = x + bias.reshape(shape)
    return x


def run_sam3(image, args, device):
    import torch
    from transformers import Sam3Model, Sam3Processor

    if device == "xpu":
        torch.nn.functional.group_norm = group_norm_in_plain_ops
    model = Sam3Model.from_pretrained(args.model or MODELS["sam3"]).to(device).eval()
    processor = Sam3Processor.from_pretrained(args.model or MODELS["sam3"])
    masks, scores = [], []
    # One pass per phrase; the model returns every instance of a phrase, and
    # the phrases together cover an object and what it carries.
    for text in args.prompt or [None]:
        kwargs = {"text": text} if text else {}
        if args.box or not text:
            kwargs["input_boxes"] = [[args.box or default_box(image)]]
            kwargs["input_boxes_labels"] = [[1]]
        inputs = processor(images=image, return_tensors="pt", **kwargs).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_instance_segmentation(
            outputs, threshold=args.threshold, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist())[0]
        masks += [m.cpu().numpy().astype(bool) for m in result["masks"]]
        scores += [float(x) for x in result["scores"]]
    return masks, scores


def run_sam2(image, args, device):
    import torch
    import transformers
    from transformers import Sam2Model, Sam2Processor

    transformers.logging.set_verbosity_error()   # the checkpoint's config type is sam2_video; loading it as sam2 is fine

    model = Sam2Model.from_pretrained(args.model or MODELS["sam2"]).to(device).eval()
    processor = Sam2Processor.from_pretrained(args.model or MODELS["sam2"])
    kwargs = {}
    w, h = image.size
    box = args.box
    if box and (box[2] - box[0]) * (box[3] - box[1]) > 0.85 * w * h:
        # A box that is nearly the whole image makes the model pick the
        # background. The grid below finds a frame-filling object fine.
        box = None
    if args.point:
        kwargs["input_points"] = [[[args.point]]]
        kwargs["input_labels"] = [[[1]]]
    elif box:
        kwargs["input_boxes"] = [[box]]
    else:
        # No hint: ask about a grid of points, one object per point, and keep
        # the masks that stay clear of the frame. Background masks touch it.
        points = grid_points(image, inside=args.box)
        kwargs["input_points"] = [[[pt] for pt in points]]
        kwargs["input_labels"] = [[[1]] * len(points)]
    inputs = processor(images=image, return_tensors="pt", **kwargs).to(device)
    with torch.no_grad():
        # A box is unambiguous, so ask for one mask. A point gets three, from
        # a part to the whole object.
        outputs = model(**inputs, multimask_output=not box)
    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0].numpy().astype(bool)
    scores = outputs.iou_scores.cpu().numpy()[0]           # [objects, masks]
    if args.point or box:
        return list(masks[0]), [float(x) for x in scores[0]]
    # Per point, the largest of its three masks is the whole object, or the
    # background when the point missed. An object stands on at most one edge
    # of the frame; the background, or the object merged into it when the two
    # are alike, reaches two or more.
    kept = [m for m in (largest(c) for c in masks) if clear(m, 1)]
    if not kept:
        # Every whole-object mask took the background along: use the smaller
        # masks, which stop at the object.
        kept = [m for m in (next((x for x in by_size(c) if clear(x, 1)), None) for c in masks) if m is not None]
    return kept, [1.0] * len(kept)


def by_size(candidates):
    return sorted(candidates, key=lambda m: -m.sum())


def largest(candidates):
    return by_size(candidates)[0]


def clear(mask, max_sides):
    return 0 < mask.mean() < 0.8 and touches_frame(mask) <= max_sides


def grid_points(image, n=4, margin=0.15, inside=None):
    x0, y0, x1, y1 = inside or (0, 0, *image.size)
    xs = np.linspace(x0 + (x1 - x0) * margin, x1 - (x1 - x0) * margin, n)
    ys = np.linspace(y0 + (y1 - y0) * margin, y1 - (y1 - y0) * margin, n)
    return [[int(x), int(y)] for y in ys for x in xs]


def touches_frame(mask):
    """How many of the four image edges the mask reaches."""
    return int(mask[0].any()) + int(mask[-1].any()) + int(mask[:, 0].any()) + int(mask[:, -1].any())


def default_box(image):
    w, h = image.size
    m = 0.02
    return [int(w * m), int(h * m), int(w * (1 - m)), int(h * (1 - m))]


def choose(masks, scores, select):
    if not masks:
        return None
    if select == "all":
        return np.any(np.stack(masks), axis=0)
    if select == "largest":
        return masks[int(np.argmax([m.sum() for m in masks]))]
    return masks[int(np.argmax(scores))]


def cut_out(image, mask, feather):
    import cv2
    alpha = (mask.astype(np.uint8) * 255)
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather)
    rgba = np.dstack([np.asarray(image.convert("RGB")), alpha])
    return Image.fromarray(rgba, "RGBA")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    parser.add_argument("output", help="RGBA PNG")
    parser.add_argument("--backend", default="auto", choices=["auto", "sam3", "sam2"],
                        help="auto: sam3 when --prompt is given, sam2 otherwise")
    parser.add_argument("--prompt", nargs="+", metavar="PHRASE",
                        help="sam3: short noun phrases, one per thing to keep, such as: farmer basket")
    parser.add_argument("--box", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), help="pixel box around the object")
    parser.add_argument("--point", type=int, nargs=2, metavar=("X", "Y"), help="sam2: a pixel on the object")
    parser.add_argument("--select", choices=["best", "largest", "all"],
                        help="keep the best-scoring mask, the largest, or the union of all. Default: largest for a "
                             "sam2 point, since its best-scoring mask is often a part; all otherwise")
    parser.add_argument("--threshold", type=float, default=0.5, help="sam3: minimum detection score")
    parser.add_argument("--feather", type=float, default=1.0, help="edge blur in pixels, 0 for a hard edge")
    parser.add_argument("--model", help="Hugging Face repo id to use instead of the default")
    parser.add_argument("--device", default="auto", choices=["auto", "xpu", "cpu"])
    args = parser.parse_args(argv)

    if args.backend == "auto":
        args.backend = "sam3" if args.prompt else "sam2"
    if args.backend == "sam2" and args.prompt:
        sys.exit("--prompt needs the sam3 backend; give sam2 a --box or --point, or nothing")
    if args.select is None:
        args.select = "largest" if args.backend == "sam2" and args.point else "all"
    image = Image.open(args.image).convert("RGB")
    device = device_for(args.device)
    t0 = time.time()
    masks, scores = (run_sam3 if args.backend == "sam3" else run_sam2)(image, args, device)
    mask = choose(masks, scores, args.select)
    if mask is None:
        sys.exit(f"{args.backend} found nothing" + (f" for {' / '.join(args.prompt)!r}" if args.prompt else "") + "; try another prompt or a box")
    print(f"{args.backend} on {device}: {len(masks)} candidate(s), best score {max(scores):.2f}, "
          f"mask covers {mask.mean():.1%} of the image, {time.time() - t0:.1f}s")
    cut_out(image, mask, args.feather).save(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
