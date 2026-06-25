"""
MedSeg-AI CLI
=============
Bağımsız komut satırı arayüzü — medseg_pipeline.MedSegPipeline'ı çağırır,
sonuçları JSON + PNG mask olarak kaydeder.

Kullanım:
    python cli.py --dicom_path /data/ct --output_dir /out
    python cli.py --dicom_path scan.dcm --output_dir /out --model_path /models/nnunet --no_sam2
    python cli.py --dicom_path /data/ct --output_dir /out --slice 50 --window bone
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np


# ── PNG mask görselleştirici ──────────────────────────────────────────────────

# Her sınıf ID için RGB renk (0=arka plan şeffaf, 1=kırmızı, 2=yeşil, ...)
_CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0:  (0,   0,   0),    # arka plan — siyah
    1:  (255, 60,  60),   # organ 1   — kırmızı
    2:  (60,  200, 60),   # organ 2   — yeşil
    3:  (60,  60,  255),  # organ 3   — mavi
    4:  (255, 200, 0),    # organ 4   — sarı
    5:  (200, 60,  255),  # organ 5   — mor
    6:  (0,   220, 220),  # organ 6   — cyan
    7:  (255, 140, 0),    # organ 7   — turuncu
    8:  (180, 255, 60),   # organ 8   — limon
    9:  (255, 60,  180),  # organ 9   — pembe
    10: (60,  180, 255),  # organ 10  — açık mavi
}


def _colorize_mask(mask_2d: np.ndarray) -> np.ndarray:
    """2D int maske → RGB uint8 (H, W, 3)."""
    h, w = mask_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in _CLASS_COLORS.items():
        rgb[mask_2d == cls_id] = color
    # Bilinmeyen sınıflar için beyaz
    known = np.isin(mask_2d, list(_CLASS_COLORS.keys()))
    rgb[~known] = (255, 255, 255)
    return rgb


def save_mask_png(mask_3d: np.ndarray, output_dir: Path,
                  slice_idx: int | None = None) -> list[Path]:
    """
    3D maske → PNG.
    slice_idx verilmişse tek dilim, verilmemişse orta 3 dilimi kaydeder.
    Döner: kaydedilen dosya yollarının listesi.
    """
    try:
        from PIL import Image
    except ImportError:
        logging.warning("Pillow kurulu değil — PNG kaydı atlandı (pip install Pillow)")
        return []

    D = mask_3d.shape[0]
    if slice_idx is not None:
        indices = [max(0, min(slice_idx, D - 1))]
    else:
        mid = D // 2
        indices = [max(0, mid - 1), mid, min(D - 1, mid + 1)]

    saved: list[Path] = []
    ts = time.strftime("%Y%m%d_%H%M%S")
    for i in indices:
        rgb = _colorize_mask(mask_3d[i])
        img = Image.fromarray(rgb, mode="RGB")
        out_path = output_dir / f"mask_slice{i:04d}_{ts}.png"
        img.save(str(out_path))
        saved.append(out_path)
    return saved


# ── Argparse ──────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="medseg-cli",
        description="MedSeg-AI: nnUNet + SAM2 tıbbi segmentasyon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dicom_path", required=True,
        help="DICOM dosyası (.dcm) veya seri klasörü",
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Çıktı klasörü (JSON, NIfTI, PNG)",
    )
    p.add_argument(
        "--model_path",
        default=None,
        help="nnUNet eğitilmiş model klasörü (varsayılan: $NNUNET_MODEL veya 'models/nnunet')",
    )
    p.add_argument(
        "--sam2_checkpoint", default="sam2_hiera_large.pt",
        help="SAM2 checkpoint .pt dosyası",
    )
    p.add_argument(
        "--sam2_cfg", default="sam2_hiera_l.yaml",
        help="SAM2 model konfigürasyon dosyası",
    )
    p.add_argument(
        "--window",
        default="soft_tissue",
        choices=["soft_tissue", "lung", "bone", "brain", "abdomen"],
        help="HU windowing ön ayarı",
    )
    p.add_argument(
        "--folds", nargs="+", type=int, default=[0],
        help="nnUNet inference fold listesi",
    )
    p.add_argument(
        "--no_sam2", action="store_true",
        help="SAM2 refinement adımını atla (yalnızca nnUNet)",
    )
    p.add_argument(
        "--slice", type=int, default=None, dest="slice_idx",
        help="PNG için dilim indeksi (varsayılan: orta 3 dilim)",
    )
    p.add_argument(
        "--device", default=None,
        help="Cihaz: cuda | cpu (varsayılan: otomatik)",
    )
    p.add_argument(
        "--no_png", action="store_true",
        help="PNG çıktısını oluşturma",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Ayrıntılı log",
    )
    return p


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("medseg.cli")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── model_path çöz ────────────────────────────────────────────────
    import os
    model_path = args.model_path or os.getenv("NNUNET_MODEL", "models/nnunet")

    # ── Pipeline ──────────────────────────────────────────────────────
    log.info("Pipeline başlatılıyor...")
    try:
        from medseg_pipeline import MedSegPipeline
    except ImportError as e:
        log.error(f"medseg_pipeline import hatası: {e}")
        return 1

    device = None
    try:
        import torch
        device = torch.device(args.device) if args.device else None
    except ImportError:
        pass

    pipeline = MedSegPipeline(
        model_folder=model_path,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_cfg=args.sam2_cfg,
        window=args.window,
        device=device,
        use_sam2=not args.no_sam2,
        nnunet_folds=tuple(args.folds),
    )

    # ── Predict ───────────────────────────────────────────────────────
    log.info(f"Segmentasyon başlıyor: {args.dicom_path}")
    try:
        result = pipeline.predict(args.dicom_path)
    except Exception as e:
        log.error(f"Segmentasyon hatası: {e}", exc_info=args.verbose)
        return 1

    # ── NIfTI kaydet ──────────────────────────────────────────────────
    try:
        nifti_path = pipeline.save_nifti(result, output_dir)
        log.info(f"NIfTI: {nifti_path}")
    except Exception as e:
        log.warning(f"NIfTI kayıt hatası: {e}")

    # ── JSON kaydet ───────────────────────────────────────────────────
    try:
        json_path = pipeline.save_json(result, output_dir)
        log.info(f"JSON: {json_path}")
    except Exception as e:
        log.warning(f"JSON kayıt hatası: {e}")

    # ── PNG kaydet ────────────────────────────────────────────────────
    png_paths: list[Path] = []
    if not args.no_png:
        try:
            png_paths = save_mask_png(
                pipeline._last_mask, output_dir, slice_idx=args.slice_idx
            )
            for p in png_paths:
                log.info(f"PNG: {p}")
        except Exception as e:
            log.warning(f"PNG kayıt hatası: {e}")

    # ── Özet JSON'a PNG yollarını ekle ───────────────────────────────
    if png_paths and json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        data["png_slices"] = [str(p) for p in png_paths]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ── Sonuç özeti ───────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  MedSeg-AI — SEGMENTASYON TAMAMLANDI")
    print("=" * 55)
    print(f"  Süre      : {result.total_sec:.1f}s")
    print(f"  Sınıflar  : {result.class_ids}")
    print(f"  Cihaz     : {result.device}")
    print(f"  Çıktı     : {output_dir}")
    print("=" * 55)
    return 0


if __name__ == "__main__":
    sys.exit(main())
