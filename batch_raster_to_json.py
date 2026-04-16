#!/usr/bin/env python3
"""
Batch-convert RPLAN PNGs to RPLAN JSON format using raster_to_json.

Reads sample IDs from a plain-text file (one ID per line) and runs
raster_to_json on each corresponding PNG in parallel.  Outputs one JSON
per sample into the output directory, plus a list.txt manifest.

Usage:
    python batch_raster_to_json.py --png-dir /path/to/pngs --id-list ids_test.txt -o /path/to/out/test
    python batch_raster_to_json.py --png-dir /path/to/pngs --id-list ids_train.txt -o /path/to/out/train --workers 64
"""

import argparse
import json
import logging
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm

from read_dd import read_data

logger = logging.getLogger(__name__)


def convert_png_to_dict(png_path):
    """Run the raster_to_json logic and return the result dict directly.

    This is the core of raster_to_json.raster_to_json() but returns the
    dict instead of writing to disk, making it safe for parallel use.
    """
    room_type, poly, doors_, walls, out = read_data(png_path)

    d = []
    all_doors = []
    for i in range(1, len(doors_) + 1):
        if (i) % 4 == 0 and i + 1 != 1:
            d.append(doors_[i - 1])
            all_doors.append(d)
            d = []
        elif i == 1:
            d = []
        if i % 4 != 0:
            d.append(doors_[i - 1])

    al_dr = 0
    for hd in range(len(all_doors)):
        dr_t = []
        dr_in = []
        doors = all_doors[hd]
        d_t = 2
        t_x = abs(doors[0][1] - doors[1][1])
        t_y = abs(doors[0][0] - doors[3][0])
        ss = t_x
        if t_x > t_y:
            d_t = 1
            ss = t_y
        elif t_x < t_y:
            d_t = 3
        for pmc in range(5):
            for dw in range(len(doors)):
                for nw in range(len(walls)):
                    if walls[nw][5] == 17:
                        continue
                    if walls[nw][5] == 15:
                        continue
                    if ((d_t <= 2) & (doors[dw][0] - doors[dw][2] <= 1)
                            & (walls[nw][0] - walls[nw][2] <= 1)
                            & (abs(doors[dw][0] - walls[nw][0]) <= (ss - 1))
                            & (abs(doors[dw][2] - walls[nw][2]) <= (ss - 1))):
                        l = doors[dw][1]
                        r = doors[dw][3]
                        if l > r:
                            t = l; l = r; r = t
                        l_ = walls[nw][1]
                        r_ = walls[nw][3]
                        if l_ > r_:
                            t = l_; l_ = r_; r_ = t
                        if ((r - r_) <= pmc) & (pmc >= (l_ - l)):
                            if len(dr_in) < 2:
                                if walls[nw][6] not in dr_t:
                                    dr_t.append(walls[nw][6])
                                    dr_in.append(nw)
                    elif ((d_t >= 2) & (doors[dw][1] - doors[dw][3] <= 1)
                          & (walls[nw][1] - walls[nw][3] <= 1)
                          & (abs(doors[dw][1] - walls[nw][1]) <= (ss - 1))
                          & (abs(doors[dw][3] - walls[nw][3]) <= (ss - 1))):
                        l = doors[dw][0]
                        r = doors[dw][2]
                        if l > r:
                            t = l; l = r; r = t
                        l_ = walls[nw][0]
                        r_ = walls[nw][2]
                        if l_ > r_:
                            t = l_; l_ = r_; r_ = t
                        if ((r - r_) <= pmc) & (pmc >= (l_ - l)):
                            if len(dr_in) < 2:
                                if walls[nw][6] not in dr_t:
                                    dr_t.append(walls[nw][6])
                                    dr_in.append(nw)
        if len(dr_t) == 2:
            walls[dr_in[0]][8] = walls[dr_in[1]][5]
            walls[dr_in[0]][7] = walls[dr_in[1]][6]
            walls[dr_in[1]][8] = walls[dr_in[0]][5]
            walls[dr_in[1]][7] = walls[dr_in[0]][6]
            al_dr += 1

        assert len(dr_t) <= 2

    assert al_dr == (len(all_doors) - 1)

    omn = []
    tr = 0
    en_pp = 0
    for nw in range(len(walls) - (len(all_doors) * 4), len(walls)):
        if tr % 4 == 0:
            omn = []
        tr += 1
        for kw in range(len(walls) - (len(all_doors) * 4) + 1):
            if (walls[kw][5] == 17) & (walls[nw][5] == 17):
                continue
            if (walls[kw][5] == 15) & (walls[nw][5] == 15):
                continue
            if (walls[kw][5] == 15) & (walls[nw][5] == 17):
                continue
            for pmc in range(5):
                if ((abs(walls[kw][0] - walls[nw][0]) <= (ss - 1))
                        & (abs(walls[kw][2] - walls[nw][2]) <= (ss - 1))):
                    l = walls[kw][1]; r = walls[kw][3]
                    if l > r:
                        t = l; l = r; r = t
                    l_ = walls[nw][1]; r_ = walls[nw][3]
                    if l_ > r_:
                        t = l_; l_ = r_; r_ = t
                    if (pmc >= r_ - r) & (l - l_ <= pmc) & (nw != kw):
                        if ((walls[nw][5] == 17) & (walls[nw][8] == 0)
                                & (walls[kw][6] not in omn)):
                            walls[nw][8] = walls[kw][5]
                            walls[nw][7] = walls[kw][6]
                            omn.append(walls[kw][6])
                        if (walls[nw][5] == 15) & (walls[nw][8] == 0):
                            walls[nw][8] = walls[kw][5]
                            walls[nw][7] = walls[kw][6]
                            en_pp = 1
                if ((abs(walls[kw][1] - walls[nw][1]) <= (ss - 1))
                        & (abs(walls[kw][3] - walls[nw][3]) <= (ss - 1))):
                    l = walls[kw][0]; r = walls[kw][2]
                    if l > r:
                        t = l; l = r; r = t
                    l_ = walls[nw][0]; r_ = walls[nw][2]
                    if l_ > r_:
                        t = l_; l_ = r_; r_ = t
                    if (pmc >= r_ - r) & (l - l_ <= pmc) & (nw != kw):
                        if ((walls[nw][5] == 17) & (walls[nw][8] == 0)
                                & (walls[kw][6] not in omn)):
                            walls[nw][8] = walls[kw][5]
                            walls[nw][7] = walls[kw][6]
                            omn.append(walls[kw][6])
                        if (walls[nw][5] == 15) & (walls[nw][8] == 0):
                            walls[nw][8] = walls[kw][5]
                            walls[nw][7] = walls[kw][6]
                            en_pp = 1

    for i in range(1):
        for iw in range(len(walls)):
            tp_out = -1; dif_x = 10; dif_y = 10; type_out = 0
        for jw in range(len(walls)):
            if walls[iw][0] == walls[iw][2]:
                if walls[jw][0] != walls[jw][2]:
                    continue
                if (walls[iw][0] - walls[jw][0]) == (walls[iw][2] - walls[jw][2]):
                    rnp = walls[jw][1]; fnp = walls[jw][3]
                    rmp = walls[iw][1]; fmp = walls[iw][3]
                    if rnp < fnp:
                        t = fnp; fnp = rnp; rnp = t
                    if rmp < fmp:
                        t = fmp; fmp = rmp; rmp = t
                    if (abs(rmp) <= abs(rnp)) | (abs(fmp) <= abs(fnp)):
                        dif_x_temp = walls[iw][0] - walls[jw][0]
                        if (abs(dif_x) > abs(dif_x_temp)) & (iw != jw):
                            dif_x = dif_x_temp
                            tp_out = walls[jw][6]
                            type_out = walls[jw][5]
            elif walls[iw][1] == walls[iw][3]:
                if (walls[iw][1] - walls[jw][1]) == (walls[iw][3] - walls[jw][3]):
                    rnp = walls[jw][0]; fnp = walls[jw][2]
                    rmp = walls[iw][0]; fmp = walls[iw][2]
                    if rnp < fnp:
                        t = fnp; fnp = rnp; rnp = t
                    if rmp < fmp:
                        t = fmp; fmp = rmp; rmp = t
                    if (abs(rmp) <= abs(rnp)) | (abs(fmp) <= abs(fnp)):
                        dif_y_temp = walls[iw][1] - walls[jw][1]
                        if (abs(dif_y) > abs(dif_y_temp)) & (iw != jw):
                            dif_y = dif_y_temp
                            tp_out = walls[jw][6]
                            type_out = walls[jw][5]

    assert en_pp == 1, f"en_pp={en_pp}"

    km = 0
    edges = []
    ed_rm = []
    bboxes = []

    for w_i in range(len(walls)):
        edges.append([
            walls[w_i][0], walls[w_i][1],
            walls[w_i][2], walls[w_i][3],
            walls[w_i][5], walls[w_i][8],
        ])
        if walls[w_i][6] == -1:
            ed_rm.append([walls[w_i][7]])
        elif walls[w_i][7] == -1:
            ed_rm.append([walls[w_i][6]])
        else:
            ed_rm.append([walls[w_i][6], walls[w_i][7]])

    for i in range(len(poly)):
        p = poly[i]
        pm = []
        for p_i in range(p):
            pm.append([edges[km + p_i][0], edges[km + p_i][1]])
        km += p
        polygon = Polygon(pm)
        bbox = np.asarray(polygon.bounds)
        bboxes.append(bbox.tolist())

    return {
        "room_type": room_type,
        "boxes": bboxes,
        "edges": edges,
        "ed_rm": ed_rm,
    }


def _worker(args):
    """Multiprocessing worker. Returns (name, status)."""
    name, png_dir, out_dir = args
    png_path = str(png_dir / f"{name}.png")
    if not os.path.exists(png_path):
        return name, "missing_png"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = convert_png_to_dict(png_path)
    except Exception:
        return name, "error"

    out_path = out_dir / f"{name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f)
    return name, "ok"


def process(names, png_dir, out_dir, workers):
    """Convert a list of PNG IDs to JSON files in *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Processing %d samples -> %s", len(names), out_dir)

    work_items = [(name, png_dir, out_dir) for name in names]
    stats = {"ok": 0, "missing_png": 0, "error": 0}

    if workers > 1:
        with Pool(processes=workers) as pool:
            for name, status in tqdm(
                pool.imap_unordered(_worker, work_items, chunksize=1),
                total=len(names), desc="converting", unit="sample",
            ):
                stats[status] += 1
    else:
        for item in tqdm(work_items, desc="converting", unit="sample"):
            _, status = _worker(item)
            stats[status] += 1

    converted = sorted(f.stem for f in out_dir.glob("*.json"))
    list_path = out_dir / "list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for name in converted:
            f.write(f"{name}.json\n")

    logger.info(
        "  ok=%d  missing=%d  errors=%d  -> %s (%d files + list.txt)",
        stats["ok"], stats["missing_png"], stats["error"],
        out_dir, len(converted),
    )
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Batch-convert RPLAN PNGs to RPLAN JSON format",
    )
    parser.add_argument(
        "--png-dir", type=str, required=True,
        help="Directory containing RPLAN {name}.png files",
    )
    parser.add_argument(
        "--id-list", type=str, required=True,
        help="Text file with one PNG sample ID per line (no extension)",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, required=True,
        help="Output directory for converted JSON files",
    )
    parser.add_argument(
        "--workers", "-j", type=int, default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.id_list, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    logger.info("Loaded %d IDs from %s", len(names), args.id_list)

    process(names, Path(args.png_dir), Path(args.output_dir), args.workers)

    logger.info("=" * 60)
    logger.info("DONE. Output: %s", args.output_dir)


if __name__ == "__main__":
    main()
