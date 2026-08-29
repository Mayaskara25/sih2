> **⚠️ CORRECTIONS APPLIED BY W1 ORCHESTRATOR (2026-08-29).** This file is recon from
> documentation. Several items were afterwards verified empirically against the real
> artifacts, and two of the findings below are WRONG. Authoritative facts live in
> PLAN.md §2.2b and §7/W1 — prefer those on any conflict.
>
> 1. **S1↔S2 join columns: the real columns are `patch_id` (S2 patch name) and `s1_name`.**
>    This file says `patch_id_s1`; that column does not exist. Verified by reading the
>    actual metadata.parquet.
> 2. **`BigEarthNet.txt.parquet` IS directly HTTP-downloadable** (HTTP 200,
>    466,819,745 bytes) at
>    `huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/resolve/main/BigEarthNet.txt.parquet`.
>    No `git clone` or HF library needed.
> 3. **Tar member layout is now known** (was listed as a gap) — see PLAN.md §2.2b.
> 4. Still genuinely open: SECOND's total size, and direct links for CDVQA/SECOND
>    (Google Drive / Baidu, so plain curl will not work).

# W1 Reconnaissance: CDVQA, SECOND, BigEarthNet.txt, BigEarthNet v2.0

**Date**: 2026-08-29 | **Free disk**: ~33 GB | **Budget**: 1-2 GB (CDVQA/SECOND), stream-only BigEarthNet imagery

---

## CDVQA Dataset

| Item | Value | Source |
|------|-------|--------|
| **Q&A annotations download** | GitHub repo (files: Train_questions.json, Train_images.json, etc.) | https://github.com/YZHJessica/CDVQA |
| **Image pair count** | 2,968 pairs | https://arxiv.org/pdf/2112.06343 |
| **Image resolution** | 512 × 512 pixels | https://arxiv.org/pdf/2112.06343 |
| **Q&A pair count** | 122,000+ pairs | Web search result |
| **Reference change masks** | Yes (pixel-level annotations from SECOND) | https://arxiv.org/pdf/2112.06343 |
| **Test split identification** | Two test sets: Test1 (same distribution as train/val); Test2 (harder, different Q-type distribution) | https://arxiv.org/pdf/2112.06343 |
| **Direct annotation download URL** | NOT FOUND (file structure documented, direct URL not listed) | https://github.com/YZHJessica/CDVQA |

**Verdict**: CDVQA Q&A annotations are on GitHub. SECOND dataset is the foundation. Direct HTTP download link for CDVQA annotations not documented.

---

## SECOND Change Detection Dataset

| Item | Value | Source |
|------|-------|--------|
| **Bi-temporal pairs** | 4,662 total (2,968 train / 1,694 test) | https://captain-whu.github.io/SCD |
| **Image resolution** | 512 × 512 pixels | https://captain-whu.github.io/SCD |
| **Spatial resolution** | 0.53 m/pixel (range: 0.3–5 m) | https://arxiv.org/pdf/2112.06343 |
| **Landcover classes** | 6: non-vegetated, trees, low vegetation, water, buildings, playgrounds | https://captain-whu.github.io/SCD |
| **Reference change masks** | Yes (pixel-level expert annotations) | https://captain-whu.github.io/SCD |
| **Coverage** | Hangzhou, Chengdu, Shanghai (China) | https://captain-whu.github.io/SCD |
| **Primary download (Google Drive)** | https://drive.google.com/file/d/1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u/view | https://captain-whu.github.io/SCD |
| **Alternative download (Baidu)** | baidu.com (password: x4n8) | Web search result |
| **Size in GB** | NOT FOUND | — |

**Verdict**: SECOND is accessible via Google Drive or Baidu. Google Drive link may require browser/rclone (not curl). Total size unconfirmed.

---

## BigEarthNet.txt Annotations (HuggingFace)

| Item | Value | Source |
|------|-------|--------|
| **Dataset URL** | https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt | HF direct |
| **File format** | Parquet | https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt |
| **File size** | 467 MB (~490 million bytes) | https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt |
| **Exact byte size** | NOT FOUND (only "467 MB" listed) | — |
| **Columns** | `ID`, `s1_name`, `patch_id`, `input`, `output`, `type`, `category`, `split`, `latitude`, `longitude`, `country`, `season`, `climate_zone` | https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt |
| **Sample-type values** | `binary`, `mcq`, `captioning`, `bounding_box` | https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt |
| **s1_name column** | Sentinel-1 patch name from BigEarthNet v2.0 | https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt |
| **patch_id column** | Sentinel-2 patch name from BigEarthNet v2.0 | https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt |
| **Direct URL loading** | NOT CONFIRMED (requires `git clone` or HF `datasets` library) | https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt |

**Verdict**: Parquet is 467 MB. Columns confirmed, joins to BEN v2.0 via `s1_name` ↔ `patch_id`. Direct HTTP URL not documented.

---

## BigEarthNet v2.0 Imagery (Zenodo)

| Item | File | Size (GB) | Size (Bytes) | Source |
|------|------|-----------|-------------|--------|
| **S1 archive** | BigEarthNet-S1.tar.zst | 54.4 | 54,439,153,171 | https://zenodo.org/records/10891137 |
| **S2 archive** | BigEarthNet-S2.tar.zst | 63.3 | 63,251,710,377 | https://zenodo.org/records/10891137 |
| **Metadata** | metadata.parquet | 3.6 MB | 3,616,349 | https://zenodo.org/records/10891137 |
| **Reference maps** | Reference_Maps.tar.zst | 282.4 MB | 282,391,301 | https://zenodo.org/records/10891137 |

**Direct HTTP Download URL Pattern:**
```
https://zenodo.org/api/records/10891137/files/[FILENAME]/content
```
Example: `https://zenodo.org/api/records/10891137/files/BigEarthNet-S1.tar.zst/content`

### Internal Directory Structure

| Aspect | Details | Source |
|--------|---------|--------|
| **Patch organization** | One directory per patch. Within each: GeoTIFF band files + JSON metadata. | https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/bigearthnet.py |
| **S2 naming** | Underscore-separated patch ID; tile = remove last 2 components | https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/bigearthnet.py |
| **S1 naming** | Underscore-separated patch ID; tile = remove last 3 components | https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/bigearthnet.py |
| **S1-S2 mapping column** | `patch_id_s1` (S1) and `patch_id` (S2) in metadata.parquet | Web search result |
| **Band count** | S2: 12 bands; S1: 2 bands (VV, VH) | https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/bigearthnet.py |
| **Resolution** | Resampled to 120 × 120 pixels (loading) | https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/bigearthnet.py |

**Verdict**: Sizes confirmed. Zenodo API URL pattern works for curl. Archives contain patch subdirs with GeoTIFFs; naming convention documented. S1↔S2 mapping via metadata.parquet.

---

## NOT FOUND / Needs Human Check

1. **SECOND dataset total size in GB** — only pair count (4,662) and resolution (512×512) documented
2. **CDVQA direct HTTP Q&A download link** — GitHub repo exists, but no direct .zip/tar URL documented
3. **BigEarthNet.txt exact byte size** — listed as "467 MB" only; exact byte count not specified
4. **BigEarthNet.txt direct HTTP URL** — HF page requires `git clone` or library; resolvable URL not documented
5. **BigEarthNet v2.0 exact tar member listing** — patch naming convention inferred from code, not from archive listing

