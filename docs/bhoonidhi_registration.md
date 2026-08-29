# Bhoonidhi — registration & data pull (W1, do this on day 1)

**Why this is on the critical path:** the hidden grading set is real Cartosat-2S optical +
RISAT SAR. Everything we train on is Sentinel. Bhoonidhi is the only free source of actual
ISRO-sensor data, so it is the only way to find out about the Sentinel→Cartosat/RISAT
domain gap *before* judging day rather than during it. Registration has human-approval
latency you cannot compress, which is why it starts on day 1 even though the data isn't
needed until week 3.

**Target:** 5–10 scenes. Minimum viable is **one Cartosat scene + one RISAT scene over the
same area**; a genuinely co-registered pair is the ideal but not required (we can register
them ourselves — that's what W4's registration code is for).

**Disk:** budget ~3 GB (PLAN.md §2.2a). Cartosat scenes are large. Prefer small AOI
subsets over full scenes wherever the ordering UI allows it.

---

## 1. Register

1. Go to **https://bhoonidhi.nrsc.gov.in**
2. **New User Registration** → choose the **Indian Academic / Student** category if
   offered; it typically gets wider free-product access than a generic public account.
3. You'll need:
   - Institutional email if you have one — use it over a personal address, it speeds
     approval and unlocks the academic tier
   - Student ID / institution name
   - Purpose of use — write something specific and true, e.g. *"Academic research for
     Smart India Hackathon 2026, ISRO problem statement 26167: multimodal optical–SAR
     remote-sensing analysis."* Naming the ISRO PS is legitimate and helps.
4. Verify your email, then **wait for approval**. This is the part with latency — it can be
   same-day or a few days. If nothing after ~3 working days, follow up via the portal's
   contact form.

> While you wait, W1 can proceed on everything else — VRSBench, RSVQA, CDVQA/SECOND and the
> BigEarthNet streaming all have zero dependency on Bhoonidhi.

## 2. Find the data

Once logged in, use **Open Data Archive** / the search-and-order interface:

**Cartosat-2S (optical):**
- Satellite/Sensor: `Cartosat-2 Series` → `PAN` (panchromatic, ~0.65 m, **1 band**) and/or
  `MX` (multispectral, ~2 m, **4 bands**)
- Draw an AOI over a target with visible mixed land cover — urban + water + vegetation in
  one frame is ideal, since that exercises every class our encoder predicts
- Filter: cloud cover < 10%, and note the acquisition date — you'll want the RISAT scene
  reasonably close in time
- Product level: georeferenced/ortho if offered (saves W4/W5 a registration step)

**RISAT (SAR):**
- Satellite/Sensor: `RISAT-1` or `RISAT-1A` → look for `MRS` / `FRS-1` / `FRS-2` modes
- **Same AOI** as the Cartosat scene — this is the whole point
- Note the polarisation (single-pol → 1 band, dual-pol → 2 bands); grab a dual-pol product
  if available, it's more informative for the fusion module
- If RISAT coverage over your AOI is thin, **EOS-04 (RISAT-1A)** or the Sentinel-1 regional
  mirror on Bhoonidhi are acceptable substitutes — record which you used

**Practical note:** if Cartosat+RISAT overlap is hard to find for one AOI, pick the AOI
*from* the RISAT coverage rather than the other way round. SAR coverage is the scarcer of
the two.

## 3. Also grab (cheap, high value)

- A **bi-temporal Cartosat pair** over one AOI at two dates, if the archive has one. This
  gives W4 a real-sensor change test instead of only SECOND/CDVQA.
- The product **metadata/XML** alongside each scene — band order, polarisation, and
  calibration constants live there, and W0's `raster.py` + W2's §4.5 band mapping both need
  to know the actual band order rather than assuming one.

## 4. Where to put it

```
data/bhoonidhi/
├── cartosat/          # scenes + their metadata XML
├── risat/
└── README.md          # what you downloaded, from where, acquisition dates, AOI, band order
```

`data/` is gitignored (PLAN.md §5.7). Write a manifest to `data/manifests/bhoonidhi.json`
(file list, checksums, sensor, acquisition date, band count, AOI) — that IS committed, and
it's what makes the sample set reproducible for anyone else picking up the repo.

## 5. Verify immediately — this is the actual deliverable

Downloading isn't the point; **confirming our IO handles it** is. As soon as the files land:

```bash
uv run python -c "
from satquery.io.raster import load_raster
for p in ['data/bhoonidhi/cartosat/<scene>.tif', 'data/bhoonidhi/risat/<scene>.tif']:
    r = load_raster(p)
    print(p, '->', r.band_count, 'bands |', r.modality, '| georef:', r.is_georeferenced,
          '| display', r.display_rgb.shape, r.display_rgb.dtype)
"
```

Expected: Cartosat PAN reports **1 band**, Cartosat MX **4 bands**, RISAT **1 or 2 bands** —
and every one produces a valid `display_rgb`. This is PLAN.md W1's acceptance criterion and
the §2.3 check that matters most. **If it crashes or silently returns 3 bands, that is a W0
bug and it must be fixed immediately** — it means the code would have failed on the graded
data.

Record the real band counts and band order in `docs/status/W1.md`, then tell W2 — the §4.5
band-mapping policy currently *assumes* a Cartosat MX band order, and this is what confirms
or corrects it.
