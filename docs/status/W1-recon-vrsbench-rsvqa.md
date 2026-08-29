# W1 Reconnaissance: VRSBench & RSVQA Download Feasibility

## VRSBench

| Item | Size | URL | Source |
|------|------|-----|--------|
| **Total dataset** | 12.5 GB | https://huggingface.co/datasets/xiang709/VRSBench | HuggingFace |
| Images_train.zip | 8.36 GB | " | HF Files tab |
| Images_val.zip | 3.98 GB | " | HF Files tab |
| Annotations_train.zip | 28.6 MB | " | HF Files tab |
| Annotations_val.zip | 12.8 MB | " | HF Files tab |
| VRSBench_train.json | 64.9 MB | " | HF Files tab |
| VRSBench_EVAL_Cap.json | 4.81 MB | " | HF Files tab |
| VRSBench_EVAL_referring.json | 10.3 MB | " | HF Files tab |
| VRSBench_EVAL_vqa.json | 9.36 MB | " | HF Files tab |
| Repo | | https://github.com/lx709/VRSBench | GitHub |
| Website | | https://vrsbench.github.io/ | Official |

**Test split downloadable separately?** No. Dataset has only train + validation splits. Evaluation files (Cap, referring, vqa) are provided separately but are small (~24.5 MB total). No separate test image set exists.

**Verdict:** ⚠️ **CANNOT obtain test-only.** Must download validation images (3.98 GB) for evaluation. Total for eval: **~4 GB** (validation images + eval JSONs). Train data (8.36 GB) would need separate handling if truly test-only required. **Budget impact: ~4 GB fits within 4 GB allowance, but leaves zero margin.**

**License:** Not explicitly stated on HF page; contact xiangli92@ieee.org per GitHub docs.

---

## RSVQA-LR

| Item | Size | URL | Source |
|------|------|-----|--------|
| **Total dataset** | ~150.5 MB | https://zenodo.org/records/6344334 | Zenodo |
| Images_LR.zip | 95.0 MB | " | Zenodo |
| all_questions.json | 15.3 MB | " | Zenodo |
| all_answers.json | 9.2 MB | " | Zenodo |
| LR_split_train_*.json (3 files) | ~15 MB | " | Zenodo |
| LR_split_val_*.json (3 files) | ~3 MB | " | Zenodo |
| LR_split_test_*.json (3 files) | ~2.5 MB | " | Zenodo |

**Test split downloadable separately?** Yes. Three separate JSON files: `LR_split_test_questions.json`, `LR_split_test_answers.json`, `LR_split_test_images.json`.

**Image specs:** 256×256 px, 772 total images (11.1% are test per search results).

**Verdict:** ✅ **Test-only is feasible.** Test split is ~2.5 MB (questions + answers) + images subset (~10 MB estimated). **Budget impact: <20 MB. Excellent fit.**

**Repo:** https://github.com/syvlo/RSVQA

---

## RSVQA-HR

| Item | Size | URL | Source |
|------|------|-----|--------|
| **Total dataset** | 14.4 GB | https://zenodo.org/records/6344366 | Zenodo |
| Images.tar | 13.5 GB | " | Zenodo |
| Split metadata (JSON) | ~900 MB combined | " | Zenodo |

**Test split downloadable separately?** Partial. Test split is identified by two variants:
- Standard: `USGS_split_test_*` prefix files
- Phili variant: `USGS_split_test_phili_*` prefix files
  
Images cannot be downloaded separately; Images.tar is monolithic (13.5 GB).

**Image specs:** 512×512 px, 10,659 total images. Test sets: Test Set 1 (20.5% of tiles), Test Set 2 (6.8% of tiles, unseen areas).

**Verdict:** ❌ **Test-only is NOT feasible.** Images (13.5 GB) ship as single .tar archive with train+val+test mixed. Must download entire archive to access test images. **Budget impact: Unaffordable at 13.5 GB.**

---

## Unresolved / "NOT FOUND"

- VRSBench: Image resolutions (per-split breakdown) — paper/GitHub has details, page only lists counts.
- VRSBench: License/registration — not on HF page.
- RSVQA-HR: Exact test image count in MB — metadata alone is ~900 MB; images monolithic.

## Summary

| Dataset | Total | Test-Only Cost | Feasible? |
|---------|-------|-----------------|-----------|
| VRSBench | 12.5 GB | 4.0 GB (val imgs) | ⚠️ Marginal |
| RSVQA-LR | 150.5 MB | <20 MB | ✅ Yes |
| RSVQA-HR | 14.4 GB | 13.5 GB (full img archive) | ❌ No |

**Recommendation:** Start with RSVQA-LR (tiny, fully separable). VRSBench requires val images (not "test" in strict sense). RSVQA-HR fails: monolithic image archive is unaffordable and violates test-only requirement.

---

**Report generated:** 2026-08-29  
**Sources:** HuggingFace, Zenodo, GitHub
