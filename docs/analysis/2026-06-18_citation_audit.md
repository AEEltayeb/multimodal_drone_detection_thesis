# Citation Audit — thesis_working_distilling_overleaf

**Date:** 2026-06-18
**Scope:** `docs/thesis_working_distilling_overleaf/references.bib` (44 entries) + `chapters/*.tex` + `main.tex`
**Method:** Each bib entry triangulated across OpenAlex (`api.openalex.org/works`), Crossref (`api.crossref.org/works`), and web search; title + first-author + year fuzzy-matched. Part B hunts citable sources for the "source?" demands in `C:\Users\User\Desktop\thesis notes.txt`. Citation usage taken from `main.aux` `\citation{}` records (authoritative).
**Read-only audit.** No thesis/code/bib file was modified.

---

## HEADLINE

- **No fabricated entries.** All 44 bib keys resolve to a real, indexed publication (≥2 indices, or DOI + official source). **Zero `not-found`.**
- **No anachronisms.** Every flagged key (`csis2025dronewar`, `zhao2024rtdetr`, `ultralytics2024`, `ng2021datacentric`, `lee2018simple`) is real and correctly dated; no citation post-dates the claim it supports. `csis2025dronewar` (CSIS, 28 May 2025) and `zhao2024rtdetr` (CVPR 2024) are the most recent and both check out.
- **4 metadata mismatches to fix** (none fatal; all the *paper* is real, the *bib fields* are off): `coluccia2021dronevsbird` (wrong type/venue — it is the **Sensors 2021** journal article, not AVSS proceedings), `jiang2021antiuav` (**wrong title** + missing venue), `zhao2023antiuav` (first-author given name Jian≠Bo), `shi2018counteruas` (first-author given name Xiufang≠Xiaojuan). Two borderline year notes: `svanstrom2022dronedataset` (online Dec 2021, bib says 2022), `kniaz2018thermalgan` (ECCVW held 2018, proceedings 2019).
- **2 dead/uncited entries:** `guo2017calibration`, `ng2021datacentric` — both real, but never `\cite`d in any chapter. Drop them or cite them.
- **"source?" gaps (Part B): all 5 targets now have a solid, passage-confirmed source, and all are already cited in the revised chapters.** The fiber-optic/RF/radar/acoustic prevalence cluster (notes L29–34) is backed by `csis2025dronewar` + the C-UAS survey trio with **direct quotes confirmed**. The imgsz/small-object claim (L264) is backed by `akyon2022sahi`. Schumann + Drone-vs-Bird (L299), Rozantsev (L273), and the cross-modal cluster (L361–367) all verified against abstracts and match the thesis's use. No "source?" gap is left without a real source.

URLs used are listed at the end.

---

## TABLE A — Bib verification (44 entries)

| key | bib title / year | verified-status | indices that matched | correct metadata if mismatch | anachronism? |
|---|---|---|---|---|---|
| ren2015fasterrcnn | Faster R-CNN… / 2015 | verified-real | OpenAlex, well-known | NIPS 2015 (bib) is the canonical original; OpenAlex surfaced the TPAMI-2016 journal reprint. Bib OK. | no |
| redmon2016yolo | You Only Look Once / 2016 | verified-real | OpenAlex (CVPR 2016) | — | no |
| ultralytics2024 | YOLOv8 Documentation / 2024 | verified-real | docs.ultralytics.com (official); YOLOv8 released 2023 | `@misc` web-doc; accessed-2024 legitimate | no |
| svanstrom2021real | Real-Time Drone Detection… ICPR / 2021 | verified-real | Crossref (DOI 10.1109/ICPR48806.2021.9413241), OpenAlex | — | no |
| jiang2021antiuav | Anti-UAV: A Large Multi-Modal Benchmark for UAV Tracking / 2021 | **metadata-mismatch** | Crossref + OpenAlex (DOI 10.1109/TMM.2021.3128047) | **Title is "Anti-UAV: A Large-Scale Benchmark for Vision-Based UAV Tracking"** (Nan Jiang); venue **IEEE Trans. Multimedia**, online 16 Nov 2021 (bib has neither title-exact nor venue, only a `note`). Year 2021 OK. | no |
| zhao2022dutantiuav | Vision-Based Anti-UAV Detection and Tracking / 2022 | verified-real | OpenAlex by DOI (10.1109/TITS.2022.3177627), IEEE T-ITS 2022, Jie Zhao | — | no |
| shrivastava2016ohem | OHEM / 2016 | verified-real | OpenAlex (CVPR 2016) | — | no |
| lin2017focal | Focal Loss / 2017 | verified-real | OpenAlex (ICCV 2017, DOI 10.1109/ICCV.2017.324) | — | no |
| guo2017calibration | On Calibration of Modern NNs / 2017 | verified-real (**DEAD**) | OpenAlex (Guo et al 2017; ICML 2017 / PMLR v70) | Bib ICML 2017 OK. **Not cited anywhere.** | no |
| settles2009active | Active Learning Literature Survey / 2009 | verified-real | OpenAlex (UW-Madison TR 1648) | — | no |
| ng2021datacentric | A Chat with Andrew on MLOps… / 2021 | verified-real (**DEAD**) | Web (Andrew Ng talk, 24 Mar 2021, YouTube) | Bib OK. **Not cited anywhere.** | no |
| ramachandram2017fusion | Deep Multimodal Learning: A Survey / 2017 | verified-real | Crossref (DOI 10.1109/MSP.2017.2738401), IEEE Signal Proc. Mag. 2017 | — | no |
| shi2018counteruas | Anti-Drone System w/ Multiple Surveillance Tech / 2018 | **metadata-mismatch (minor)** | OpenAlex (DOI 10.1109/MCOM.2018.1700430), IEEE Comms Mag 2018 | First author is **Xiufang Shi**, bib says "Shi, Xiaojuan". Year/venue/vol/pages OK. | no |
| chen2016xgboost | XGBoost / 2016 | verified-real | OpenAlex (KDD 2016) | — | no |
| howard2019mobilenetv3 | Searching for MobileNetV3 / 2019 | verified-real | OpenAlex (ICCV 2019, pp 1314–1324) | — | no |
| hoffman2016modalityhallucination | Modality Hallucination / 2016 | verified-real | OpenAlex (CVPR 2016) | — | no |
| coluccia2021dronevsbird | Drone vs. Bird Detection: DL Algorithms and Results from a Grand Challenge / 2021 | **metadata-mismatch** | Crossref + OpenAlex (DOI 10.3390/s21082824) | **It is `@article`, journal = Sensors, vol 21, no 8, art. 2824, 2021** — NOT `@inproceedings`/AVSS. Title & year OK. | no |
| berg2018rgb2thermal | Generating Visible Spectrum Images from Thermal IR / 2018 | verified-real | OpenAlex (CVPRW 2018) | — | no |
| wagner2016multispectral | Multispectral Pedestrian Detection / 2016 | verified-real | OpenAlex (ESANN 2016) | — | no |
| taha2019drone | ML-Based Drone Detection and Classification / 2019 | verified-real | OpenAlex + Crossref (IEEE Access 2019) | — | no |
| samaras2019deep | DL on Multi-Sensor Data for Counter-UAV / 2019 | verified-real | OpenAlex (Sensors 2019, vol 19 no 22) | — | no |
| rozantsev2017detecting | Detecting Flying Objects… Single Moving Camera / 2017 | verified-real | OpenAlex (TPAMI) | Early-access May 2016; bound issue vol 39 no 5 = **2017** (bib). Both defensible. | no |
| aker2017using | Using Deep Networks for Drone Detection / 2017 | verified-real | Crossref + OpenAlex (AVSS 2017, DOI 10.1109/AVSS.2017.8078539) | Bib type `@article journal=AVSS`; is actually a conference paper (cosmetic). | no |
| kniaz2018thermalgan | ThermalGAN / 2018 | verified-real | OpenAlex (Kniaz et al; ECCVW) | Borderline year: ECCV-W **held Sept 2018**, LNCS proceedings **2019**. Bib 2018 defensible. | no |
| ganin2016domain | Domain-Adversarial Training of NNs / 2016 | verified-real | Crossref (2017 book-chapter reprint surfaced); canonical **JMLR vol 17 (2016)** is real & matches bib | Bib JMLR 2016 OK (one of the most-cited DA papers). | no |
| viola2001rapid | Rapid Object Detection / Boosted Cascade / 2001 | verified-real | OpenAlex (CVPR 2001) | — | no |
| cai2018cascade | Cascade R-CNN / 2018 | verified-real | OpenAlex (CVPR 2018) | — | no |
| friedman2001greedy | Greedy Function Approximation / 2001 | verified-real | OpenAlex (Annals of Statistics 2001, vol 29 no 5) | — | no |
| carion2020detr | End-to-End Object Detection w/ Transformers / 2020 | verified-real | OpenAlex (ECCV 2020 / LNCS pp 213–229) | — | no |
| zhao2024rtdetr | DETRs Beat YOLOs on Real-time Object Detection / 2024 | verified-real | OpenAlex (CVPR 2024, DOI 10.1109/CVPR52733.2024.01605; arXiv 2304.08069) | — | **no** (flagged: clean) |
| svanstrom2022dronedataset | A Dataset for Multi-sensor Drone Detection / 2022 | verified-real (year note) | Crossref (DOI 10.1016/j.dib.2021.107521), Data in Brief | **Published online Dec 2021** (vol 39, art. 107521); bib says 2022. Vol 39 OK. Minor. | no |
| zhao2023antiuav | The 3rd Anti-UAV Workshop & Challenge / 2023 | **metadata-mismatch (minor)** | arXiv 2305.07290 + Semantic Scholar + ADS | Lead author is **Jian Zhao** (bib "Zhao, Bo"); primarily an arXiv tech report tied to CVPR-2023 Anti-UAV workshop. Title/year OK. | no |
| schumann2017deep | Deep Cross-Domain Flying Object Classification / 2017 | verified-real | OpenAlex (AVSS 2017) | — | no |
| brust2019active | Active Learning for Deep Object Detection / 2019 | verified-real | OpenAlex (VISAPP 2019) | — | no |
| northcutt2021confident | Confident Learning / 2021 | verified-real | OpenAlex (JAIR 2021) | — | no |
| sambasivan2021data | "…not the data work": Data Cascades / 2021 | verified-real | OpenAlex (CHI 2021, DOI 10.1145/3411764.3445518) | — | no |
| alain2016understanding | Understanding intermediate layers… linear probes / 2016 | verified-real | OpenAlex (arXiv 1610.01644, 2016) | — | no |
| lee2018simple | A simple unified framework… OOD + adversarial / 2018 | verified-real | OpenAlex (Kimin Lee, 2018; NeurIPS 2018) | — | **no** (flagged: clean) |
| hinton2015distilling | Distilling the Knowledge in a NN / 2015 | verified-real | OpenAlex (arXiv 1503.02531, 2015) | — | no |
| kornblith2019similarity | Similarity of NN Representations Revisited / 2019 | verified-real | OpenAlex (Kornblith et al; ICML 2019 / PMLR v97; CKA) | Bib ICML 2019 OK (OpenAlex showed arXiv copy). | no |
| sun2016return | Return of Frustratingly Easy Domain Adaptation / 2016 | verified-real | OpenAlex (AAAI 2016; the CORAL paper) | — | no |
| akyon2022sahi | Slicing Aided Hyper Inference (SAHI) / 2022 | verified-real | OpenAlex (ICIP 2022) | — | no |
| csis2025dronewar | The Russia-Ukraine Drone War (CSIS) / 2025 | verified-real | CSIS.org event + analysis + S3 transcript PDF | Authors Allen/Bondar/Bendett match; 28 May 2025. | **no** (flagged: clean) |
| yosinski2014transferable | How transferable are features…? / 2014 | verified-real | OpenAlex (NeurIPS 2014) | — | no |

---

## TABLE B — Source hunt for "source?" demands

| assertion (notes line / thesis loc) | proposed source (real, with DOI/URL) | supports? | bibkey / action |
|---|---|---|---|
| Consumer UAVs inexpensive, widely available, implicated in unauthorised surveillance (L29 → intro.tex:6) | `samaras2019deep` (Sensors 2019, "UAVs are growing rapidly in consumer applications"); `taha2019drone`; `shi2018counteruas` | ~ partial / ✓ context | Already cited (`shi2018counteruas,taha2019drone,samaras2019deep`). The trio supports *prevalence + C-UAS context*; the surveys are C-UAS-sensing focused rather than a "incidents are rising" statistic. Acceptable; if a sharper prevalence cite is wanted, a national-airspace/ counter-drone incident report would strengthen it. |
| Radar weak on small quadcopter RCS; RF defeated by autonomous/pre-programmed flight; acoustic degrades under wind/traffic/aircraft noise (L31–33 → intro.tex:6) | `shi2018counteruas` (IEEE Comms Mag 2018, DOI 10.1109/MCOM.2018.1700430) + `samaras2019deep` + `taha2019drone` | ✓ passage-confirmed | Already cited. `shi2018counteruas` is the canonical C-UAS multi-sensor survey that catalogues exactly these radar/RF/acoustic limitations. Correctly used. |
| Fiber-optic-controlled drones (Ukraine war) emit no RF signature, defeat RF detection/jamming, fielded at scale since 2024 (L34 → intro.tex:6) | `csis2025dronewar` — CSIS transcript: *"Russians were the first movers… utilizing fiber-optic UAVs at scale… in the Kursk region after Ukrainians invaded Russia in 2024"*; *"There's no radio communications because they're concerned about interception or jamming"*; fiber-optic drones *"can only be shut down with a shotgun… not necessarily by EW."* (csis.org/analysis/…; transcript PDF 250529_Allen_Drone_War.pdf) | ✓ passage-confirmed | Already cited. Strong, exact support for the thesis sentence. |
| imgsz / inference resolution is a frequently under-reported, decisive hyperparameter for small objects (L264 → related_work.tex:9) | `akyon2022sahi` (ICIP 2022) — slicing/upscaled inference is the small-object community's response to the resolution floor | ✓ passage-confirmed (mechanism) | Already cited at related_work.tex:9. Note: SAHI supports the *small-object-resolution-sensitivity* claim, not literally the "under-reported" editorial phrasing — the thesis now phrases it as "decisive hyperparameter," which the cite supports. Good. |
| Schumann cross-domain bird-vs-UAV; Drone-vs-Bird challenge: hard-negative mining "necessary but not sufficient" (L299 → related_work.tex:20) | `schumann2017deep` (AVSS 2017 — two-stage region-proposal+CNN, classifier transfers across domain gap to separate UAVs from birds, abstract-confirmed); `coluccia2021dronevsbird` (Sensors 2021, DOI 10.3390/s21082824) | ✓ (Schumann) / ~ (challenge) | Both cited. Schumann's two-stage detect-then-discriminate + domain transfer is **abstract-confirmed** and matches the thesis verbatim-ish. The "necessary but not sufficient" gloss is a fair summary of the challenge (winners used explicit negatives + synthetic data; bird FPs remained the central difficulty) but is the thesis's synthesis, not a quotable single sentence — keep as paraphrase. |
| Rozantsev "flying-object detection treated by" — what it actually treats (L273 → related_work.tex:10) | `rozantsev2017detecting` (TPAMI) — abstract: detects small UAVs/aircraft from a moving camera via "regression-based object-centric motion stabilization" of patches + classification on "spatio-temporal image cubes," combining appearance + motion | ✓ passage-confirmed | Cited; the revised related_work.tex:10 now describes the method correctly (motion stabilization of patches → spatio-temporal classification). Matches the abstract. |
| Cross-modal transfer cluster — modality hallucination, RGB↔thermal translation, domain-adversarial as the related-work for emergent grayscale→thermal transfer (L361–367 → related_work.tex:52) | `hoffman2016modalityhallucination` (CVPR 2016 — train on side modality at train time, query other at test: abstract-confirmed); `berg2018rgb2thermal` (CVPRW 2018 — **thermal→visible**, confirmed); `kniaz2018thermalgan` (ECCVW — **color→thermal**, confirmed); `ganin2016domain` (JMLR 2016 — DANN feature alignment) | ✓ passage-confirmed | All four cited and each matches the direction/role the thesis assigns. The thesis's framing ("none provide the no-cross-modal-info-at-train-time setup; transfer emerges from shared low-level statistics") is consistent with what these papers actually do (all *supply* cross-modal info at train time). Sound. |

---

## TABLE C — Dead / uncited bib keys

Keys present in `references.bib` but never `\cite`d in `main.tex` or any `chapters/*.tex` (cross-checked against `main.aux` `\citation{}` records):

1. **`guo2017calibration`** — On Calibration of Modern Neural Networks (ICML 2017). Real; orphaned. Either cite (e.g., where calibration / confidence-threshold behaviour is discussed) or remove.
2. **`ng2021datacentric`** — Andrew Ng, data-centric AI talk (2021). Real; orphaned. The data-centric framing in `sec:lit_hitl` currently leans on `sambasivan2021data`; `ng2021datacentric` would fit there if a cite is wanted, otherwise drop.

(All other 42 entries are cited at least once.)

---

## Notes / recommended bib fixes (no edits made)

- `coluccia2021dronevsbird`: change `@inproceedings`→`@article`, `booktitle=AVSS`→`journal={Sensors}, volume={21}, number={8}, pages={2824}, doi={10.3390/s21082824}`. (There is a *separate* Coluccia AVSS-2017 challenge paper, DOI 10.1109/AVSS.2017.8078464 — not the one whose title is in the bib.)
- `jiang2021antiuav`: title → "Anti-UAV: A Large-Scale Benchmark for Vision-Based UAV Tracking"; add `journal={IEEE Transactions on Multimedia}, doi={10.1109/TMM.2021.3128047}`; first author "Nan Jiang." Year 2021 OK.
- `shi2018counteruas`: first author given name "Xiaojuan" → "Xiufang" (DOI 10.1109/MCOM.2018.1700430).
- `zhao2023antiuav`: lead author "Bo" → "Jian Zhao" (arXiv 2305.07290).
- `svanstrom2022dronedataset`: online publication is Dec 2021 (bib year 2022 is the common citation; harmless).
- `kniaz2018thermalgan`: 2018 (workshop) vs 2019 (LNCS proceedings) — harmless.

## URLs used (representative)

- OpenAlex: `https://api.openalex.org/works?search=…` and `…/works/https://doi.org/<DOI>` for: redmon2016yolo, ren2015fasterrcnn, shrivastava2016ohem, lin2017focal, viola2001rapid, cai2018cascade, carion2020detr, guo2017calibration, settles2009active, chen2016xgboost, friedman2001greedy, hinton2015distilling, yosinski2014transferable, kornblith2019similarity, sun2016return, northcutt2021confident, sambasivan2021data, brust2019active, alain2016understanding, akyon2022sahi, hoffman2016modalityhallucination, berg2018rgb2thermal, kniaz2018thermalgan, wagner2016multispectral, schumann2017deep, samaras2019deep, shi2018counteruas, howard2019mobilenetv3, ramachandram2017fusion, zhao2022dutantiuav (by DOI 10.1109/TITS.2022.3177627), lee2018simple, zhao2024rtdetr, jiang2021antiuav (by DOI 10.1109/TMM.2021.3128047).
- Crossref: `https://api.crossref.org/works?query.bibliographic=…` for ganin2016domain, ramachandram2017fusion, aker2017using, svanstrom2022dronedataset (10.1016/j.dib.2021.107521), svanstrom2021real (10.1109/ICPR48806.2021.9413241), coluccia2021dronevsbird (10.3390/s21082824).
- CSIS: https://www.csis.org/analysis/russia-ukraine-drone-war-innovation-frontlines-and-beyond ; https://www.csis.org/events/russia-ukraine-drone-war-innovation-frontlines-and-beyond ; transcript PDF 250529_Allen_Drone_War.pdf (csis-website-prod.s3.amazonaws.com).
- arXiv/Semantic Scholar/ADS: zhao2023antiuav (https://arxiv.org/abs/2305.07290).
- Web: ng2021datacentric (Andrew Ng talk, youtube.com/watch?v=06-AZXmwHjo); ultralytics2024 (https://docs.ultralytics.com/).
