# Model card — LXNet

> **This is not a medical device and must not be used to make clinical decisions.**
> It is a research artefact. Every number below was measured on one public dataset
> of unknown provenance, and none of it has been validated against a clinical
> reference standard.

| | |
|---|---|
| **Model** | LXNet — 3-block CNN, global average pooling, 2 dense layers |
| **Parameters** | 356,585 |
| **Input** | Single chest radiograph, 224×224, grayscale, CLAHE-equalised |
| **Output** | Probability distribution over 9 classes |
| **Version** | Trained 2026-07-31, seed 42, group-aware split |
| **License** | See repository |

---

## Intended use

**In scope**

- Research on lightweight architectures for radiograph classification.
- A worked example of near-duplicate–aware evaluation, which is the point of the
  repository.
- Education: a small model that trains end-to-end on one consumer GPU in ~40
  minutes.
- A baseline to beat.

**Explicitly out of scope**

- Any clinical use: diagnosis, triage, screening, or second reads.
- Any use where a missed finding causes harm. Pneumonia recall is 73.2%.
- Populations, scanners, or acquisition protocols unlike the training corpus —
  which is undocumented, so this is effectively *any* population.
- Paediatric imaging, portable/bedside films, lateral views. The training data
  composition on these is unknown and unverified.

---

## Measured performance

Group-aware hold-out set (997 images, no near-duplicate of any training image).

| Metric | Value |
|---|---|
| Accuracy | 92.88% |
| Macro precision | 93.16% |
| Macro recall | 95.06% |
| Macro F1 | 0.9382 |
| 5-fold CV accuracy | 94.39% ± 0.38 |

### Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 93.7 | **88.6** | 0.910 | 201 |
| Pneumonia | 93.5 | **73.2** | 0.821 | 157 |
| Higher Density | 88.5 | 100.0 | 0.939 | 100 |
| Lower Density | 96.8 | 100.0 | 0.984 | 91 |
| Obstructive Pulmonary | 94.8 | 94.8 | 0.948 | 96 |
| Degenerative Infectious | 83.0 | 100.0 | 0.907 | 88 |
| Encapsulated Lesions | 91.4 | 99.0 | 0.950 | 97 |
| Mediastinal Changes | 96.7 | 100.0 | 0.983 | 88 |
| Chest Changes | 100.0 | 100.0 | 1.000 | 79 |

---

## Limitations

**1. It misses roughly one pneumonia in four.** Recall 73.2% is the single most
important number here. 8% of pneumonia films are classified *Normal* — a
false-negative on the most common actionable finding in the set. Any use that
treats a Normal output as reassurance is misuse.

**2. Normal recall is 88.6%.** The two classes a screening tool exists to
separate are the two the model is worst at. The remaining seven classes sit at
99–100% and carry the headline accuracy.

**3. Accuracy is dataset-specific and probably optimistic.** Seven of nine classes
scoring ~100% is not a realistic difficulty profile for chest radiography. It more
likely indicates that those classes are separable by acquisition artefacts —
scanner, exposure, framing, or collection source — than that the model has learned
the pathology. This has not been tested and should be assumed until disproven.

**4. No external validation.** One dataset, one split, one institution's worth of
imaging (probably — provenance is undocumented). Nothing here predicts behaviour
on another hospital's films.

**5. Class definitions are coarse and non-standard.** "Higher Density" bundles
effusion, consolidation, hydrothorax and empyema — entities with different causes
and different management. The label set is not a clinical taxonomy.

**6. The interpretability maps are weak evidence.** Several Grad-CAM and Score-CAM
maps are diffuse or weight image borders rather than lung tissue. They show what
the network keys on; they do not demonstrate clinical reasoning, and a plausible
heatmap over a lung is not proof of a correct mechanism.

**7. No calibration.** Softmax outputs are not calibrated probabilities. A 0.95 is
not a 95% chance of being right, and the model has not been assessed for
reliability, expected calibration error, or behaviour under distribution shift.

**8. No out-of-distribution detection.** Given a hand, an abdomen, or a photograph
of a cat, the model returns a confident nine-class distribution. It cannot say
"this is not a chest X-ray."

---

## Training data

[X-Ray Lung Diseases Images (9 classes)](https://www.kaggle.com/datasets/fernando2rad/x-ray-lung-diseases-images-9-classes),
6,743 files.

**Known data quality issues, all handled and documented:**

- 85 byte-identical duplicate files → removed before splitting.
- 1 image filed under two contradictory classes (*Pneumonia* and *Lower Density*)
  → both copies dropped.
- 6,657 byte-unique files reduce to **4,635 visually distinct images**: 30% of the
  corpus is the same X-ray re-saved at different compression. Splits are dealt out
  by perceptual group so no evaluation image has a twin in training.
- 40 distinct resolutions, mixed L/RGB/RGBA colour modes.

**Unknown and unverifiable:** patient identities, whether one patient contributes
multiple films, scanner and acquisition parameters, geographic and demographic
composition, label provenance and whether labels were adjudicated. **No
demographic breakdown of performance is possible**, so fairness across sex, age,
or ethnicity is untested. This is a serious gap for any model touching medicine.

---

## Evaluation protocol

Splitting is group-aware: every re-saved copy of an X-ray is confined to a single
split. Under the random split this dataset invites, 40.4% of the test set has a
near-duplicate in training, which inflates hold-out accuracy by 2.5 points and
widens fold-to-fold spread four-fold.

Cross-validation carves each fold's early-stopping monitor out of that fold's
*training* rows, so the reported fold is scored exactly once and never influences
weight selection.

Oversampling to balance classes is applied to training folds only, after the
split.

Full detail: [RESULTS.md](RESULTS.md).

---

## Ethical considerations

Deploying an automated reader into a diagnostic pathway shifts risk onto patients
who did not consent to it and usually cannot see it. The failure mode that matters
is not the average accuracy but the missed finding: at 73% pneumonia recall, a
system trusted as a filter would send roughly a quarter of pneumonia patients home
with a reassuring result.

Automation bias is the compounding risk — a confident label shown to a tired
reader shifts their judgement even when it is wrong, and the interpretability
panel makes wrong answers *more* persuasive, not less.

Absent demographic metadata, this model cannot be audited for differential
performance across groups. A model that is 93% accurate overall can be far worse
for an under-represented subgroup, and nothing in this repository would reveal it.

---

## Reproducing

```bash
lxnet-train --out-dir runs/grouped --split-mode grouped --cv-models LXNet
lxnet-report
```

Seed 42, batch size 32, max 40 epochs with early stopping (patience 8), Adam.
Both evaluation arms consume the same cached CLAHE'd tensor, so preprocessing is
bit-for-bit identical between them and the split function is the only variable.
