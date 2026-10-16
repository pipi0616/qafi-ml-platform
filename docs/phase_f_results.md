# Phase F: the science, in numbers

What these numbers add up to is in [`contributions.md`](contributions.md); this page is the numbers themselves.

Every number below is produced by a notebook in `notebooks/F_science/` and recorded in MLflow. The test
proteins were fixed in notebook 07a and the exam was opened once; each chapter says what it held out.

## 14a · Folding stability in kcal/mol

Trained on the mega-scale dataset (Tsuboyama et al. 2023) using the split published with ThermoMPNN, so
the numbers sit beside theirs. 28 held-out domains, 28,172 mutations.

| Model | Spearman | Pearson | Median per domain | RMSE (kcal/mol) |
|---|---|---|---|---|
| Conservation × burial (zero parameters) | 0.442 | 0.515 | 0.582 | — |
| **Ours, comparable (270 domains)** | **0.691** | **0.723** | **0.764** | **0.736** |
| Ours, decontaminated (218 domains) | 0.681 | 0.711 | 0.759 | 0.744 |
| ThermoMPNN (published) | 0.725 | 0.754 | 0.77 | 0.708 |
| SPURS (published) | 0.74 | 0.78 | 0.83 | — |

Reproduced the published filtering chain exactly (776,298 → 607,839 → 391,090 → 272,712) and traced a
140-row discrepancy in the paper's stated test size to wild-type control rows.

**Cross-dataset leakage:** 56 mega-scale domains are homologous to this project's locked test proteins.
Removing them costs 0.010 Spearman; everything downstream uses the decontaminated model.

## 14b · Sequencing reads to fitness

Reprocessed a published ddPCA experiment (PSD95-PDZ3, Faure et al. 2022) from the raw FASTQ with the
authors' own DiMSum version and settings, wrapped in Nextflow on AWS Batch.

| | Binding | Abundance |
|---|---|---|
| Variants shared with the deposited table | 7,901 | 6,291 |
| **Pearson** | **1.0000** | **1.0000** |
| Median absolute difference | 9.1e-05 | 1.9e-04 |

Slope 0.9998, intercept −0.0001. The variants only one side has are the noisy ones (median sigma 0.197
against 0.107 for the shared set).

## 14c · Two energies instead of one score

A thermodynamic readout layer: abundance assays see the folded fraction, activity assays see
folded-and-functional. The decomposition is only identifiable for proteins measured both ways, so the
readout is fitted on the nine such proteins.

**Held out: KRAS**, whose only relative HRAS is also on the locked test side, against the folding and six
binding free energies Weng et al. 2024 inferred in kcal/mol.

| Measured | Our folding energy | Our functional energy | ESM-2 (one score) |
|---|---|---|---|
| Folding ΔΔG | **0.369** | 0.245 | 0.366 |
| Binding, median of six partners | 0.202 | **0.312** | 0.362 |

The right energy wins all seven comparisons in 1000 of 1000 bootstrap resamples. A single ESM-2 score
gives essentially the same answer for folding and for binding; two energies tell them apart. We do not
beat ESM-2 on absolute binding accuracy.

**Energies add, fitness does not.** Single mutants alone fix each substitution's two energies; the
doubles are then predicted with nothing fitted to them:

| | Abundance R² | Binding R² |
|---|---|---|
| **Energies add** | **0.742** | **0.840** |
| Fitness adds | −0.041 | −0.127 |
| Worse of the two | 0.675 | 0.711 |

## 14d · How many measurements are worth it

128 assays of 92 locked test proteins treated as new projects. Median Spearman on the variants not
measured:

| Measured variants | 24 | 48 | 96 | 384 |
|---|---|---|---|---|
| **Zero-shot (measure nothing)** | **0.420** | **0.420** | 0.419 | 0.421 |
| One-hot + ridge | 0.175 | 0.272 | 0.396 | 0.579 |
| ESM-2 + ridge | 0.323 | 0.388 | 0.448 | 0.558 |
| ESM-2 + zero-shot + ridge | 0.347 | 0.414 | **0.479** | **0.595** |

Below about 50 measurements the free zero-shot score ranks better than anything fitted on them; one
96-well plate is where measuring starts to pay. One-hot encoding needs roughly four times as many.

**Design loop**, 96 per round, four rounds, median over assays:

| | Best variant found | Top 1% found |
|---|---|---|
| Random | 0.960 | 4 |
| Greedy | 0.998 | 6 |
| **Upper confidence bound** | **0.999** | **7** |

## 14e · What each position is for

The two energies averaged over a position's 19 substitutions: structural, functional, both or tolerant.
118 proteins, 103,744 positions, 1.97M substitutions.

**Against UniProt's annotated functional sites** (never shown to the model):

| Position is | Positions | Odds ratio | p |
|---|---|---|---|
| Both | 17,820 | **9.19** | ≈0 |
| Functional | 8,164 | **2.84** | ≈0 |
| Structural | 8,164 | 0.21 | 1.0 |
| Tolerant | 69,596 | 0.08 | 1.0 |

**Against pockets that marketed drugs bind** — where their residues rank inside their own protein by
functional cost (0 = most sensitive):

| Pocket | Residues | Pocket | Elsewhere | p |
|---|---|---|---|---|
| KRAS switch-II (sotorasib) | 17 | **0.238** | 0.532 | 0.0011 |
| ABL1 myristoyl (asciminib) | 18 | **0.232** | 0.506 | 0.0017 |
| SHP2 tunnel (SHP099) | 15 | 0.504 | 0.498 | 0.39 |

SHP2's tunnel is a conformational switch at a three-domain junction, which a folded-and-functional model
does not represent — a real boundary of the method.

The map is served: 103,744 rows in the S3 lake behind a Glue table and a `sites_current` view, a
`/v1/proteins/{protein}/sites` route on the API, and a `get_site_map` tool for the agent. KRAS's three
oncogenic hotspots G12, G13 and Q61 rank 11th, 26th and 44th of its 189 positions by functional cost.

## 14f · Which variants a pharmacological chaperone rescues

Migalastat is approved for Fabry disease; its makers published which GLA variants respond in their cell
assay and which do not. 555 labelled substitutions (254 rescued), GLA absent from every split and
unrelated to any training protein, cross-validation grouped by position.

| Score | AUROC | AUPRC |
|---|---|---|
| **Our functional energy alone** | **0.759** | 0.650 |
| ESM-2 log-odds | 0.729 | 0.646 |
| Conservation × burial | 0.715 | 0.597 |
| Our folding energy alone | 0.693 | 0.629 |

| Combined (grouped CV) | AUROC |
|---|---|
| **Two energies + burial** | **0.769** |
| Two energies | 0.764 |
| ESM-2 | 0.725 |

The direction on the folding energy was the opposite of the prior: the drug binds the active site, so it
rescues mildly destabilised variants whose machinery is intact, not severely broken ones.

## What is not claimed

- Absolute accuracy is not state of the art: 14a is behind SPURS, and 14c does not beat a single ESM-2
  score on binding. What the two energies add is **which kind of damage**, not a higher score.
- Nothing here is a clinical tool. 14f's labels are an in-vitro cell assay used as an ML benchmark.
- The decomposition needs proteins measured both ways; there are nine in this training set, which is the
  main thing limiting it.
