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

**Double mutants** (added 2026-10-21). SPURS publishes a second benchmark on the same split: the 22,081
doubles of those 20 test domains. A single-substitution model enters by summing its two components, as
the `-additive` rows do. Spearman:

| Method | Spearman | |
|---|---|---|
| Summing *measured* singles | **0.744** | the ceiling for any additive method |
| SPURS (predicts doubles directly) | 0.59 | |
| MutateEverything | 0.54 | |
| SPURS-additive | 0.52 | |
| **Ours-additive** | **0.496** | 4th of 9 |
| ThermoMPNN-additive | 0.45 | |
| ESM2-additive | 0.40 | |
| DDGun-3D / DDGun-seq | 0.36 / 0.33 | |
| ProteinMPNN-additive | 0.16 | |

0.474 on the 19,682 genuine doubles (2,399 of the 22,081 carry a synonymous component such as
`T9Q:N23N` and are singles in disguise; SPURS counts them, so the headline figure keeps its convention).
Median per domain 0.724.

The gap to SPURS is smaller here than on singles — 0.024 against 0.066 — so it is single-mutant accuracy
that lags, not the summing. And every published model sits far below the 0.744 ceiling, which agrees
with chapter 15: most of the apparent non-additivity in this dataset is assay range compression, not
epistasis left on the table.

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

The right energy wins all seven comparisons. Resampling whole positions rather than mutations (the 19
substitutions at a position are not independent), six of the seven stay significant (p ≤ 0.016) and
DARPin K27 does not (p = 0.06). Two more held-out proteins measured both ways: abundance follows the
folding energy in both CYP2C9 and G6PD (100% of position resamples); activity beyond what abundance
explains follows the functional energy in G6PD (100%) but not in CYP2C9 (72%). A single ESM-2 score
gives essentially the same answer for folding and for binding; two energies tell them apart. We do not
beat ESM-2 on absolute binding accuracy.

**Energies add, fitness does not.** Single mutants alone fix each substitution's two energies; the
doubles are then predicted with nothing fitted to them:

| | Abundance R² | Binding R² |
|---|---|---|
| **Energies add** | **0.742** | **0.840** |
| Fitness adds | −0.041 | −0.127 |
| Worse of the two | 0.675 | 0.711 |

## 14d · What each position is for

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

**More checks, all on proteins the model never saw:**

| Check | Truth | Result |
|---|---|---|
| Four classes against measured ones | KRAS energies; CYP2C9 and G6PD abundance + activity beyond abundance | agreement 59% / 55% / 65% vs chance 42% / 40% / 38%, kappa 0.29 / 0.24 / 0.43, p < 1e-4 |
| Folding axis | abundance assays of 102 test proteins | abundance tracks the folding cost better than the functional cost in 75 of 102 proteins (p 1e-6); functional positions lose far less protein than structural ones (p 5e-30) |
| Functional axis | BioLiP ligand, peptide and nucleic-acid contacts (1,582 residues, 5x UniProt) | functional positions enriched 4.5x, structural 0.65x, tolerant 0.23x; same direction in 42 of 44 proteins (p 6e-11) |
| Drug pockets | BioLiP co-crystals of 21 approved drugs (28 listed in advance) | active-site pockets significant in 15 of 16, allosteric in 2 of 5 (4 of 8 with the three above); the functional cost ranks the pocket ahead of ESM-2 in 19 of 21 (p 1e-4) |

The classes are good for prioritising positions, not a verdict on each one: per class they recover two to
four in ten measured positions. Allosteric pockets at subunit interfaces (IDH1, IDH2) and in the membrane
(CFTR) join SHP2 as the method's blind spot.

The map is served: 103,744 rows in the S3 lake behind a Glue table and a `sites_current` view, a
`/v1/proteins/{protein}/sites` route on the API, and a `get_site_map` tool for the agent. KRAS's three
oncogenic hotspots G12, G13 and Q61 rank 11th, 26th and 44th of its 189 positions by functional cost.

## 14e · Which variants a pharmacological chaperone rescues

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

## Appendix 14x · How many measurements are worth it

Not on the main line: this chapter does not use the two energies. It asks a deployment question —
what the released model is worth to a lab that can only measure a few dozen variants.

128 assays of 92 locked test proteins treated as new projects, scored by the model 09b registered as
`@champion`. A few-shot model has to beat the prediction the project already has for free, so every table
is run twice: starting from the public ESM-2 zero-shot score, and starting from ours. Median Spearman on
the variants not measured:

| Measured variants | 24 | 48 | 96 | 192 | 288 | 384 |
|---|---|---|---|---|---|---|
| ESM-2 zero-shot (measure nothing) | 0.420 | 0.420 | 0.419 | 0.422 | 0.419 | 0.419 |
| **Our stability score** (measure nothing) | **0.588** | **0.588** | **0.588** | 0.589 | 0.588 | 0.587 |
| **Our function score** (measure nothing) | **0.576** | **0.576** | **0.576** | 0.577 | 0.576 | 0.577 |
| One-hot + ridge | 0.175 | 0.272 | 0.396 | 0.506 | 0.555 | 0.580 |
| ESM-2 + ridge | 0.323 | 0.388 | 0.448 | 0.512 | 0.542 | 0.559 |
| ESM-2 + zero-shot + ridge | 0.347 | 0.414 | 0.479 | 0.547 | 0.580 | 0.597 |
| ESM-2 + our stability score + ridge | 0.361 | 0.427 | 0.493 | 0.561 | 0.595 | 0.612 |
| **ESM-2 + our function score + ridge** | **0.361** | **0.429** | **0.495** | **0.564** | **0.597** | **0.614** |

Our function score beats the zero-shot one as a free starting line in 97% of assays (mean +0.156,
p = 2e-22); the stability score in 92% (+0.168). Both are reported because neither is picked per assay
type — choosing by assay kind would be choosing with knowledge of the answer.

**The released score is worth about 300 measurements.** The standard pipeline has to measure 288 variants
to reach 0.580, which is where our score starts for free. Accordingly the crossover moves: fitting beats
the zero-shot start after a median of 48 measured variants, but beats our score only after a median of
288, and in 37% of assays 384 measurements still do not beat it. As a ridge feature the same score adds
only 0.014–0.017 at every budget — almost all of its value is in the starting point, not the fitting.

**Design loop**, 96 per round, four rounds, 3 seeds, median over assays (mean in brackets):

| | Best variant found | Top 1% found |
|---|---|---|
| Random | 0.960 | 4 (6.40) |
| Greedy, from zero-shot | 0.998 | 6 (9.25) |
| UCB, from zero-shot | 0.999 | 7 (9.30) |
| **UCB, from our score** | **1.000** | **7 (9.51)** |

Paired per assay: guided picking finds +3.11 more of the top 1% than random (91% of assays, p = 3e-20),
but our prior beats the zero-shot prior by only +0.21 variants (p = 1e-4), and UCB does not beat greedy
(p = 0.65). Once the first 96 real measurements are in, the data outweighs the prior: **the model's value
is front-loaded**, which is visible only because our own model was compared against the public one.

These proteins were already opened once in 09a, so this is a second question asked of the same exam —
the numbers are estimates, not a fresh test.

## What is not claimed

- Absolute accuracy is not state of the art: 14a is behind SPURS, and 14c does not beat a single ESM-2
  score on binding. What the two energies add is **which kind of damage**, not a higher score.
- Nothing here is a clinical tool. 14e's labels are an in-vitro cell assay used as an ML benchmark.
- The decomposition needs proteins measured both ways; there are nine in this training set, which is the
  main thing limiting it.
