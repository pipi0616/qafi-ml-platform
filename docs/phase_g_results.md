# Phase G: generative design, in numbers

What the phase set out to do and what it actually established. Every number is produced by a notebook in
`notebooks/G_design/`. The test proteins were fixed in 07a and the exam was opened once; each chapter
says what it held out.

## The premise, and what it cost to check

Two assumptions the phase started from were wrong, and chapter 15 says so before anything else runs.

**The lake's two energies are the test set.** The 2.17e8 scores in the lake come from 09c's single-score
model. The two-energy rescoring covers 118 proteins, 108 of them the locked test proteins, so training a
generator on them would feed the exam to the model.

**The multi-mutant pool is all doubles, and mostly assay artefact.** 159,051 multi-mutant rows on 153
domains survive the stability filter chain, every one of them a double. Summing measured singles ranks
them at 0.81 Spearman but scores R² −0.42, and the gap is a systematic +0.89 kcal/mol bias that grows
with predicted severity (r = −0.72) and is flat in sequence separation. Doubles and singles share the
same −5.5 floor. The non-additivity is the assay's range, not epistasis. Otwinowski 2018 reported the
same mechanism on GB1; what is added here is the quantification and the separation test.

At the stabilising end, 350 of 6,875 measured singles stabilise and the held-out side has 173 rows above
+0.5 kcal/mol, so the design benchmark the phase originally planned has no statistical power. That
retired one benchmark, not the phase.

## 15a · What runs, which structures, which assays

ProteinMPNN runs locally on the CPU: 4 sequences for a 48-residue domain in 0.25 s, sequence recovery
0.40–0.48 against its published ~0.50. ESM-IF does not — three compiled extensions with no macOS ARM
wheels for torch 2.14 — so its published ProteinGym 0.42 is cited, not reproduced.

Three rules, fixed before any scoring:

| | Rule | Where it comes from |
|---|---|---|
| A | The design target is each assay's measured span | domainome coverage is a median 1.00, contiguous |
| B | Median pLDDT ≥ 70 over that span | AlphaFold's own confident boundary; removes 2% of train |
| C | A whole-chain AlphaFold model must exist | the 8 missing proteins all exceed 2700 residues |

Result: **128 assays, 97 proteins, 199,173 variants.** Not appendix 14x's 128 — only 113 overlap.

## 15c · Which ranking picks the positions to protect

Sumida et al. froze by ligand distance, found 24 of 24 designs inactive, then layered on an MSA
conservation percentile tuned per protein. This asks which ranking actually finds the positions where
mutations destroy the phenotype.

**146 assays, 107 held-out proteins.** Median Spearman between a ranking and measured damage:

| Ranking | Median Spearman |
|---|---|
| Our folding energy | **0.588** |
| Our two energies summed | 0.581 |
| The generator alone (ProteinMPNN entropy) | 0.516 |
| Burial | 0.493 |
| Our functional energy | 0.454 |
| ESM-2 substitution cost | 0.453 |
| Conservation | 0.364 |

Against the ranking Sumida settled on, paired across assays: **0.454 to 0.364, better in 107 of 146,
Wilcoxon p = 6.3e-11.**

Split by assay type, the generator is second on abundance, which reads folding, and **last of seven**
once activity enters (0.427 on the MaveDB group).

### The functional axis, on KRAS

KRAS is the one held-out protein measured on both axes. 166 positions:

| Ranking | vs measured folding | vs measured binding |
|---|---|---|
| ESM-2 substitution cost | 0.359 | **0.563** |
| **Our functional energy** | 0.261 | **0.508** |
| Conservation | 0.214 | 0.498 |
| **Our folding energy** | **0.460** | 0.312 |
| Burial | 0.407 | 0.222 |
| **The generator** | 0.425 | **0.161** |

The generator is second on folding and last on binding. Our two energies take one axis each.

On the 34 positions that lose binding beyond what folding explains — the class Sumida's first round
missed — AUROC: ESM-2 0.785, **ours 0.781**, conservation 0.780, folding 0.614, **the generator 0.532
(p = 0.28)**, burial 0.530 (p = 0.29).

**Stated against our own interest:** on KRAS we tie conservation and ESM-2 rather than beating them. The
advantage rests on the 146-assay paired test, not on KRAS. And our model reads ESM-2 features, so this is
conservation against conservation plus structure plus supervision.

## 15b · What the two energies add to the generator on top of its own pretraining

The generator saw these proteins' structures in its pretraining and our split cannot undo that, so the
only honest answer is to measure what we add. A substitution's generator score is log P(mutant) − log
P(wild type); the blend is a rank average inside each assay, so nothing is fitted on the measurements.

**132 assays, 99 proteins, 215,471 variants.** Median Spearman:

| | |
|---|---|
| Our folding energy alone | **0.621** |
| Generator + our two energies | 0.612 |
| Our two energies alone | 0.609 |
| Generator + ESM-2 | 0.527 |
| **The generator alone** | **0.496** |
| Our functional energy alone | 0.495 |
| ESM-2 zero-shot | 0.415 |

Paired against the generator alone: **+ our energies 0.612, better in 130 of 132, p = 2.5e-23**; + ESM-2
0.527 in 91 of 132, p = 2.3e-08.

**Stated against our own interest:** our folding energy alone (0.621) beats the blend (0.612). For
ranking variants that already exist the generator adds nothing we do not have. Ranking and designing are
different jobs — only the generator can write a sequence — but the blend is not sold as an improvement.

## 15d · The loop as something that can be re-run

`qafi_core.design` screens a campaign's intent against an allow-list, ranks positions, takes the budget,
writes ProteinMPNN's fixed-position file, and records the decision with a hash of the protected set. No
model is called inside it, so the same code runs in a notebook and in a Nextflow task.

End to end on a locked test protein: 560 positions, 280 fixed by the functional energy, four designs
changing 245–259 positions, **no protected position touched**. Three tasks, plan → generate → audit,
complete under the local profile.

The audit is not a formality: ProteinMPNN matches its fixed-position file on the pdb name, and when that
disagrees with the accession it silently redesigns everything while the output looks normal.

**Not done:** the Batch profile is the shape a deployment would take, not something that has run; there
is no `/design` route and no agent tool.

## What the phase establishes, and what it does not

**Establishes**

- Inverse folding alone cannot find the positions a function depends on: AUROC 0.532 at p = 0.28 on
  KRAS's functional positions, last of seven once an assay reads activity.
- Distance or burial cannot either (AUROC 0.530, p = 0.29) — the measured reason Sumida's first round
  failed.
- Our functional energy beats the conservation ranking that fixed it, across 146 assays, p = 6.3e-11.
- Adding our energies to the generator helps in 130 of 132 assays, far more than adding ESM-2.

**Does not establish**

- That a design made this way is active. That needs wet lab.
- That we beat conservation on KRAS specifically — there we tie.
- Anything about three or more simultaneous substitutions: the measured data is all doubles.
