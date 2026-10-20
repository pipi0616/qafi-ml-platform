# Talking about this project

The claims themselves, with their evidence, are in [`contributions.md`](contributions.md); this page is how to say them out loud.
The Chinese walkthrough of the claims, with worked examples and the order to say them in, is notebook
[`14f`](../notebooks/F_science/14f_contributions_and_pitch.ipynb).

Three audiences ask for different things from the same work. Each section below has a 30-second answer, a
2-minute answer, the questions that follow, and CV lines.

A rule that applies to all three: **say what the numbers are, including the ones that lose.** This project
is behind SPURS on stability and behind a zero-shot language model on absolute binding accuracy, and it
says so. Interviewers test for that, and the honest version is the stronger pitch, because what the work
actually buys is *which kind of damage* a mutation does — which no single score provides.

---

## 1 · Protein / biologics AI drug discovery

### 30 seconds

> I built a model that predicts what a mutation costs a protein as two physical quantities in kcal/mol:
> how much harder it is to fold, and how much worse it works once folded. On KRAS — held out together
> with its only relative — the folding energy tracks the measured folding free energy and the functional
> energy tracks binding to all six partners the experiment measured. A single language-model score gives
> essentially the same answer for both. And because they are energies, they add: single mutants alone
> predict about five thousand held-out double mutants at R² 0.74 and 0.84.

### 2 minutes

Start with the limitation of the field's standard output: one score per variant, which orders variants
but cannot say *why* one is bad. That distinction decides what you do next — a protein that misfolds
might be rescued by a chaperone or a corrector; a protein with a broken active site will not be.

The model predicts two free energies and a thermodynamic layer turns them into what each assay sees:
abundance assays see the folded fraction, activity assays see folded-and-functional. The separation is
imposed by the assay type, not suggested by the architecture.

The part worth telling is the identifiability problem. From an assay that sees only the product of the
two energies, any split giving the same readout fits equally well — the two energies came out 0.82
correlated, an L1 penalty collapsed one of them, and a MoCHI-style two-step fit barely helped. The
decomposition is only identifiable for proteins measured **both** ways; there are nine in the training
set, and fitting the readout on just those nine is what made it work.

Then the held-out test on KRAS, the additivity result on double mutants, and finally the practical
chapter: what the released model is worth to a project that has measured nothing. It replaces the free
zero-shot score as the starting line, and the standard few-shot pipeline needs about 300 measured
variants to catch up with it — three 96-well plates. A guided design loop then beats random picking by
7 of the top 1% against 4.

### Questions that follow

| Question | Answer |
|---|---|
| "How do you know the test set is clean?" | Proteins held out whole with their relatives, grouped by local alignment; the split is a dbt seed, opened once. Each new dataset is re-checked against it — 14a found 56 mega-scale domains homologous to locked test proteins and trained a decontaminated model, costing 0.010 Spearman. |
| "Do you beat the state of the art?" | On stability, no: median per-domain Spearman 0.76 against SPURS' 0.83 and ThermoMPNN's 0.77, with a generic architecture. On binding accuracy, no: 0.31 against ESM-2's 0.36, and AlphaMissense is better still at 0.38. What we have that they do not is two numbers that each track their own measurement — on the same KRAS data ESM-2 gives 0.366 for folding and 0.362 for binding, the same answer to both questions, while our two energies lean 0.23 apart in opposite directions. Full comparison in `docs/why_not_just_use_alphamissense.md`. |
| "Why not just fine-tune ESM?" | Did that in 08c: LoRA on a small ESM-2 gained +0.014 on stability and **lost** 0.014 on function, and a fine-tuned small model stayed behind a frozen 650M one. The bottleneck was functional training data, not model capacity. |
| "Would this work on my target?" | 14x answers it with a number. On 128 held-out assays the released score ranks variants at 0.58 Spearman before a single measurement, and the standard few-shot pipeline needs ~288 measured variants to match that — so on a protein nobody has measured the model is worth about three 96-well plates. Below ~50 measurements, fitting anything on them is worse than using a free score. For absolute accuracy expectations, 14c's KRAS result is the honest one. |
| "Where does the model stop paying?" | Also 14x, and it is the uncomfortable half. Once a design loop is running at 96 variants per round, swapping the public zero-shot prior for ours adds only 0.2 top-1% variants (p = 1e-4) against a 0.156 Spearman gap when nothing is measured. The data outweighs the prior after the first plate — the value is front-loaded, and I would not sell it as a per-round gain. |

### CV lines

- Built a protein stability and function model predicting two free energies (kcal/mol) with a
  thermodynamic readout; on a fully held-out protein each energy tracks its own measured quantity across
  seven experimental free energies (6 of 7 significant by a position-level bootstrap), and energies add to predict ~5,000 double mutants at
  R² 0.74–0.84.
- Quantified the measurement budget at which supervised fitting overtakes a zero-shot protein language
  model (~96 variants) and built an active-learning loop finding 75% more top-1% variants than random
  selection at equal budget.

---

## 2 · Target assessment / small-molecule discovery

### 30 seconds

> I turned a variant-effect model into a map of what each residue of a protein is *for* — structural
> scaffold, functional machinery, or changeable. Positions the model calls functionally costly are
> enriched 2.8-fold for UniProt active and binding sites, while structurally costly ones are depleted.
> As a prospective check I took three pockets that marketed drugs bind: the KRAS switch-II pocket that
> sotorasib binds and the ABL1 myristoyl pocket that asciminib binds both land in the top quarter of
> their own protein, with the model having seen no drug and no pocket.

### 2 minutes

The map comes from averaging the two energies over a position's 19 substitutions. Four classes, with
thresholds set per protein because the question is which positions stand out *within* a target.

Then two checks the model was never trained for. Against UniProt annotations the direction is right and —
this is the point — it *splits*: functional positions enriched 2.8×, both-energies positions 9.2×,
structural positions depleted to 0.21×. A single score cannot produce that, because it would move all
classes the same way.

The prospective check is the three drugged pockets. Two of three land in the top quarter at p ≈ 0.001.
The third, SHP2's tunnel, does not — and that is worth volunteering: it is a conformational switch at a
three-domain junction, not a site where the folded protein does its chemistry, so it falls outside what a
folded-and-functional model represents. Knowing the boundary of your method is part of the method.

The last chapter closes the loop to a real drug decision: migalastat, an approved pharmacological
chaperone for Fabry disease, whose makers published which GLA variants respond and which do not. On that
set, with the protein absent from every split, the two energies reach AUROC 0.769 against 0.725 for a
zero-shot language model. The direction surprised me — the drug binds the active site, so it rescues
mildly destabilised variants with intact machinery, not severely broken ones.

### Questions that follow

| Question | Answer |
|---|---|
| "Could I use this to find a new pocket?" | As a prior, not an answer. It ranks positions within a protein; two of three known pockets rank in the top quarter. It says nothing about druggability, cavity shape or ligand chemistry — pair it with a pocket detector. |
| "What about resistance mutations?" | Only half the question. The functional energy covers whether the protein still works; drug binding is not modelled. That would need the drug's own structure in the input. |
| "Why did SHP2 fail?" | Because its allosteric site maintains an autoinhibited conformation rather than doing chemistry. The model represents folded and functional states, not conformational switches. |

### CV lines

- Produced a proteome-scale residue map (104k positions) classifying each position as structural,
  functional or tolerant from two predicted free energies; functional positions are 2.8× enriched for
  annotated active and binding sites while structural ones are depleted.
- Showed prospectively that two of three allosteric pockets bound by marketed drugs (sotorasib, asciminib)
  rank in the top quarter of their protein by predicted functional cost.

---

## 3 · Bioinformatics scientist / computational biology

### 30 seconds

> I process deep mutational scanning data end to end: raw FASTQ to per-variant fitness in a Nextflow
> pipeline with pinned BioContainers, running on AWS Batch. As a correctness check I reprocessed a
> published experiment with the authors' own DiMSum version and settings and reconciled variant by
> variant against their deposited values: Pearson 1.0000 on both phenotypes, median absolute difference
> 1e-04. Those fitness values then feed a thermodynamic model that infers free energies.

### 2 minutes

Walk the pipeline: FastQC and MultiQC, cutadapt to remove the constant regions, VSEARCH to merge the read
pairs, starcode to count variants, then DiMSum's error model and fitness. Each step in a pinned
BioContainers image, parameters per experiment in a YAML file, a stub run in CI that tests the channel
wiring without data.

Then the QC judgement, which is what the job really is: for an amplicon library, per-base sequence
content, GC distribution, duplication and overrepresented sequences all fail FastQC by construction —
every read is the same gene. What matters is per-base quality, N content and adapters. Read 2 decays to
about Q27 at the end, which the pair merging and a strict max-expected-errors filter handle.

Then the operational story, because it is the part that shows experience. The pipeline died on the laptop
with an opaque R error; the kernel log showed the OOM killer taking DiMSum's workers. Profiling found two
memory peaks, one proportional to the FASTQ chunk, so chunk size and worker count came down — neither
changes a count. Then the disk filled. Rather than tune around a laptop I moved it to AWS Batch with a
profile switch, where the first submission sat in RUNNABLE with a healthy compute environment and no
error: CloudTrail showed a 5-vCPU account quota and Free-plan instance-type restrictions. Then the
official AWS CLI would not run inside the minimal containers for want of libz, so it went in as a conda
environment. Then the temporary credentials expired mid-run, so the SDK got a `credential_process`
profile that refreshes them, and `-resume` picked up from the cached steps.

### Questions that follow

| Question | Answer |
|---|---|
| "Nextflow or Snakemake?" | Nextflow here: it is what pharma runs, nf-core gives a convention to follow, and the executor is a profile switch from laptop to Batch. Snakemake is a reasonable choice in an academic group. |
| "How do you know the pipeline is right?" | Reconciliation against the authors' deposited values, variant by variant: Pearson 1.0000, median difference 1e-04, and I can account for the variants only one side has — ours are the noisier ones. |
| "What is missing from your experience?" | Single-cell and bulk RNA-seq. This project is amplicon and variant data; I have not run a differential-expression or a single-cell pipeline on real data. |
| "How would you scale it?" | It already runs on AWS Batch. Raising `max_vcpus` and the EC2 quota is the whole change; the pipeline code does not move. |

### CV lines

- Built a Nextflow DSL2 pipeline (FastQC, cutadapt, VSEARCH, starcode, DiMSum) in pinned BioContainers,
  running on an AWS Batch Spot compute environment defined in Terraform, with a stub run in CI.
- Reproduced a published deep mutational scanning analysis from raw FASTQ and reconciled it variant by
  variant with the authors' deposited fitness: Pearson 1.0000 over 14,000 shared variants.

---

## The one-paragraph version, for a cover letter

> I rebuilt my PhD's variant-effect predictor as a production platform and then took the science further
> than the thesis did. The platform ingests public data into S3 and a Databricks lakehouse, trains with a
> locked homology-aware test set, scores all 218 million human missense variants, and serves them through
> a lambda API with a Bedrock agent — all Terraform and CI. On top of it the model now predicts two
> physical free energies rather than one score, so it can say whether a mutation breaks a protein's fold
> or its function. On a protein held out together with its only relative, each energy tracks the matching
> experimental measurement across seven measured free energies; the energies add well enough to predict
> five thousand double mutants from single mutants alone; and on an approved drug's published response
> list the decomposition beats a zero-shot language model. I also process the upstream sequencing data
> myself, in Nextflow on AWS Batch, reproducing a published analysis from raw reads at Pearson 1.0000.
