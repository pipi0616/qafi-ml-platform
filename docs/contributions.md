Written for: an interviewer, hiring manager or collaborator asking "what is new here, and why would I use it".

# What this work contributes

Everything below is a claim with a number and a notebook behind it. Where a published tool is better, this
page says so in the same breath — that is the point of having the numbers.

> A Chinese walkthrough of this page, with runnable worked examples for the thermodynamic readout,
> the identifiability problem and additivity, is
> [`14g`](../notebooks/F_science/14g_contributions_and_pitch.ipynb).

| # | The claim | The evidence | Where |
|---|---|---|---|
| **1** | Predicts **two** free energies — folding and function — not one score, for a protein it has never seen | On held-out KRAS the right energy wins all 7 measured free energies, 1000/1000 bootstrap | [14c](../notebooks/F_science/14c_two_energies.ipynb) |
| **2** | A single score cannot do this, measurably | ESM-2 correlates 0.366 with folding and 0.362 with binding — the same answer; our two energies lean 0.23 apart in opposite directions | [14c](../notebooks/F_science/14c_two_energies.ipynb) |
| **3** | Found and solved *why* the decomposition usually fails | It is not identifiable from one assay type; fitting the readout on the 9 proteins measured both ways is what makes it work. Two alternative fixes measured and rejected | [14c](../notebooks/F_science/14c_two_energies.ipynb) |
| **4** | Energies add, so single mutants predict combinations | ~5,000 held-out double mutants at R² 0.742 / 0.840; adding measured fitness instead gives a **negative** R² | [14c](../notebooks/F_science/14c_two_energies.ipynb) |
| **5** | Output in kcal/mol, comparable with experiment | Median per-domain Spearman 0.76 and RMSE 0.74 kcal/mol on the published mega-scale split | [14a](../notebooks/F_science/14a_stability_kcal.ipynb) |
| **6** | The decomposition splits the proteome the way biology does | Functional positions 2.8× enriched for UniProt active/binding sites; structural ones *depleted* to 0.21× | [14e](../notebooks/F_science/14e_site_map.ipynb) |
| **7** | It points at pockets real drugs bind, prospectively | KRAS switch-II (sotorasib) and ABL1 myristoyl (asciminib) in the top quarter of their own protein, p ≈ 0.001, no drug ever shown to the model | [14e](../notebooks/F_science/14e_site_map.ipynb) |
| **8** | The mechanism it reports is useful for a real drug decision | On migalastat's published amenability list: AUROC 0.769 against 0.725 for a zero-shot language model | [14f](../notebooks/F_science/14f_chaperone_rescue.ipynb) |
| **9** | Answers "how many variants must I measure" with a number | Below ~50 the free zero-shot score wins; one 96-well plate is the crossover; one-hot needs 4× as many | [14d](../notebooks/F_science/14d_low_n_and_design.ipynb) |
| **10** | A guided design loop beats random picking | 7 of the top 1% against 4, at equal budget | [14d](../notebooks/F_science/14d_low_n_and_design.ipynb) |
| **11** | Processes the upstream data too, exactly | A published experiment reprocessed from raw FASTQ: **Pearson 1.0000**, median difference 1e-04 | [14b](../notebooks/F_science/14b_reads_to_fitness.ipynb) |
| **12** | Validation that would survive an audit | Whole proteins held out with their relatives, split locked in a dbt seed, opened once, every chapter re-checked against it | [07a](../notebooks/C_ml/07a_feature_design.ipynb), [14a](../notebooks/F_science/14a_stability_kcal.ipynb) |
| **13** | Shipped, not just trained | 218M variants scored into a queryable lake, a serverless API, a guarded agent, 88 Terraform resources, 263 tests | [09c](../notebooks/C_ml/09c_proteome_scoring.ipynb), [10](../notebooks/D_serving_ai/10a_lake_athena.ipynb), [13](../notebooks/E_wrap_up/13_operations_and_story.ipynb) |

---

## A · What is new in the method

### The separation is forced by the assay, not suggested by the architecture

A mutation changes two free energies. Experiments never see them directly — they see a phenotype, and
which phenotype decides which energy is visible:

```
abundance assays (VAMP-seq, Domainome, AbundancePCA)   see the folded fraction      ← only ΔΔG_fold
activity and binding assays                            see folded AND functional    ← both energies
```

So an abundance assay **pins** the folding energy, and whatever an activity assay shows beyond that can
only be the functional energy. The functional energy is free of stability by construction, not because a
network was asked nicely.

This is the thermodynamic model of ddPCA and MoCHI (Lehner lab). What is new is the direction: **they fit
these energies to one protein's measurements; this predicts them for a protein with no measurements at
all.** MoCHI needs double mutants of the protein in question; this needs only its sequence and structure.

### The identifiability problem, found and solved

This is the part worth telling, because it is the reason the obvious version fails.

From an assay that sees only the *product* of the two energies, a low reading could be "did not fold" or
"folded but inactive" — any split giving the same readout fits equally well. Measured: the two predicted
energies came out **0.82 correlated**, i.e. not separated at all. Two standard fixes were tried and both
failed, with numbers:

| Attempted fix | Result |
|---|---|
| L1 penalty on the functional energy | It collapses to zero (Spearman 0.001); the folding energy absorbs everything, total score unchanged |
| Pin the folding energy so activity rows cannot move it (MoCHI's two-step fit, as one network) | Correlation 0.82 → 0.79. Barely moves |

The decomposition is identifiable only for a protein measured **both** ways. There are 9 such proteins in
the training set, and fitting the readout layer on just those nine is what makes it work — at a small cost
in single-task accuracy (folding 0.430 → 0.368) and a large gain in separation.

**This also explains why [14b](../notebooks/F_science/14b_reads_to_fitness.ipynb) matters**: paired
abundance + binding measurements on the same library are exactly what ddPCA produces, and more of them is
the single thing that would most improve this model.

### Why two numbers beat one, in the one place it can be measured

KRAS is held out with its only relative HRAS, and Weng et al. 2024 published a folding free energy and six
binding free energies per variant. Spearman, signs aligned, higher is better:

| Predictor | vs folding | vs binding (median of six) | **Difference** |
|---|---|---|---|
| **Our folding energy** | **0.368** | 0.202 | **+0.167** |
| **Our functional energy** | 0.245 | **0.310** | **−0.066** |
| AlphaMissense | 0.293 | 0.382 | −0.089 |
| CPT-1 | 0.336 | 0.340 | −0.004 |
| ESM-2 log-odds | 0.366 | 0.362 | +0.004 |

ESM-2 and CPT-1 give the same answer to both questions. A single score has one value, so it can only have
one preference. Two energies can lean in opposite directions — and the matching one wins all seven
comparisons in 1000 of 1000 bootstrap resamples.

**AlphaMissense is better than our functional energy at binding** (0.382 vs 0.310). The claim is not
accuracy; it is that the output can be *asked which mechanism is broken*, which decides the next
experiment. Full treatment:
[`why_not_just_use_alphamissense.md`](why_not_just_use_alphamissense.md).

### Energies add; scores do not

Free energy is additive — that is thermodynamics, not an empirical regularity. Using single mutants alone
to fix each substitution's two energies, the model predicts ~5,000 held-out double mutants:

| | Abundance R² | Binding R² |
|---|---|---|
| **Energies add** | **0.742** | **0.840** |
| Fitness adds | −0.041 | −0.127 |
| Worse of the two | 0.675 | 0.711 |

Adding fitness is worse than predicting the mean, because fitness saturates. Any project that stacks
mutations — which is every protein engineering project — needs the first row.

---

## B · What is new in how it is validated

This is the part that is hardest to fake and easiest to check, and in an industrial setting it is often
what decides whether a model gets used.

**Whole proteins held out, with their relatives.** Variant-level splits leak: the same protein on both
sides is memorisation. Similar sequences are grouped by local alignment (E-value, identity and coverage
thresholds, as sequence-clustering tools use) and whole groups go to one side.

**The test set was chosen once, written into a dbt seed, and opened once** (2026-10-14, recorded in
MLflow). A test set looked at twice stops being a test.

**Every chapter re-checks its own new data against that split.** [14a](../notebooks/F_science/14a_stability_kcal.ipynb)
found 56 mega-scale domains homologous to locked test proteins, trained a second decontaminated model for
everything downstream, and **states the cost** (0.010 Spearman). Most papers would not have looked.

**Published comparisons use the publishers' own split.** 14a reuses ThermoMPNN's `mega_splits.pkl`, so our
number sits beside theirs — and SPURS' and StaB-ddG's — without rerunning anything. Inventing a split would
have thrown all three comparisons away.

**Reconciliation before use.** 14a reproduced the published filtering chain row by row
(776,298 → 607,839 → 391,090 → 272,712) and traced a 140-row discrepancy in the paper's stated test size
to wild-type control rows. 14b reconciled a reprocessed experiment variant by variant at **Pearson
1.0000**.

**Negative results are kept, not buried.** The L1 penalty and the pinning above; 08b's graph network
(neighbours hurt once ESM embeddings are present); 08c's LoRA fine-tuning (+0.014 stability, −0.014
function); and SHP2, the one drugged pocket the site map does not find, with the reason.

---

## C · What is new in how it is built

The science above sits on a platform, and the platform is a deliberate part of the work.

| | What it means here |
|---|---|
| **Raw data to served answer, end to end** | Sequencing reads → fitness (Nextflow/DiMSum on AWS Batch) → energies → 218M variants scored → S3 lake → Athena → API → agent |
| **Everything versioned** | Inputs pinned by release and fetched from the publishers into S3 by a Lambda that validates and writes a receipt; models in a Unity Catalog registry gated on the locked test set; MLflow for every number quoted anywhere in these docs |
| **Portable compute** | The reads-to-fitness pipeline runs on a laptop with Docker or on AWS Batch by changing a profile, not a rewrite; containers pinned to the versions the original authors used |
| **Training-serving parity** | One SQL macro computes residue features for both training and serving, with a parity test — the most common silent failure in deployed ML |
| **Tested** | 263 unit tests, including golden tests pinning the published paper's numbers, and a CI stub run of the Nextflow pipeline that checks the wiring without data |
| **Operated** | 88 Terraform resources, OIDC instead of stored keys, CloudWatch alarms, cost under a few dollars a month, serving pay-per-request |

A useful detail for an interview: when the same registry was rate-limited and broke CI on 3 of 6 pushes,
the fix was structural (cache the image, keyed on its version, with the Dockerfile taking it as a build
argument) and verified by showing both build paths produce the **identical image digest** — not a retry
loop.

---

## D · What it lets someone do that they could not before

| Question a project actually asks | What this gives |
|---|---|
| "Is this variant bad?" | Use AlphaMissense. This is not the right tool, and says so |
| "**Why** is it bad — can a chaperone or corrector rescue it?" | Two energies; validated on an approved drug's own response list (AUROC 0.769 vs 0.725) |
| "If I stack these two mutations, what do I get?" | Add the energies: R² 0.74 / 0.84 on held-out doubles |
| "Which residues of my target can I change, and which are the machinery?" | A site map, served through SQL, an API route and an agent tool; 2.8× enriched for annotated functional sites |
| "Where might an allosteric pocket be?" | A prior, not an answer: two of three drugged pockets rank in the top quarter |
| "How many variants must I measure before a model helps?" | About 96. Below 50, do not bother — the free zero-shot score is better |
| "Which 96 should I measure next?" | A design loop that finds 7 of the top 1% against random's 4 |

---

## E · What is **not** claimed

Stated here as plainly as the claims, because an interviewer will look for it.

- **Not the most accurate stability predictor.** Median per-domain Spearman 0.76 against SPURS' 0.83 and
  ThermoMPNN's 0.77, using a generic architecture with no stability-specific design.
- **Not the most accurate anything, on a single measurement.** AlphaMissense beats the functional energy at
  binding (0.382 vs 0.310) and beats this model on clinical pathogenicity (ClinVar AUROC 0.963 vs 0.921).
- **Not a clinical tool.** 14f's labels are an in-vitro cell assay used as an ML benchmark; nothing here is
  a basis for a decision about a person.
- **Limited by paired data.** Nine proteins measured both ways is the binding constraint on the
  decomposition, and the honest answer to "how would you improve this".
- **The site map covers 118 proteins**, not the whole proteome; extending it is a list change and ~6 hours
  of compute, not new work.
- **Conformational switches are outside the model.** SHP2's allosteric tunnel maintains an autoinhibited
  state rather than doing chemistry, and the site map does not find it (p = 0.39).

---

## Where to find the detail

| You want | Read |
|---|---|
| Every number, one page | [`phase_f_results.md`](phase_f_results.md) |
| Why not just use AlphaMissense or SPURS | [`why_not_just_use_alphamissense.md`](why_not_just_use_alphamissense.md) |
| How to say all this out loud, per role | [`interview_pack.md`](interview_pack.md) |
| This page in Chinese, with runnable examples | [`14g`](../notebooks/F_science/14g_contributions_and_pitch.ipynb) |
| The published work this builds on | [`related_work.md`](related_work.md) |
| What was built, step by step, and why | [`plan.md`](plan.md) and `notebooks/` |
| The system around the model | [`../README.md`](../README.md) |
