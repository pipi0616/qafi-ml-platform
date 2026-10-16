Written for: someone evaluating this work who already has AlphaMissense, ESM-2 or SPURS available.

# Why not just use AlphaMissense, or SPURS?

Often you should. This page says when, and what this model gives that they do not.

The short version: **they answer a different question.** AlphaMissense answers "is this variant likely
pathogenic", SPURS and ThermoMPNN answer "how much does it destabilise the fold", ESM-2 answers "how
surprising is this substitution". Each produces **one number**. This model produces **two free energies**
— what the mutation costs the fold, and what it costs the function, separately, in kcal/mol — which is the
question you have when you are deciding what to do about the variant rather than how worried to be.

## The evidence, on a protein nobody's model had seen

KRAS makes the difference measurable. Weng et al. 2024 inferred, from 26,000 measurements, a folding free
energy change per variant **and** a binding free energy change to each of six partners. KRAS and its only
relative HRAS are both on this project's locked test side, so nothing here was trained on it.

Spearman against each measured quantity, signs aligned so higher is better (notebook 14c):

| Predictor | vs measured **folding** | vs measured **binding** (median of six) | Difference |
|---|---|---|---|
| **This model: folding energy** | **0.368** | 0.202 | **+0.167** |
| **This model: functional energy** | 0.245 | **0.310** | **−0.066** |
| AlphaMissense | 0.293 | 0.382 | −0.089 |
| CPT-1 | 0.336 | 0.340 | −0.004 |
| ESM-2 log-odds | 0.366 | 0.362 | +0.004 |

Read the last column. ESM-2 and CPT-1 give **the same answer to both questions** (difference ±0.004):
asked whether a variant breaks the fold or the binding, they cannot distinguish. AlphaMissense leans
slightly towards binding. This model's two numbers lean in **opposite** directions, 0.23 apart, and across
all seven measured energies the matching one wins — in 1000 of 1000 bootstrap resamples.

That is not a claim that the two energies are more accurate. **AlphaMissense is better than our functional
energy at binding** (0.382 against 0.310), and says so above. It is a claim about what the output can be
asked.

## Three things only energies give you

**1. Mechanism.** "Stability is fine, function is broken" is a different action from "it cannot fold". A
pharmacological chaperone or a corrector can rescue the second and will do nothing for the first. On the
published migalastat amenability list for GLA — an approved drug, a protein absent from every split — the
two energies reach AUROC 0.769 against 0.725 for a zero-shot language model (notebook 14f).

**2. Addition.** Free energies add; scores do not. Using single mutants alone to fix each substitution's
two energies, the thermodynamic model predicts ~5,000 held-out double mutants at R² 0.742 and 0.840. Adding
the measured single *fitness* instead gives a **negative** R², because fitness saturates (notebook 14c).
Any protein engineering project that stacks mutations needs the first behaviour.

**3. Units.** The output is in kcal/mol, so it can be compared with a calorimetry measurement, with an
inferred ddPCA energy, or with ThermoMPNN's output — rather than only ranked within one protein.

A fourth consequence, at the level of a position rather than a variant: averaging the two energies over a
position's 19 substitutions **splits** the proteome in a way one score cannot. Positions this model calls
functionally costly are enriched 2.8× for UniProt active and binding sites while structurally costly ones
are *depleted* to 0.21× (notebook 14e). A single score moves every class the same way.

## Where the published tools are better, and should be used

| If your question is | Use | Not this |
|---|---|---|
| Is this variant likely pathogenic in a patient? | **AlphaMissense**, EVE, CPT-1 | This model is behind on clinical pathogenicity: ClinVar AUROC 0.921 against AlphaMissense's 0.963 and CPT-1's 0.969 (notebook 09c) |
| How much does this mutation destabilise the fold, as accurately as possible? | **SPURS**, then ThermoMPNN | This model reaches median per-domain Spearman 0.76 on the same split against SPURS' 0.83 (notebook 14a) |
| I have no measurements and want a quick ranking | **ESM-2 zero-shot** — it is free and, below about 50 measured variants, better than anything fitted on them (notebook 14d) | |

## Closest related work

**DETANGO** (bioRxiv 2026) also separates stability from function with protein language models. Three
differences, from its own text: it trains **one model per protein** and cross-validates *within* a protein,
so it cannot predict a protein it has no data for; it replaces the Boltzmann relation with a learned
network, so its output is on a log-odds scale, not kcal/mol; and its supervision is ESM-1v's own log-odds
plus FoldX *predictions*, where this model is supervised by experimental measurements and the separation
is forced by the assay type. Its authors write that their separation "is operational rather than strictly
guaranteed".

**MoCHI / ddPCA** (Lehner lab) is where the thermodynamic model comes from, and does it better per protein
— but it *fits* energies to one protein's measurements and needs double mutants to do so. It cannot answer
for a protein with no data. This model predicts the same quantities from sequence and structure alone.

## The honest limitation

The decomposition is only identifiable from proteins measured **both** ways — abundance and activity on the
same library. There are nine such proteins in this training set. From a single assay type, any split of the
two energies that gives the same readout fits equally well; an L1 penalty collapses the functional energy
and a two-step fit barely helps, both measured (notebook 14c). More paired assays is the single thing that
would most improve this, which is why notebook 14b builds the pipeline that produces them.

## One more difference, which is not about the model

Most published predictors ship as a paper and a repository. This one ships with the system around it: data
ingested from publishers into S3 with pinned versions, a homology-aware test set written into a dbt seed
and opened once, every chapter re-checking its own new data against that split, MLflow records for every
number quoted here, 218M variants scored into a queryable lake, a serverless API, and raw sequencing reads
processed in Nextflow on AWS Batch. For an industrial setting that is often the part that decides whether a
model gets used.
