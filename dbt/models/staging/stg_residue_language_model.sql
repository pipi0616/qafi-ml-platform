-- ESM-2 log-probabilities per residue (notebook 07e), with the position's entropy: how many amino acids the
-- model finds plausible there. 0 = only one fits (conserved), ln 20 = 3.0 = any fits (variable).

select
    uniprot_id,
    position,
    aa,
    logprobs,
    aggregate(logprobs, cast(0 as double), (total, x) -> total - exp(x) * x) as esm_entropy
from {{ source('silver', 'residue_language_model') }}
