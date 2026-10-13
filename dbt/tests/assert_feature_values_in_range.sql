-- Values a feature cannot take: an entropy outside [0, ln 20], a wild-type residue that is not the label's
-- reference residue, or a positive log-probability. Returns the offending residues.
select r.uniprot_id, r.position, r.esm_entropy, r.esm_wt_logprob
from {{ ref('dim_residue') }} r
where r.esm_entropy < 0 or r.esm_entropy > ln(20) + 1e-4 or r.esm_wt_logprob > 0
union all
select f.uniprot_id, f.uniprot_position, null, null
from {{ ref('fct_variant_features') }} f
join {{ ref('dim_residue') }} r on r.uniprot_id = f.uniprot_id and r.position = f.uniprot_position
where r.aa <> f.ref_aa
