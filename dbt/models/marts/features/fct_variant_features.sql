-- The feature table (notebook 07f): one row per labelled variant per dataset, with its label, its side of
-- the split and every feature the models read. Features come from the residue (dim_residue) and from the
-- substitution (the seeds amino_acid and blosum62); none is derived from a label.

{{ config(liquid_clustered_by='uniprot_id') }}

select
    l.dataset_id,
    case when l.is_domainome then 'domainome' else l.source end as data_kind,
    l.uniprot_id,
    l.uniprot_position,
    l.ref_aa,
    l.alt_aa,
    s.split,
    s.similarity_group,
    l.label,
    -- substitution
    b.score as blosum62,
    alt.hydrophobicity - ref.hydrophobicity as d_hydrophobicity,
    alt.volume - ref.volume as d_volume,
    alt.charge - ref.charge as d_charge,
    -- language model
    {{ logprob_of('r.logprobs', 'l.alt_aa') }} - r.esm_wt_logprob as esm_delta,
    r.esm_wt_logprob,
    r.esm_entropy,
    -- structure
    r.plddt,
    r.rsa,
    r.contacts_8,
    r.contacts_12,
    r.hse_up,
    r.hse_down,
    -- neighbourhood
    r.neighbour_entropy,
    (alt.volume - ref.volume) * (1 - least(r.rsa, 1)) as d_volume_buried,
    r.relative_position
from {{ ref('fct_variant_label') }} l
join {{ ref('protein_split') }} s on s.uniprot_id = l.uniprot_id
join {{ ref('dim_residue') }} r on r.uniprot_id = l.uniprot_id and r.position = l.uniprot_position
join {{ ref('amino_acid') }} ref on ref.aa = l.ref_aa
join {{ ref('amino_acid') }} alt on alt.aa = l.alt_aa
join {{ ref('blosum62') }} b on b.ref_aa = l.ref_aa and b.alt_aa = l.alt_aa
