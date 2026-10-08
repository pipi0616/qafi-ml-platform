-- One row per labelled missense variant (notebook 07c): its score rescaled between the two anchors of its
-- dataset, so that 1 = like the wild type and 0 = function lost, for every assay alike. Only datasets whose
-- anchors passed the checks of dim_label_dataset, and only variants whose residue matches UniProt.

with variants as (
    select urn as dataset_id, uniprot_id, uniprot_position, ref_aa, alt_aa, score
    from {{ ref('stg_mavedb_variants') }}
    where kind = 'missense' and ref_matches_uniprot
    union all
    select dms_id, uniprot_id, uniprot_position, ref_aa, alt_aa, dms_score
    from {{ ref('fct_dms_measurement') }}
    where uniprot_position is not null
)

select
    v.dataset_id,
    d.source,
    v.uniprot_id,
    v.uniprot_position,
    v.ref_aa,
    v.alt_aa,
    v.score,
    least(
        greatest((v.score - d.zero_score) / (d.one_score - d.zero_score), {{ var('label_min') }}),
        {{ var('label_max') }}
    ) as label,
    d.is_domainome
from variants v
join {{ ref('dim_label_dataset') }} d on d.dataset_id = v.dataset_id and d.status = 'used'
