-- MaveDB variants from silver (notebook 07c): missense, nonsense and synonymous, one row per protein variant.
select
    urn,
    uniprot_id,
    kind,
    assay_position,
    uniprot_position,
    position_mapping,
    ref_aa,
    alt_aa,
    score,
    measurements,
    ref_matches_uniprot
from {{ source('silver', 'mavedb_variants') }}
