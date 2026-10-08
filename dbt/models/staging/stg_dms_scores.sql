-- Human DMS measurements whose protein has a UniProt accession.
select
    dms_id,
    uniprot_id,
    position,
    ref_aa,
    alt_aa,
    dms_score,
    dms_score_bin
from {{ source('silver', 'dms_scores') }}
where taxon = 'Human'
    and uniprot_id is not null
