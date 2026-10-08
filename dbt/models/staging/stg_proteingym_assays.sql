-- One row per ProteinGym assay, with typed columns and readable names.
select
    DMS_id as dms_id,
    UniProt_ID as uniprot_entry_name,
    taxon,
    coarse_selection_type as selection_type,
    first_author,
    try_cast(year as int) as year,
    target_seq,
    try_cast(seq_len as int) as assay_sequence_length
from {{ source('bronze', 'proteingym_assays') }}
