-- One row per human DMS assay with a UniProt accession: what was measured, and how much of it maps
-- onto UniProt numbering.

with measured as (
    select
        dms_id,
        min(uniprot_id) as uniprot_id,
        count(*) as measurements,
        avg(case when position_mapping = 'unmapped' then 0 else 1 end) as share_mapped
    from {{ ref('fct_dms_measurement') }}
    group by dms_id
)

select
    a.dms_id,
    m.uniprot_id,
    a.uniprot_entry_name,
    a.selection_type,
    a.first_author,
    a.year,
    a.assay_sequence_length,
    m.measurements,
    round(m.share_mapped, 4) as share_mapped
from {{ ref('stg_proteingym_assays') }} a
join measured m on m.dms_id = a.dms_id
