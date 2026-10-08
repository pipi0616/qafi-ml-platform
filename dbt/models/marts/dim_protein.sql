-- One row per human protein (UniProt reference proteome): identifiers, sequence and DMS coverage.
-- The list of proteins the model will score proteome-wide, and the lookup the API will use.

with coverage as (
    select
        uniprot_id,
        count(distinct dms_id) as dms_assays,
        count(*) as dms_measurements
    from {{ ref('fct_dms_measurement') }}
    group by uniprot_id
)

select
    p.uniprot_id,
    p.entry_name,
    p.gene_name,
    g.gene_id,
    p.protein_name,
    p.reviewed,
    p.length,
    p.sequence,
    coalesce(c.dms_assays, 0) as dms_assays,
    coalesce(c.dms_measurements, 0) as dms_measurements,
    c.uniprot_id is not null as has_dms
from {{ ref('stg_uniprot_proteins') }} p
left join {{ ref('stg_uniprot_gene_ids') }} g on g.uniprot_id = p.uniprot_id
left join coverage c on c.uniprot_id = p.uniprot_id
