-- One row per protein of the UniProt human reference proteome.
select
    accession as uniprot_id,
    entry_name,
    gene_name,
    protein_name,
    reviewed,
    length,
    sequence
from {{ source('bronze', 'uniprot_proteome') }}
