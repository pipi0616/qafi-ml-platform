-- MaveDB human score sets kept for labels, with their protein and publication.
select
    urn,
    title,
    uniprot_id,
    gene_name,
    primary_pubmed,
    first_author,
    try_cast(publication_year as int) as publication_year,
    is_domainome
from {{ source('silver', 'mavedb_score_sets') }}
