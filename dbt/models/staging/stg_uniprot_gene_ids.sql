-- NCBI GeneID per UniProt accession; a few proteins map to more than one gene, kept as "id;id".
select
    accession as uniprot_id,
    array_join(array_sort(collect_set(id)), ';') as gene_id
from {{ source('bronze', 'uniprot_idmapping') }}
where id_type = 'GeneID'
group by accession
