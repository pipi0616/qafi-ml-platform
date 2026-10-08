-- Every protein with DMS measurements must be in the locked split; returns those that are not.
-- A new DMS protein (say, from a new ProteinGym release) fails this test until it is given a side.
select p.uniprot_id, p.gene_name
from {{ ref('dim_protein') }} p
left join {{ ref('protein_split') }} s on s.uniprot_id = p.uniprot_id
where p.has_dms and s.uniprot_id is null
