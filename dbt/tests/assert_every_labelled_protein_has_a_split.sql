-- Every protein the model can learn from must be on one side of the locked split; returns those that are not.
-- Both sources count: DMS measurements (notebook 04) and variant labels (notebook 07c, MaveDB and Domainome
-- included). A newly labelled protein fails this test until qafi_core.validation.extend_split gives it a side.
with labelled as (
    select uniprot_id from {{ ref('dim_protein') }} where has_dms
    union
    select distinct uniprot_id from {{ ref('fct_variant_label') }}
)

select l.uniprot_id
from labelled l
left join {{ ref('protein_split') }} s on s.uniprot_id = l.uniprot_id
where s.uniprot_id is null
