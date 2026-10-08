-- Every labelled variant must sit on its reference residue in the UniProt sequence, or structural features
-- (notebook 07d) would describe another residue. Returns the offenders.
select l.dataset_id, l.uniprot_id, l.uniprot_position, l.ref_aa, substring(p.sequence, l.uniprot_position, 1) as uniprot_aa
from {{ ref('fct_variant_label') }} l
join {{ ref('dim_protein') }} p on p.uniprot_id = l.uniprot_id
where substring(p.sequence, l.uniprot_position, 1) <> l.ref_aa
