-- Every converted position must point at the substituted amino acid in the UniProt sequence:
-- otherwise structural features would later be attached to the wrong residue. Returns the offenders.
select m.dms_id, m.uniprot_id, m.uniprot_position, m.ref_aa, substring(p.sequence, m.uniprot_position, 1) as uniprot_aa
from {{ ref('fct_dms_measurement') }} m
join {{ ref('dim_protein') }} p on p.uniprot_id = m.uniprot_id
where m.uniprot_position is not null
    and substring(p.sequence, m.uniprot_position, 1) <> m.ref_aa
