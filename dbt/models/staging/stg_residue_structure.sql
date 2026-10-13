-- AlphaFold structure features per residue (notebook 07d): the columns the features use, names unchanged.

select
    uniprot_id,
    position,
    aa,
    plddt,
    rsa,
    contacts_8,
    contacts_12,
    hse_up,
    hse_down,
    neighbours
from {{ source('silver', 'residue_structure') }}
