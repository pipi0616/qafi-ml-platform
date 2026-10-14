-- dim_residue for every protein of the UniProt human reference proteome, 20,652 proteins and 11.4 million
-- residues: what the released models read when they score the proteome (notebook 09c). Same macro, same
-- SQL as dim_residue, so the models see exactly the values they were trained on. Built only in 09c:
-- the training jobs leave it out (it takes minutes, not seconds).

{{ config(liquid_clustered_by='uniprot_id') }}

{{ residue_features("select p.uniprot_id, p.length from " ~ ref('dim_protein') ~ " p") }}
