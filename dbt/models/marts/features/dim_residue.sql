-- One row per residue of the proteins with labels (the split): everything the features need to know about
-- a position, whatever the substitution (notebook 07f). Language-model values exist for every residue;
-- structure values only where the AlphaFold model has the same amino acid as UniProt. The SQL is the macro
-- residue_features, shared with dim_residue_proteome, which covers every human protein (notebook 09c).

{{ config(liquid_clustered_by='uniprot_id') }}

{{ residue_features("select p.uniprot_id, p.length from " ~ ref('dim_protein') ~ " p where p.uniprot_id in (select uniprot_id from " ~ ref('protein_split') ~ ")") }}
