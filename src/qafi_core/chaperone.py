"""Can a small molecule rescue the variant? (notebook 14e)

Some mutations do not break what a protein does, they stop it from folding; the cell then destroys it
before it ever reaches the place it works. A **pharmacological chaperone** is a small molecule that binds
the protein while it folds and holds it in shape, so enough of it survives to do the job. It only works
when the protein's machinery is intact — exactly the case 14c calls "folding energy damaged, functional
energy fine".

Fabry disease is where this has been taken furthest: migalastat (Galafold) is approved, and before it was
approved its makers measured, one variant at a time in cells, which GLA mutations respond to it. Benjamin
et al. 2017 published the whole list, both the ones that respond and the ones that do not, which makes it
a rare labelled set for this question.

``amenability`` reads that list out of the supplement's two tables.
"""

from __future__ import annotations

import re

import pandas as pd

# "c.1234 A>G   c.A1234G   R123H": the last column is the amino-acid change we key on.
SUBSTITUTION = re.compile(r"\b([A-Z])(\d+)([A-Z])\b\s*$")
AMENABLE, NOT_AMENABLE = "amenable", "not amenable"


def amenability(text: str) -> pd.DataFrame:
    """Which GLA substitutions respond to migalastat in the cell assay, from the supplement's text.

    ``text`` is the whole supplement as plain text (``pdftotext -layout``). Table 11SA lists the variants
    that respond and 11SB the ones that do not; everything between the two headings belongs to the first.
    Rows that are not single amino-acid substitutions (truncations, frameshifts, splice changes) have no
    protein change of this shape and are skipped, as are stop codons (written X).

    Both headings repeat on every page of their table and once more in the contents, so the split is made
    at the first heading that carries the dose note, which the contents entry does not.

    Returns one row per substitution: position, ref_aa, alt_aa, variant, responds (bool).
    """
    start_a = text.index("Amenable GLA Mutations (")
    start_b = text.index("Non-amenable GLA Mutations (")
    if not start_a < start_b:
        raise ValueError("the amenable table should come before the non-amenable one")
    rows = []
    for responds, part in ((True, text[start_a:start_b]), (False, text[start_b:])):
        for line in part.splitlines():
            found = SUBSTITUTION.search(line.strip())
            if not found:
                continue
            ref, position, alt = found.group(1), int(found.group(2)), found.group(3)
            if "X" in (ref, alt):  # a stop codon, not a substitution of one amino acid for another
                continue
            rows.append(
                {
                    "position": position,
                    "ref_aa": ref,
                    "alt_aa": alt,
                    "variant": f"{ref}{position}{alt}",
                    "responds": responds,
                }
            )
    out = pd.DataFrame(rows)
    # a variant in both tables would be a contradiction; keep the amenable row and record which
    listed_twice = sorted(set(out[out.duplicated("variant", keep=False)].variant))
    out = out.drop_duplicates("variant", keep="first").reset_index(drop=True)
    out.attrs["listed_twice"] = listed_twice
    return out
