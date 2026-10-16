import pytest

from qafi_core import chaperone

SUPPLEMENT = """
Table of contents
Table 11SA:                                                      68
The Pharmacogenetic Reference Table ...
Amenable GLA Mutations
Table 11SB:                                                      77
Non-amenable GLA Mutations

Table 11SA:
           Amenable GLA Mutations (assumes 150 mg dose of migalastat HCl)
       Nucleotide Change             Nucleotide Change      Protein Sequence Change
              c.8 T>C                      c.T8C                      L3P
           c.37 G>A                       c.G37A                      A13T
          c.1288 T>A                     c.T1288A                    X430Q

           Amenable GLA Mutations (assumes 150 mg dose of migalastat HCl)
           c.43 G>A                       c.G43A                      A15T
         c.1123 C>T                      c.C1123T                     R375C

Table 11SB:
        Non-amenable GLA Mutations (assume 150 mg dose of migalastat HCl)
                    Nucleotide Change                     Protein Sequence Change
         c.1 A>C                         c.A1C                     M1L
         c.2 T>G                         c.T2G                     M1R
         c.260 del                       c.260delA                  frameshift
        Non-amenable GLA Mutations (assume 150 mg dose of migalastat HCl)
         c.1123 C>A                      c.C1123A                    R375H
"""


def test_the_two_tables_become_labelled_substitutions():
    out = chaperone.amenability(SUPPLEMENT)

    assert dict(zip(out.variant, out.responds, strict=True)) == {
        "L3P": True,
        "A13T": True,
        "A15T": True,
        "R375C": True,
        "M1L": False,
        "M1R": False,
        "R375H": False,
    }
    assert list(out.columns) == ["position", "ref_aa", "alt_aa", "variant", "responds"]
    first = out.iloc[0]
    assert (first.position, first.ref_aa, first.alt_aa) == (3, "L", "P")


def test_rows_that_are_not_one_substitution_are_skipped():
    out = chaperone.amenability(SUPPLEMENT)

    assert "X430Q" not in set(out.variant)  # a stop codon, not an amino-acid substitution
    assert not any("del" in v for v in out.variant)  # the frameshift line has no protein change to read


def test_the_heading_in_the_contents_does_not_split_the_tables():
    # the contents lists both headings before either table; splitting there would label everything
    out = chaperone.amenability(SUPPLEMENT)

    assert out.responds.sum() == 4
    assert (~out.responds).sum() == 3


def test_a_variant_in_both_tables_is_recorded():
    both = SUPPLEMENT.replace("                    R375H", "                    R375C")

    out = chaperone.amenability(both)

    assert out.attrs["listed_twice"] == ["R375C"]
    assert out[out.variant == "R375C"].responds.item() is True  # the amenable row is kept
    assert len(out[out.variant == "R375C"]) == 1


def test_tables_in_the_wrong_order_are_refused():
    a = SUPPLEMENT.index("Amenable GLA Mutations (")
    b = SUPPLEMENT.index("Non-amenable GLA Mutations (")
    swapped = SUPPLEMENT[:a] + SUPPLEMENT[b:] + SUPPLEMENT[a:b]  # the non-amenable table now comes first

    with pytest.raises(ValueError, match="should come before"):
        chaperone.amenability(swapped)
