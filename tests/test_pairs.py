from jspace.prompts import all_pairs, pairs_by_category

EXPECTED_CATEGORIES = {
    "role_reversal",
    "relation_binding",
    "negation",
    "causal_flip",
    "polysemy",
    "safety_latent",
}


def test_all_categories_present():
    cats = {p.category for p in all_pairs()}
    assert cats == EXPECTED_CATEGORIES


def test_pair_ids_unique():
    ids = [p.pair_id for p in all_pairs()]
    assert len(ids) == len(set(ids))


def test_pairs_are_minimal_pairs():
    for p in all_pairs():
        assert p.prompt_a != p.prompt_b, p.pair_id
        assert p.expected_a != p.expected_b, p.pair_id
        assert p.probe, p.pair_id


def test_full_prompts_include_probe():
    p = all_pairs()[0]
    assert p.probe in p.full_a() and p.probe in p.full_b()
    assert p.full_a() != p.full_b()


def test_pairs_by_category_filters():
    poly = pairs_by_category("polysemy")
    assert poly and all(p.category == "polysemy" for p in poly)


def test_starter_bank_size():
    # starter banks should stay non-trivial; growth tracked in PLAN.md phase 2
    assert len(all_pairs()) >= 40
