import re

import arviz as az


def test_info_attr():
    info_message = az.info
    assert isinstance(info_message, str)
    pat = re.compile(r"arviz_(base|stats|plots)[\s\.0-9abdevrc]+available")
    for line in info_message.splitlines()[2:]:
        assert pat.match(line)


def test_aliases():
    for obj_name in dir(az):
        if not obj_name.startswith("_") and obj_name != "info":
            obj = getattr(az, obj_name)
            if hasattr(obj, "__module__"):
                orig_lib = obj.__module__.split(".")[0]
            elif hasattr(obj, "__package__"):
                orig_lib = obj.__package__
            else:
                assert False, obj_name
            assert orig_lib.startswith("arviz"), obj_name
            assert orig_lib != "arviz", obj_name


def test_base_alias():
    import arviz_base

    assert az.base is arviz_base


def test_stats_alias():
    import arviz_stats

    assert az.stats is arviz_stats


def test_plots_alias():
    import arviz_plots

    assert az.plots is arviz_plots
