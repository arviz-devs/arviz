import pymc3 as pm
from numpy.testing import assert_equal
from pandas.testing import assert_frame_equal
from ..utils import trace_to_dataframe, save_trace, load_trace


with pm.Model() as model:
    a = pm.Normal('a', 0, 1, shape=(2, 2))
    b = pm.Normal('b', 0, 1)
    trace = pm.sample(1000, chains=2)


def test_trace_to_dataframe():
    df_tc = trace_to_dataframe(trace, combined=True)
    df_fc = trace_to_dataframe(trace, combined=False)

    assert trace_to_dataframe(trace).shape == (2000, 5)
    assert trace_to_dataframe(trace, combined=False).shape == (1000, 10)

    for k, l in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert_equal(trace['a'][:, k, l][:1000], df_fc['a__{}_{}'.format(k, l)].iloc[:, 0].values)
        assert_equal(trace['a'][:, k, l][1000:], df_fc['a__{}_{}'.format(k, l)].iloc[:, 1].values)

        assert_equal(trace['a'][:, k, l], df_tc['a__{}_{}'.format(k, l)].values)

    assert_equal(trace['b'], df_tc['b'])
    assert_equal(trace['b'], df_tc['b'])

    assert_equal(trace['b'][:1000], df_fc['b'].iloc[:, 0])
    assert_equal(trace['b'][1000:], df_fc['b'].iloc[:, 1])


def test_save_and_load():
    save_trace(trace)
    trl0 = load_trace('trace.gzip')
    tr = trace_to_dataframe(trace, combined=False)
    save_trace(tr)
    trl1 = load_trace('trace.gzip')

    assert_frame_equal(tr, trl0)
    assert_frame_equal(tr, trl1)
    assert_frame_equal(trl0, trl1)
