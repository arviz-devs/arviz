from numpy.testing import assert_equal
from pandas.testing import assert_frame_equal
import pymc3 as pm

from ..utils import trace_to_dataframe, save_trace, load_trace


class TestUtils():

    @classmethod
    def setup_class(cls):
        with pm.Model():
            pm.Normal('a', 0, 1, shape=(2, 2))
            pm.Normal('b', 0, 1)
            cls.trace = pm.sample(1000, chains=2)

    def test_trace_to_dataframe(self):
        df_tc = trace_to_dataframe(self.trace, combined=True)
        df_fc = trace_to_dataframe(self.trace, combined=False)

        assert trace_to_dataframe(self.trace).shape == (2000, 5)
        assert trace_to_dataframe(self.trace, combined=False).shape == (1000, 10)

        for j, k in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            assert_equal(self.trace['a'][:, j, k][:1000],
                         df_fc['a__{}_{}'.format(j, k)].iloc[:, 0].values)
            assert_equal(self.trace['a'][:, j, k][1000:],
                         df_fc['a__{}_{}'.format(j, k)].iloc[:, 1].values)

            assert_equal(self.trace['a'][:, j, k], df_tc['a__{}_{}'.format(j, k)].values)

        assert_equal(self.trace['b'], df_tc['b'])
        assert_equal(self.trace['b'], df_tc['b'])

        assert_equal(self.trace['b'][:1000], df_fc['b'].iloc[:, 0])
        assert_equal(self.trace['b'][1000:], df_fc['b'].iloc[:, 1])

    def test_save_and_load(self, tmpdir_factory):
        filename = tmpdir_factory.mktemp('traces').join('trace.gzip')
        save_trace(self.trace, filename=filename)
        trl0 = load_trace(filename)
        tr = trace_to_dataframe(self.trace, combined=False)
        save_trace(tr, filename=filename)
        trl1 = load_trace(filename)

        assert_frame_equal(tr, trl0)
        assert_frame_equal(tr, trl1)
        assert_frame_equal(trl0, trl1)
