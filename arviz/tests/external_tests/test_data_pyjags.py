# pylint: disable=no-member, invalid-name, redefined-outer-name, unused-import
import typing as tp
import numpy as np
import pytest
import pyjags

from arviz import from_pyjags, InferenceData, waic

from arviz.data.io_pyjags import (
    _convert_pyjags_dict_to_arviz_dict,
    _convert_arviz_dict_to_pyjags_dict,
    _extract_arviz_dict_from_inference_data,
)

from arviz.tests.helpers import check_multiple_attrs, eight_schools_params


PYJAGS_POSTERIOR_DICT = {
    "b": np.random.randn(3, 10, 3),
    "int": np.random.randn(1, 10, 3),
    "log_like": np.random.randn(1, 10, 3),
}
PYJAGS_PRIOR_DICT = {"b": np.random.randn(3, 10, 3), "int": np.random.randn(1, 10, 3)}


PARAMETERS = ("mu", "tau", "theta_tilde")
VARIABLES = tuple(list(PARAMETERS) + ["log_like"])

NUMBER_OF_WARMUP_SAMPLES = 1000
NUMBER_OF_POST_WARMUP_SAMPLES = 5000


def verify_equality_of_numpy_values_dictionaries(
    dict_1: tp.Mapping[tp.Any, np.ndarray], dict_2: tp.Mapping[tp.Any, np.ndarray]
) -> bool:
    if dict_1.keys() != dict_2.keys():
        return False

    for key in dict_1.keys():
        if not np.all(dict_1[key] == dict_2[key]):
            return False

    return True


class TestDataPyJAGSWithoutEstimation:
    def test_convert_pyjags_samples_dictionary_to_arviz_samples_dictionary(self):
        arviz_samples_dict_from_pyjags_samples_dict = _convert_pyjags_dict_to_arviz_dict(
            PYJAGS_POSTERIOR_DICT
        )

        pyjags_dict_from_arviz_dict_from_pyjags_dict = _convert_arviz_dict_to_pyjags_dict(
            arviz_samples_dict_from_pyjags_samples_dict
        )

        assert verify_equality_of_numpy_values_dictionaries(
            PYJAGS_POSTERIOR_DICT, pyjags_dict_from_arviz_dict_from_pyjags_dict,
        )

    def test_extract_samples_dictionary_from_arviz_inference_data(self):
        arviz_samples_dict_from_pyjags_samples_dict = _convert_pyjags_dict_to_arviz_dict(
            PYJAGS_POSTERIOR_DICT
        )

        arviz_inference_data_from_pyjags_samples_dict = from_pyjags(PYJAGS_POSTERIOR_DICT)
        arviz_dict_from_idata_from_pyjags_dict = _extract_arviz_dict_from_inference_data(
            arviz_inference_data_from_pyjags_samples_dict
        )

        assert verify_equality_of_numpy_values_dictionaries(
            arviz_samples_dict_from_pyjags_samples_dict, arviz_dict_from_idata_from_pyjags_dict,
        )

    def test_roundtrip_from_pyjags_via_arviz_to_pyjags(self):
        arviz_inference_data_from_pyjags_samples_dict = from_pyjags(PYJAGS_POSTERIOR_DICT)
        arviz_dict_from_idata_from_pyjags_dict = _extract_arviz_dict_from_inference_data(
            arviz_inference_data_from_pyjags_samples_dict
        )

        pyjags_dict_from_arviz_idata = _convert_arviz_dict_to_pyjags_dict(
            arviz_dict_from_idata_from_pyjags_dict
        )

        assert verify_equality_of_numpy_values_dictionaries(
            PYJAGS_POSTERIOR_DICT, pyjags_dict_from_arviz_idata
        )

    @pytest.mark.parametrize("posterior", [None, PYJAGS_POSTERIOR_DICT])
    @pytest.mark.parametrize("prior", [None, PYJAGS_PRIOR_DICT])
    @pytest.mark.parametrize("save_warmup", [True, False])
    @pytest.mark.parametrize("warmup_iterations", [0, 5])
    def test_inference_data_attrs(self, posterior, prior, save_warmup, warmup_iterations: int):
        arviz_inference_data_from_pyjags_samples_dict = from_pyjags(
            posterior=posterior,
            prior=prior,
            log_likelihood={"y": "log_like"},
            save_warmup=save_warmup,
            warmup_iterations=warmup_iterations,
        )
        posterior_warmup_prefix = (
            "" if save_warmup and warmup_iterations > 0 and posterior is not None else "~"
        )
        prior_warmup_prefix = (
            "" if save_warmup and warmup_iterations > 0 and prior is not None else "~"
        )
        print(f'posterior_warmup_prefix="{posterior_warmup_prefix}"')
        test_dict = {
            f'{"~" if posterior is None else ""}posterior': ["b", "int"],
            f'{"~" if prior is None else ""}prior': ["b", "int"],
            f'{"~" if posterior is None else ""}log_likelihood': ["y"],
            f"{posterior_warmup_prefix}warmup_posterior": ["b", "int"],
            f"{prior_warmup_prefix}warmup_prior": ["b", "int"],
            f"{posterior_warmup_prefix}warmup_log_likelihood": ["y"],
        }

        fails = check_multiple_attrs(test_dict, arviz_inference_data_from_pyjags_samples_dict)
        assert not fails


# class TestDataPyJAGSWithEstimation:
#     @pytest.fixture(scope="class")
#     def jags_prior_model(self) -> pyjags.Model:
#         EIGHT_SCHOOL_PRIOR_MODEL_CODE = """
#         model {
#             mu ~ dnorm(0.0, 1.0/25)
#             tau ~ dt(0.0, 1.0/25, 1.0) T(0, )
#             for (j in 1:J) {
#                 theta_tilde[j] ~ dnorm(0.0, 1.0)
#             }
#         }
#         """
#
#         prior_model = pyjags.Model(
#             code=EIGHT_SCHOOL_PRIOR_MODEL_CODE,
#             data={"J": 8},
#             chains=4,
#             threads=4,
#             chains_per_thread=1,
#         )
#
#         return prior_model
#
#     @pytest.fixture(scope="class")
#     def jags_posterior_model(
#         self, eight_schools_params: tp.Dict[str, tp.Union[int, np.ndarray]]
#     ) -> pyjags.Model:
#         EIGHT_SCHOOL_POSTERIOR_MODEL_CODE = """
#         model {
#             mu ~ dnorm(0.0, 1.0/25)
#             tau ~ dt(0.0, 1.0/25, 1.0) T(0, )
#             for (j in 1:J) {
#                 theta_tilde[j] ~ dnorm(0.0, 1.0)
#                 y[j] ~ dnorm(mu + tau * theta_tilde[j], 1.0/(sigma[j]^2))
#                 log_like[j] = logdensity.norm(y[j], mu + tau * theta_tilde[j], 1.0/(sigma[j]^2))
#             }
#         }
#         """
#
#         posterior_model = pyjags.Model(
#             code=EIGHT_SCHOOL_POSTERIOR_MODEL_CODE,
#             data=eight_schools_params,
#             chains=4,
#             threads=4,
#             chains_per_thread=1,
#         )
#
#         return posterior_model
#
#     @pytest.fixture(scope="class")
#     def jags_prior_samples(self, jags_prior_model: pyjags.Model) -> tp.Dict[str, np.ndarray]:
#         return jags_prior_model.sample(
#             NUMBER_OF_WARMUP_SAMPLES + NUMBER_OF_POST_WARMUP_SAMPLES, vars=PARAMETERS
#         )
#
#     @pytest.fixture(scope="class")
#     def jags_posterior_samples(
#         self, jags_posterior_model: pyjags.Model
#     ) -> tp.Dict[str, np.ndarray]:
#         return jags_posterior_model.sample(
#             NUMBER_OF_WARMUP_SAMPLES + NUMBER_OF_POST_WARMUP_SAMPLES, vars=VARIABLES
#         )
#
#     @pytest.fixture()
#     def pyjags_data(
#         self,
#         jags_prior_samples: tp.Dict[str, np.ndarray],
#         jags_posterior_samples: tp.Dict[str, np.ndarray],
#     ) -> InferenceData:
#         return from_pyjags(
#             posterior=jags_posterior_samples,
#             prior=jags_prior_samples,
#             log_likelihood={"y": "log_like"},
#             save_warmup=True,
#             warmup_iterations=NUMBER_OF_WARMUP_SAMPLES,
#         )
#
#     def test_waic(self, pyjags_data):
#         waic_result = waic(pyjags_data)
#
#         assert -31.0 < waic_result.waic < -30.0
#         assert 0.75 < waic_result.p_waic < 0.90
#
#     # @pytest.fixture(scope="class")
#     # def data(self, jags_posterior_model, jags_posterior_samples, jags_prior_samples):
#     #     class Data:
#     #         model = self.jags_posterior_model
#     #         posterior = self.jags_posterior_samples
#     #         prior = self.jags_prior_samples
#     #
#     #     return Data
#     #
#     # def get_inference_data(self, data) -> InferenceData:
#     #     return from_pyjags(
#     #         posterior=data.posterior,
#     #         prior=data.prior,
#     #         log_likelihood={"y": "log_like"},
#     #         save_warmup=True,
#     #         warmup_iterations=NUMBER_OF_WARMUP_SAMPLES,
#     #     )
#     #
#     # def test_waic(self, data):
#     #     pyjags_data = self.get_inference_data(data)
#     #     waic_result = waic(pyjags_data)
#     #
#     #     assert -31.0 < waic_result.waic < -30.0
#     #     assert 0.75 < waic_result.p_waic < 0.90
