# pylint: disable=redefined-outer-name

import typing as tp
import numpy as np
import pytest

import arviz as az

from arviz.data.io_pyjags import (
    _convert_pyjags_samples_dictionary_to_arviz_samples_dictionary,
    _convert_arviz_samples_dictionary_to_pyjags_samples_dictionary,
    _extract_samples_dictionary_from_arviz_inference_data,
    from_pyjags
)


@pytest.fixture()
def pyjags_samples_dict() -> tp.Dict[str, np.ndarray]:
    return \
    {'b': np.array([[[0.86197455, 0.74254685, 0.98143397],
                     [1.83545503, 2.47641691, 1.02592144],
                     [1.57544457, 1.75598012, 1.74688833],
                     [1.58015715, 1.68615547, 1.5567067],
                     [1.31283497, 1.50969295, 1.51691764],
                     [1.14948955, 1.66328105, 1.24730512],
                     [1.17592415, 1.68095813, 1.41572386],
                     [1.68424726, 1.09456233, 0.99070935],
                     [1.0407399, 1.31174258, 1.19530819],
                     [1.79751128, 1.02484299, 1.71675953]],

                    [[-0.72944677, -1.68140793, -0.75300271],
                     [-1.21015253, -1.77640948, -0.7980358],
                     [-1.2852104, -1.54409522, -1.46837056],
                     [-1.05667714, -1.48144313, -1.94813303],
                     [-1.07139407, -1.42933302, -1.94190871],
                     [-0.70164718, -1.81934218, -1.300584],
                     [-0.85893062, -1.8678766, -1.03536173],
                     [-1.12606775, -1.75823432, -0.53537779],
                     [-1.10136968, -1.47137801, -0.74102949],
                     [-1.1605209, -1.43664385, -0.96413043]],

                    [[1.7668459, 2.1677869, 1.39234313],
                     [1.30384137, 1.62498528, 1.58997167],
                     [1.56322518, 1.87070794, 1.87492816],
                     [2.01354145, 1.95079102, 2.43022863],
                     [1.27886166, 2.19556119, 1.76758746],
                     [1.38009609, 2.05569825, 1.91388112],
                     [1.26937554, 2.25655042, 2.01148552],
                     [1.78605414, 2.24317195, 1.13696074],
                     [2.04328967, 2.14251024, 1.56192831],
                     [1.84430403, 2.39644352, 1.41523414]]]),
     'int': np.array([[[-4.44639839e-01, -2.82083775e-01, 1.65670705e-01],
                       [-1.09586137e-01, -3.82176633e-02, -4.59948901e-01],
                       [-1.21555094e-01, -9.03544940e-02, 1.15638469e-01],
                       [-5.31406819e-02, 1.27035091e-01, -5.04614201e-01],
                       [-6.11189634e-02, -1.50734274e-01, -8.75863241e-02],
                       [-4.67516752e-01, -9.83038059e-01, -4.30877084e-02],
                       [-9.94845141e-02, -9.23517697e-01, -5.33564824e-02],
                       [-2.24778885e-01, -5.64131712e-01, 5.62124814e-02],
                       [-3.58558497e-01, 4.93352091e-01, 2.41002545e-04],
                       [-9.14217243e-02, -4.22057786e-01, 3.35735235e-01]]])}


@pytest.fixture()
def arviz_samples_dict_from_pyjags_samples_dict(
        pyjags_samples_dict: tp.Dict[str, np.ndarray]) \
        -> tp.Dict[str, np.ndarray]:
    return _convert_pyjags_samples_dictionary_to_arviz_samples_dictionary(
            pyjags_samples_dict)


@pytest.fixture()
def arviz_inference_data_from_pyjags_samples_dict(
        pyjags_samples_dict: tp.Dict[str, np.ndarray]) \
        -> az.InferenceData:
    return from_pyjags(pyjags_samples_dict)


def verify_equality_of_numpy_values_dictionaries(
        dict_1: tp.Dict[tp.Any, np.ndarray],
        dict_2: tp.Dict[tp.Any, np.ndarray]) -> bool:
    if dict_1.keys() != dict_2.keys():
        return False

    for key in dict_1.keys():
        if not np.all(dict_1[key] == dict_2[key]):
            return False

    return True


def test_convert_pyjags_samples_dictionary_to_arviz_samples_dictionary(
        pyjags_samples_dict: tp.Dict[str, np.ndarray],
        arviz_samples_dict_from_pyjags_samples_dict: tp.Dict[str, np.ndarray]):
    pyjags_samples_dict_from_arviz_samples_dict_from_pyjags_samples_dict = \
        _convert_arviz_samples_dictionary_to_pyjags_samples_dictionary(
            arviz_samples_dict_from_pyjags_samples_dict)

    assert verify_equality_of_numpy_values_dictionaries(
        pyjags_samples_dict,
        pyjags_samples_dict_from_arviz_samples_dict_from_pyjags_samples_dict)


def test_extract_samples_dictionary_from_arviz_inference_data(
        arviz_inference_data_from_pyjags_samples_dict: az.InferenceData,
        arviz_samples_dict_from_pyjags_samples_dict: tp.Dict[str, np.ndarray]):
    arviz_samples_dict_from_arviz_inference_data_from_pyjags_samples_dict = \
        _extract_samples_dictionary_from_arviz_inference_data(
            arviz_inference_data_from_pyjags_samples_dict)

    assert verify_equality_of_numpy_values_dictionaries(
        arviz_samples_dict_from_pyjags_samples_dict,
        arviz_samples_dict_from_arviz_inference_data_from_pyjags_samples_dict)


def test_from_pyjags(
        pyjags_samples_dict: tp.Dict[str, np.ndarray],
        arviz_inference_data_from_pyjags_samples_dict: az.InferenceData):
    arviz_samples_dict_from_arviz_inference_data_from_pyjags_samples_dict = \
        _extract_samples_dictionary_from_arviz_inference_data(
            arviz_inference_data_from_pyjags_samples_dict)

    pyjags_samples_dict_from_arviz_inference_data = \
        _convert_arviz_samples_dictionary_to_pyjags_samples_dictionary(
            arviz_samples_dict_from_arviz_inference_data_from_pyjags_samples_dict)

    assert verify_equality_of_numpy_values_dictionaries(
        pyjags_samples_dict,
        pyjags_samples_dict_from_arviz_inference_data)
