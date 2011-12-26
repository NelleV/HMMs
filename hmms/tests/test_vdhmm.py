import numpy as np

from hmms import vdhmms


def test_alpha():
    P = np.array([[0.7, 0.3],
                  [0.5, 0.5],
                  [0.4, 0.6],
                  [0.1, 0.9],
                  [0.7, 0.3],
                  [0.6, 0.4],
                  [0.6, 0.4],
                  [0.5, 0.5],
                  [0.4, 0.6],
                  [0.4, 0.6]])
    A = np.array([[0.35, 0.65],
                  [0.45, 0.55]])
    D = np.array([[0.3, 0.4, 0.3],
                  [0.2, 0.5, 0.3]])
    alpha = vdhmms.alpha(A, P, D)
    alpha_true = np.array([[0.7, 0.3],
                           [0.0798, 0.0372],
                           [0.0599, 0.0537]])
    np.testing.assert_almost_equal(alpha[:3, :], alpha_true, decimal=3)


def test_beta():
    P = np.array([[0.7, 0.3],
                  [0.5, 0.5],
                  [0.4, 0.6],
                  [0.1, 0.9],
                  [0.7, 0.3],
                  [0.6, 0.4],
                  [0.6, 0.4],
                  [0.5, 0.5],
                  [0.4, 0.6],
                  [0.4, 0.6]])

    A = np.array([[0.35, 0.65],
                  [0.45, 0.55]])
    D = np.array([[0.3, 0.4, 0.3],
                  [0.2, 0.5, 0.3]])
    num_obs, _ = P.shape
    num_states, _ = A.shape
    beta = vdhmms.beta(A, P, D)
    beta_true = np.array([[0.1538, 0.1422],
                          [0.12, 0.12],
                          [1, 1]])
    np.testing.assert_almost_equal(beta[-3:], beta_true, decimal=3)
