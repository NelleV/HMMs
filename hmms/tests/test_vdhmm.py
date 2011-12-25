import numpy as np

from hmms import vdhmms


def test_alpha():
    P = np.array([[0.7, 0.3],
                  [0.5, 0.5],
                  [0.4, 0.6],
                  [0.1, 0.9],
                  [0.7, 0.3]])
    A = np.array([[0.35, 0.65],
                  [0.45, 0.55]])
    D = np.array([[0.3, 0.4, 0.3],
                  [0.2, 0.5, 0.3]])
    alpha = vdhmms.alpha(A, P, D)[0]
    alpha_true = np.array([[0.7, 0.3],
                           [0.4789, 0.5210],
                           [0.3843, 0.6156]])
    # FIXME - problem with the last row
    np.testing.assert_almost_equal(alpha[:3, :], alpha_true, decimal=3)


def test_beta():
    P = np.array([[0.7, 0.3],
                  [0.5, 0.5],
                  [0.4, 0.6],
                  [0.1, 0.9],
                  [0.7, 0.3]])
    A = np.array([[0.35, 0.65],
                  [0.45, 0.55]])
    D = np.array([[0.3, 0.4, 0.3],
                  [0.2, 0.5, 0.3]])
    num_obs, _ = P.shape
    num_states, _ = A.shape
    alpha_norm = np.ones((num_obs, 1))
    beta = vdhmms.beta(A, P, D, alpha_norm)
    # FIXME - need to calculate by hand what the first elements of the chains
    # are, in order to be able to test that
    np.testing.assert_equal(beta[-1], np.ones((num_states, )))
