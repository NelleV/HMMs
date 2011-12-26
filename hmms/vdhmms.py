import numpy as np


def alpha(A, P, D):
    """
    Alpha computation

    Parameters
    ----------
        A: ndarray, matrice de passage
            Probability of having state j knowing that the previous state was
            i

        P: ndarray, probabilite d'emission
            Probability of having observation y knowing state i

        D: ndarray, probabilite d'emettre d observations sachant l'etat i

    Returns
    -------
        alpha: (num_states, num_obs) la chaine alpha
    """
    # Get the number of possible states
    num_states, _ = A.shape
    # Maximum length of a segment
    _, max_length = D.shape
    # Number of observations (ie, length of the chain)
    num_obs, _ = P.shape

    print "running alpha computation on %d observation for %d" % (num_obs,
                                                                  num_states)
    print "states of max %d observation" % max_length

    alpha = np.zeros((num_obs, num_states))
    for t in range(num_obs):
        if t == 0:
            # First element: we need to initialize the chain.
            alpha[0, :] = P[0, :]
        else:
            for i in range(num_states):
                for d in range(max_length):
                    if t - d - 1 < 0:
                        print " for t %d, breaking at d: %d" % (t, d)
                        break
                    alpha[t, i] += P[t - d - 1:t, i].prod() * D[i, d] * \
                                        (A[:, i] * alpha[t - d - 1, :]).sum()
    return alpha

def beta(A, P, D, verbose=False):
    """
    Beta computation

    Parameters
    ----------
        A: ndarray, matrice de passage
            Probability of having state j knowing that the previous state was
            i

        P: ndarray, probabilite d'emission
            Probability of having observation y knowing state i

        D: ndarray, probabilite d'emettre d observations sachant l'etat i

        verbose: boolean, optional
            more verbose output

    Returns
    -------
        beta: (num_states, num_obs) la chaine beta
    """
    # Get the number of possible states
    num_states, _ = A.shape
    # Maximum length of a segment
    _, max_length = D.shape
    # Number of observations (ie, length of the chain)
    num_obs, _ = P.shape

    print "running beta computation on %d observation for %d" % (num_obs,
                                                                 num_states)
    print "states of max %d observation" % max_length

    beta = np.zeros((num_obs, num_states)).astype(float)
    for iteration, t in enumerate(range(num_obs - 1, 0, -1)):
        if iteration == 0:
            print "Initialising betas"
            beta[t] += 1.
        else:
            for i in range(num_states):
                for j in range(num_states):
                    b = 0
                    for d in range(max_length):
                        if t + d + 1 > num_obs - 1:
                            break
                        b += D[j, d] * beta[t + d + 1, j] * \
                                P[t + 1:t + d + 2, j].prod()
                    beta[t, i] += A[i, j] * b
    return beta
