import numpy as np
from scipy import linalg
from pykalman import KalmanFilter
from pykalman.sqrt.cholesky import CholeskyKalmanFilter
from pykalman.sqrt.bierman import BiermanKalmanFilter
from pykalman.unscented import UnscentedKalmanFilter
            

class MyKalmanFilter(KalmanFilter):
    """
    This module implements Kalman Filter.
    """
    index = None
    def _filter_correct(self,observation_matrix, observation_covariance,
                        observation_offset, predicted_state_mean,
                        predicted_state_covariance, observation):
        """Correct a predicted state with a Kalman Filter update

        Incorporate observation `observation` from time `t` to turn
        
        :math:`P(x_t | z_{0:t-1})` into :math:`P(x_t | z_{0:t})`

        Args:
            observation_matrix : [n_dim_obs, n_dim_state] array
                observation matrix for time t
            observation_covariance : [n_dim_obs, n_dim_obs] array
                covariance matrix for observation at time t
            observation_offset : [n_dim_obs] array
                offset for observation at time t
            predicted_state_mean : [n_dim_state] array
                mean of state at time t given observations from times
                [0...t-1]
            predicted_state_covariance : [n_dim_state, n_dim_state] array
                covariance of state at time t given observations from times
                [0...t-1]
            observation : [n_dim_obs] array
                observation at time t.  If `observation` is a masked array and any of
                its values are masked, the observation will be ignored.

        Returns:
            kalman_gain : [n_dim_state, n_dim_obs] array
                Kalman gain matrix for time t
            corrected_state_mean : [n_dim_state] array
                mean of state at time t given observations from times
                [0...t]
            corrected_state_covariance : [n_dim_state, n_dim_state] array
                covariance of state at time t given observations from times
                [0...t]
                
        """
        mask = np.ma.getmask(observation)
        if not np.any(mask):
            predicted_observation_mean = (
                np.dot(observation_matrix,
                       predicted_state_mean)
                + observation_offset
            )
            predicted_observation_covariance = (
                np.dot(observation_matrix,
                       np.dot(predicted_state_covariance,
                              observation_matrix.T))
                + observation_covariance
            )
            kalman_gain = (
                np.dot(predicted_state_covariance,
                       np.dot(observation_matrix.T,
                              linalg.pinv(predicted_observation_covariance)))
            )
            corrected_state_mean = (
                predicted_state_mean
                + np.dot(kalman_gain, observation - predicted_observation_mean)
            )
            corrected_state_covariance = (
                predicted_state_covariance
                - np.dot(kalman_gain,
                         np.dot(observation_matrix,
                                predicted_state_covariance))
            )
                
        elif np.all(mask==False):
            n_dim_state = predicted_state_covariance.shape[0]
            n_dim_obs = observation_matrix.shape[0]
            kalman_gain = np.zeros((n_dim_state, n_dim_obs))
            corrected_state_mean = predicted_state_mean
            corrected_state_covariance = predicted_state_covariance
            
        else:
            n_dim_state = predicted_state_covariance.shape[0]
            ind_missing = [i for i,x in zip(MyKalmanFilter.index,mask) if x]
            ind = [ i for i in range(n_dim_state) if i not in ind_missing]
            
            predicted_observation_mean = (
                np.dot(observation_matrix,
                       predicted_state_mean)
                + observation_offset
            )
            predicted_observation_covariance = (
                np.dot(observation_matrix,
                       np.dot(predicted_state_covariance,
                              observation_matrix.T))
                + observation_covariance
            )
            kalman_gain = (
                np.dot(predicted_state_covariance,
                       np.dot(observation_matrix.T,
                              linalg.pinv(predicted_observation_covariance)))
            )
            v = np.array(observation[~mask] - predicted_observation_mean[~mask])
            corrected_state_mean = predicted_state_mean
            corrected_state_mean[ind] = (
                predicted_state_mean[ind] + np.dot(kalman_gain[np.ix_(ind,~mask)], v)
            )
            corrected_state_covariance = (
                predicted_state_covariance
                - np.dot(kalman_gain,
                         np.dot(observation_matrix,
                                predicted_state_covariance))
            )
            

        return (kalman_gain, corrected_state_mean,
                corrected_state_covariance)
        

    def filter_update(self, filtered_state_mean, filtered_state_covariance,
                      observation=None, transition_matrix=None,
                      transition_offset=None, transition_covariance=None,
                      observation_matrix=None, observation_offset=None,
                      observation_covariance=None):
        r"""Update a Kalman Filter state estimate.

        Perform a one-step update to estimate the state at time :math:`t+1`
        give an observation at time :math:`t+1` and the previous estimate for
        time :math:`t` given observations from times :math:`[0...t]`.  This
        method is useful if one wants to track an object with streaming
        observations.

        Args:
            filtered_state_mean : [n_dim_state] array
                mean estimate for state at time t given observations from times
                [1...t]
            filtered_state_covariance : [n_dim_state, n_dim_state] array
                covariance of estimate for state at time t given observations from
                times [1...t]
            observation : [n_dim_obs] array or None
                observation from time t+1.  If `observation` is a masked array and
                any of `observation`'s components are masked or if `observation` is
                None, then `observation` will be treated as a missing observation.
            transition_matrix : optional, [n_dim_state, n_dim_state] array
                state transition matrix from time t to t+1.  If unspecified,
                `self.transition_matrices` will be used.
            transition_offset : optional, [n_dim_state] array
                state offset for transition from time t to t+1.  If unspecified,
                `self.transition_offset` will be used.
            transition_covariance : optional, [n_dim_state, n_dim_state] array
                state transition covariance from time t to t+1.  If unspecified,
                `self.transition_covariance` will be used.
            observation_matrix : optional, [n_dim_obs, n_dim_state] array
                observation matrix at time t+1.  If unspecified,
                `self.observation_matrices` will be used.
            observation_offset : optional, [n_dim_obs] array
                observation offset at time t+1.  If unspecified,
                `self.observation_offset` will be used.
            observation_covariance : optional, [n_dim_obs, n_dim_obs] array
                observation covariance at time t+1.  If unspecified,
                `self.observation_covariance` will be used.

        Returns:
            next_filtered_state_mean : [n_dim_state] array
                mean estimate for state at time t+1 given observations from times
                [1...t+1]
            next_filtered_state_covariance : [n_dim_state, n_dim_state] array
                covariance of estimate for state at time t+1 given observations
                from times [1...t+1]
                
        """
        from pykalman.standard import _arg_or_default,_filter_predict
        # initialize matrices
        (transition_matrices, transition_offsets, transition_cov,
         observation_matrices, observation_offsets, observation_cov,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )
        transition_offset = _arg_or_default(
            transition_offset, transition_offsets,
            1, "transition_offset"
        )
        observation_offset = _arg_or_default(
            observation_offset, observation_offsets,
            1, "observation_offset"
        )
        transition_matrix = _arg_or_default(
            transition_matrix, transition_matrices,
            2, "transition_matrix"
        )
        observation_matrix = _arg_or_default(
            observation_matrix, observation_matrices,
            2, "observation_matrix"
        )
        transition_covariance = _arg_or_default(
            transition_covariance, transition_cov,
            2, "transition_covariance"
        )
        observation_covariance = _arg_or_default(
            observation_covariance, observation_cov,
            2, "observation_covariance"
        )

        # Make a masked observation if necessary
        if observation is None:
            n_dim_obs = observation_covariance.shape[0]
            observation = np.ma.array(np.zeros(n_dim_obs))
            observation.mask = True
        else:
            observation = np.ma.asarray(observation)

        predicted_state_mean, predicted_state_covariance = (
            _filter_predict(
                transition_matrix, transition_covariance,
                transition_offset, filtered_state_mean,
                filtered_state_covariance
            )
        )
        (_, next_filtered_state_mean,
         next_filtered_state_covariance) = (
            self._filter_correct(
                observation_matrix, observation_covariance,
                observation_offset, predicted_state_mean,
                predicted_state_covariance, observation
            )
        )

        return (next_filtered_state_mean, next_filtered_state_covariance)
        

class MyBiermanKalmanFilter(BiermanKalmanFilter):
    """

    This module implements Bierman's version of the Kalman Filter.  In particular,
    the UDU' decomposition of the covariance matrix is used instead of the full
    matrix, where U is upper triangular and D is diagonal.
    
    """
    index = None
    def _filter_correct(self,observation_matrix, observation_covariance,
                        observation_offset, predicted_state_mean,
                        predicted_state_covariance, observation):
        r"""Correct a predicted state with a Kalman Filter update

        Incorporate observation `observation` from time `t` to turn
        :math:`P(x_t | z_{0:t-1})` into :math:`P(x_t | z_{0:t})`

        Args:
            observation_matrix : [n_dim_obs, n_dim_state] array
                observation matrix for time t
            observation_covariance : n_dim_state UDU_decomposition
                UDU' decomposition of observation covariance matrix for observation at
                time t
            observation_offset : [n_dim_obs] array
                offset for observation at time t
            predicted_state_mean : [n_dim_state] array
                mean of state at time t given observations from times
                [0...t-1]
            predicted_state_covariance : n_dim_state UDU_decomposition
                UDU' decomposition of the covariance of state at time t given
                observations from times [0...t-1]
            observation : [n_dim_obs] array
                observation at time t.  If `observation` is a masked array and any of
                its values are masked, the observation will be ignored.

        Returns:
            corrected_state_mean : [n_dim_state] array
                mean of state at time t given observations from times
                [0...t]
            corrected_state_covariance : n_dim_state UDU_decomposition
                UDU' decomposition of the covariance of state at time t given
                observations from times [0...t]

        References:
            * Gibbs, Bruce P. Advanced Kalman Filtering, Least-Squares, and Modeling: A
              Practical Handbook. Page 394-396
        """
        from pykalman.sqrt.bierman import _filter_correct_single
        mask = np.ma.getmask(observation)
        if not np.any(mask):
            # extract size of state space
            #n_dim_state = len(predicted_state_mean)
            n_dim_obs = len(observation)

            # calculate corrected state mean, covariance
            corrected_state_mean = predicted_state_mean
            corrected_state_covariance = predicted_state_covariance
            for i in range(n_dim_obs):
                # extract components for updating i-th coordinate's covariance
                o = observation[i]
                b = observation_offset[i]
                h = observation_matrix[i]
                R = observation_covariance[i, i]

                # calculate new UDU' decomposition for corrected_state_covariance
                # and a new column of the kalman gain
                (corrected_state_covariance, k) = _filter_correct_single(corrected_state_covariance, h, R)

                # update corrected state mean
                predicted_observation_mean = h.dot(corrected_state_mean) + b
                corrected_state_mean = corrected_state_mean + k.dot(o - predicted_observation_mean)

        elif np.all(mask==False):
            #n_dim_state = len(predicted_state_mean)
            n_dim_obs = len(observation)

            #kalman_gain = np.zeros((n_dim_state, n_dim_obs))

            corrected_state_mean = predicted_state_mean
            corrected_state_covariance = predicted_state_covariance
            
        else:
            n_dim_obs = len(observation)

            # calculate corrected state mean, covariance
            corrected_state_mean = predicted_state_mean
            corrected_state_covariance = predicted_state_covariance
            for i in range(n_dim_obs):
                # extract components for updating i-th coordinate's covariance
                o = observation[i]
                if not np.isnan(o):
                   b = observation_offset[i]
                   h = observation_matrix[i]
                   R = observation_covariance[i, i]

                   # calculate new UDU' decomposition for corrected_state_covariance
                   # and a new column of the kalman gain
                   (corrected_state_covariance, k) = _filter_correct_single(corrected_state_covariance, h, R)

                   # update corrected state mean
                   predicted_observation_mean = h.dot(corrected_state_mean) + b
                
                   corrected_state_mean = corrected_state_mean + k.dot(o - predicted_observation_mean)


        return (corrected_state_mean, corrected_state_covariance)
    
    
    def filter_update(self, filtered_state_mean, filtered_state_covariance,
                      observation=None, transition_matrix=None,
                      transition_offset=None, transition_covariance=None,
                      observation_matrix=None, observation_offset=None,
                      observation_covariance=None):
        r"""Update a Kalman Filter state estimate

        Perform a one-step update to estimate the state at time :math:`t+1`
        give an observation at time :math:`t+1` and the previous estimate for
        time :math:`t` given observations from times :math:`[0...t]`.  This
        method is useful if one wants to track an object with streaming
        observations.

        Args:
            filtered_state_mean : [n_dim_state] array
                mean estimate for state at time t given observations from times
                [1...t]
            filtered_state_covariance : [n_dim_state, n_dim_state] array
                covariance of estimate for state at time t given observations from
                times [1...t]
            observation : [n_dim_obs] array or None
                observation from time t+1.  If `observation` is a masked array and
                any of `observation`'s components are masked or if `observation` is
                None, then `observation` will be treated as a missing observation.
            transition_matrix : optional, [n_dim_state, n_dim_state] array
                state transition matrix from time t to t+1.  If unspecified,
                `self.transition_matrices` will be used.
            transition_offset : optional, [n_dim_state] array
                state offset for transition from time t to t+1.  If unspecified,
                `self.transition_offset` will be used.
            transition_covariance : optional, [n_dim_state, n_dim_state] array
                state transition covariance from time t to t+1.  If unspecified,
                `self.transition_covariance` will be used.
            observation_matrix : optional, [n_dim_obs, n_dim_state] array
                observation matrix at time t+1.  If unspecified,
                `self.observation_matrices` will be used.
            observation_offset : optional, [n_dim_obs] array
                observation offset at time t+1.  If unspecified,
                `self.observation_offset` will be used.
            observation_covariance : optional, [n_dim_obs, n_dim_obs] array
                observation covariance at time t+1.  If unspecified,
                `self.observation_covariance` will be used.

        Returns:
            next_filtered_state_mean : [n_dim_state] array
                mean estimate for state at time t+1 given observations from times
                [1...t+1]
            next_filtered_state_covariance : [n_dim_state, n_dim_state] array
                covariance of estimate for state at time t+1 given observations
                from times [1...t+1]
                
        """
        from pykalman.standard import _arg_or_default
        from pykalman.sqrt.bierman import _filter_predict,udu,decorrelate_observations,_reconstruct_covariances
        # initialize matrices
        (transition_matrices, transition_offsets, transition_cov,
         observation_matrices, observation_offsets, observation_cov,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )
        transition_offset = _arg_or_default(
            transition_offset, transition_offsets,
            1, "transition_offset"
        )
        observation_offset = _arg_or_default(
            observation_offset, observation_offsets,
            1, "observation_offset"
        )
        transition_matrix = _arg_or_default(
            transition_matrix, transition_matrices,
            2, "transition_matrix"
        )
        observation_matrix = _arg_or_default(
            observation_matrix, observation_matrices,
            2, "observation_matrix"
        )
        transition_covariance = _arg_or_default(
            transition_covariance, transition_cov,
            2, "transition_covariance"
        )
        observation_covariance = _arg_or_default(
            observation_covariance, observation_cov,
            2, "observation_covariance"
        )

        # Make a masked observation if necessary
        if observation is None:
           n_dim_obs = observation_covariance.shape[0]
           observation = np.ma.array(np.zeros(n_dim_obs))
           observation.mask = True
        else:
           observation = np.ma.asarray(observation)

        # transform filtered_state_covariance into its UDU decomposition
        filtered_state_covariance = udu(filtered_state_covariance)

        # decorrelate observations
        (observation_matrix, observation_offset,
         observation_covariance, observation) = (
            decorrelate_observations(
                observation_matrix,
                observation_offset,
                observation_covariance,
                observation
            )
        )

        # predict
        predicted_state_mean, predicted_state_covariance = (
           _filter_predict(
                transition_matrix, transition_covariance,
                transition_offset, filtered_state_mean,
                filtered_state_covariance
            )
        )

        # correct
        (next_filtered_state_mean, next_filtered_state_covariance) = (
           self._filter_correct(
                observation_matrix, observation_covariance,
                observation_offset, predicted_state_mean,
                predicted_state_covariance, observation
            )
        )

        # reconstruct actual covariance
        next_filtered_state_covariance = (
           _reconstruct_covariances(next_filtered_state_covariance)
        )

        return (next_filtered_state_mean, next_filtered_state_covariance)


class MyCholeskyKalmanFilter(CholeskyKalmanFilter):
    """
    This module implements the Kalman Filter in "Square Root" form (Cholesky factorization).
    """
    index = None
    def _filter_correct(self,observation_matrix, observation_covariance2,
                        observation_offset, predicted_state_mean,
                        predicted_state_covariance2, observation):
        r"""Correct a predicted state with a Kalman Filter update

        Incorporate observation `observation` from time `t` to turn
        
        :math:`P(x_t | z_{0:t-1})` into :math:`P(x_t | z_{0:t})`

        Args:
            observation_matrix : [n_dim_obs, n_dim_state] array
                observation matrix for time t
            observation_covariance2 : [n_dim_obs, n_dim_obs] array
                square root of the covariance matrix for observation at time t
            observation_offset : [n_dim_obs] array
                offset for observation at time t
            predicted_state_mean : [n_dim_state] array
                mean of state at time t given observations from times
                [0...t-1]
            predicted_state_covariance2 : [n_dim_state, n_dim_state] array
                square root of the covariance of state at time t given observations
                from times [0...t-1]
            observation : [n_dim_obs] array
                observation at time t.  If `observation` is a masked array and any of
                its values are masked, the observation will be ignored.

        Returns:
            corrected_state_mean : [n_dim_state] array
                mean of state at time t given observations from times
                [0...t]
            corrected_state_covariance2 : [n_dim_state, n_dim_state] array
                square root of the covariance of state at time t given observations
                from times [0...t]

        References:
            * Salzmann, M. A. Some Aspects of Kalman Filtering. August 1988. Page 31.
        
        """
        mask = np.ma.getmask(observation)
        if not np.any(mask):
            # extract size of state space
            n_dim_state = len(predicted_state_mean)
            n_dim_obs = len(observation)

            # construct matrix M = [    R^{1/2}^{T},            0;
            #                       (C S_{t|t-1})^T,  S_{t|t-1}^T]
            M = np.zeros(2 * [n_dim_obs + n_dim_state])
            M[0:n_dim_obs, 0:n_dim_obs] = observation_covariance2.T
            M[n_dim_obs:, 0:n_dim_obs] = observation_matrix.dot(predicted_state_covariance2).T
            M[n_dim_obs:, n_dim_obs:] = predicted_state_covariance2.T

            # solve for [((C P_{t|t-1} C^T + R)^{1/2})^T,         K^T;
            #                                          0,   S_{t|t}^T] = QR(M)
            (_, S) = linalg.qr(M)
            kalman_gain = S[0:n_dim_obs,  n_dim_obs:].T
            N = S[0:n_dim_obs, 0:n_dim_obs].T

            # correct mean
            predicted_observation_mean = (
                np.dot(observation_matrix,
                       predicted_state_mean)
                + observation_offset
            )
            corrected_state_mean = (
                predicted_state_mean
                + np.dot(kalman_gain,
                         np.dot(linalg.pinv(N),
                                observation - predicted_observation_mean)
                  )
            )

            corrected_state_covariance2 = S[n_dim_obs:, n_dim_obs:].T
            
        elif np.all(mask==False):
            n_dim_state = predicted_state_covariance2.shape[0]
            n_dim_obs = observation_matrix.shape[0]
            kalman_gain = np.zeros((n_dim_state, n_dim_obs))

            corrected_state_mean = predicted_state_mean
            corrected_state_covariance2 = predicted_state_covariance2
            
        else:
            
            n_dim_state = len(predicted_state_mean)
            ind_missing = [i for i,x in zip(MyCholeskyKalmanFilter.index,mask) if x]
            ind = [ i for i in range(n_dim_state) if i not in ind_missing]
            
            # extract size of state space
            n_dim_state = len(predicted_state_mean)
            n_dim_obs = len(observation)

            # construct matrix M = [    R^{1/2}^{T},            0;
            #                       (C S_{t|t-1})^T,  S_{t|t-1}^T]
            M = np.zeros(2 * [n_dim_obs + n_dim_state])
            M[0:n_dim_obs, 0:n_dim_obs] = observation_covariance2.T
            M[n_dim_obs:, 0:n_dim_obs] = observation_matrix.dot(predicted_state_covariance2).T
            M[n_dim_obs:, n_dim_obs:] = predicted_state_covariance2.T

            # solve for [((C P_{t|t-1} C^T + R)^{1/2})^T,         K^T;
            #                                          0,   S_{t|t}^T] = QR(M)
            (_, S) = linalg.qr(M)
            kalman_gain = S[0:n_dim_obs,  n_dim_obs:].T
            N = S[0:n_dim_obs, 0:n_dim_obs].T

            # correct mean
            predicted_observation_mean = (
                np.dot(observation_matrix,
                       predicted_state_mean)
                + observation_offset
            )
            corrected_state_mean = predicted_state_mean
            v = np.array(observation[~mask] - predicted_observation_mean[~mask])
            iN = linalg.pinv(N)
            iN = iN[:,~mask]
            temp = np.dot(iN,v)
            corrected_state_mean[ind] = (
                predicted_state_mean[ind] + np.dot(kalman_gain[np.ix_(ind,~mask)],temp[~mask])
            )

            corrected_state_covariance2 = S[n_dim_obs:, n_dim_obs:].T
            

        return (corrected_state_mean, corrected_state_covariance2)
        
    def filter_update(self, filtered_state_mean, filtered_state_covariance,
                      observation=None, transition_matrix=None,
                      transition_offset=None, transition_covariance=None,
                      observation_matrix=None, observation_offset=None,
                      observation_covariance=None):
        """Update a Kalman Filter state estimate.

        Perform a one-step update to estimate the state at time :math:`t+1`
        give an observation at time :math:`t+1` and the previous estimate for
        time :math:`t` given observations from times :math:`[0...t]`.  This
        method is useful if one wants to track an object with streaming
        observations.

        Args:
            filtered_state_mean : [n_dim_state] array
                mean estimate for state at time t given observations from times
                [1...t]
            filtered_state_covariance : [n_dim_state, n_dim_state] array
                covariance of estimate for state at time t given observations from
                times [1...t]
            observation : [n_dim_obs] array or None
                observation from time t+1.  If `observation` is a masked array and
                any of `observation`'s components are masked or if `observation` is
                None, then `observation` will be treated as a missing observation.
            transition_matrix : optional, [n_dim_state, n_dim_state] array
                state transition matrix from time t to t+1.  If unspecified,
                `self.transition_matrices` will be used.
            transition_offset : optional, [n_dim_state] array
                state offset for transition from time t to t+1.  If unspecified,
                `self.transition_offset` will be used.
            transition_covariance : optional, [n_dim_state, n_dim_state] array
                state transition covariance from time t to t+1.  If unspecified,
                `self.transition_covariance` will be used.
            observation_matrix : optional, [n_dim_obs, n_dim_state] array
                observation matrix at time t+1.  If unspecified,
                `self.observation_matrices` will be used.
            observation_offset : optional, [n_dim_obs] array
                observation offset at time t+1.  If unspecified,
                `self.observation_offset` will be used.
            observation_covariance : optional, [n_dim_obs, n_dim_obs] array
                observation covariance at time t+1.  If unspecified,
                `self.observation_covariance` will be used.

        Returns:
            next_filtered_state_mean : [n_dim_state] array
                mean estimate for state at time t+1 given observations from times
                [1...t+1]
            next_filtered_state_covariance : [n_dim_state, n_dim_state] array
                covariance of estimate for state at time t+1 given observations
                from times [1...t+1]
        """
        
        from pykalman.standard import _arg_or_default
        from pykalman.sqrt.cholesky import _filter_predict,_reconstruct_covariances
        
        # initialize matrices
        (transition_matrices, transition_offsets, transition_cov,
         observation_matrices, observation_offsets, observation_cov,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )
        transition_offset = _arg_or_default(
            transition_offset, transition_offsets,
            1, "transition_offset"
        )
        observation_offset = _arg_or_default(
            observation_offset, observation_offsets,
            1, "observation_offset"
        )
        transition_matrix = _arg_or_default(
            transition_matrix, transition_matrices,
            2, "transition_matrix"
        )
        observation_matrix = _arg_or_default(
            observation_matrix, observation_matrices,
            2, "observation_matrix"
        )
        transition_covariance = _arg_or_default(
            transition_covariance, transition_cov,
            2, "transition_covariance"
        )
        observation_covariance = _arg_or_default(
            observation_covariance, observation_cov,
            2, "observation_covariance"
        )

        # Make a masked observation if necessary
        if observation is None:
           n_dim_obs = observation_covariance.shape[0]
           observation = np.ma.array(np.zeros(n_dim_obs))
           observation.mask = True
        else:
           observation = np.ma.asarray(observation)

        # turn covariance into cholesky factorizations
        transition_covariance2 = linalg.cholesky(transition_covariance, lower=True)
        observation_covariance2 = linalg.cholesky(observation_covariance, lower=True)
        filtered_state_covariance2 = linalg.cholesky(filtered_state_covariance, lower=True)

        # predict
        predicted_state_mean, predicted_state_covariance2 = (
            _filter_predict(
                transition_matrix, transition_covariance2,
                transition_offset, filtered_state_mean,
                filtered_state_covariance2
            )
        )

        # correct
        (next_filtered_state_mean, next_filtered_state_covariance2) = (
            self._filter_correct(
                observation_matrix, observation_covariance2,
                observation_offset, predicted_state_mean,
                predicted_state_covariance2, observation
            )
        )

        next_filtered_state_covariance = (
            _reconstruct_covariances(next_filtered_state_covariance2)
        )

        return (next_filtered_state_mean, next_filtered_state_covariance)
    
    
class MyUnscentedKalmanFilter(UnscentedKalmanFilter):
    '''
    This module contains "Square Root" implementations to the Unscented Kalman
    Filter.  Square Root implementations typically propagate the mean and Cholesky
    factorization of the covariance matrix in order to prevent numerical error.
    When possible, Square Root implementations should be preferred to their
    standard counterparts.

    References:
        
    * Terejanu, G.A. Towards a Decision-Centric Framework for Uncertainty
      Propagation and Data Assimilation. 2010. Page 108.
    * Van Der Merwe, R. and Wan, E.A. The Square-Root Unscented Kalman Filter for
      State and Parameter-Estimation. 2001.
    '''
    index = None
    def _unscented_correct(self,cross_sigma, moments_pred, obs_moments_pred, z):
        '''Correct predicted state estimates with an observation

        Args:
            cross_sigma : [n_dim_state, n_dim_obs] array
                cross-covariance between the state at time t given all observations
                from timesteps [0, t-1] and the observation at time t
            moments_pred : [n_dim_state] Moments
                mean and covariance of state at time t given observations from
                timesteps [0, t-1]
            obs_moments_pred : [n_dim_obs] Moments
                mean and covariance of observation at time t given observations from
                times [0, t-1]
            z : [n_dim_obs] array
                observation at time t

        Returns:
            moments_filt : [n_dim_state] Moments
                mean and covariance of state at time t given observations from time
                steps [0, t]
        '''
        from pykalman.sqrt.unscented import cholupdate
        from pykalman.unscented import Moments
        mu_pred, sigma2_pred = moments_pred
        obs_mu_pred, obs_sigma2_pred = obs_moments_pred

        #n_dim_state = len(mu_pred)
        #n_dim_obs = len(obs_mu_pred)
        mask = np.ma.getmask(z)
        if not np.any(mask):
            ##############################################
            # Same as this, but more stable (supposedly) #
            ##############################################
            # K = cross_sigma.dot(
            #     linalg.pinv(
            #         obs_sigma2_pred.T.dot(obs_sigma2_pred)
            #     )
            # )
            ##############################################

            # equivalent to this MATLAB code
            # K = (cross_sigma / obs_sigma2_pred.T) / obs_sigma2_pred
            K = linalg.lstsq(obs_sigma2_pred, cross_sigma.T)[0]
            K = linalg.lstsq(obs_sigma2_pred.T, K)[0]
            K = K.T

            # correct mu, sigma
            mu_filt = mu_pred + K.dot(z - obs_mu_pred)
            U = K.dot(obs_sigma2_pred)
            sigma2_filt = cholupdate(sigma2_pred, U.T, -1.0)
        elif np.all(mask==False):
            # no corrections to be made
            mu_filt = mu_pred
            sigma2_filt = sigma2_pred
        else:
            n_dim_state = sigma2_pred.shape[0]
            ind_missing = [i for i,x in zip(MyUnscentedKalmanFilter.index,mask) if x]
            ind = [ i for i in range(n_dim_state) if i not in ind_missing]
            
            K = linalg.lstsq(obs_sigma2_pred, cross_sigma.T)[0]
            K = linalg.lstsq(obs_sigma2_pred.T, K)[0]
            K = K.T

            # correct mu, sigma
            v = z[~mask] - obs_mu_pred[ind]
            mu_filt = mu_pred
            mu_filt += np.dot(K[ind],v)
            U = K.dot(obs_sigma2_pred)
            sigma2_filt = cholupdate(sigma2_pred, U.T, -1.0)
            
        return Moments(mu_filt, sigma2_filt)
    
    
    def unscented_filter_correct(self,observation_function, moments_pred,
                             points_pred, observation,
                             points_observation=None,
                             sigma2_observation=None):
        """Integrate new observation to correct state estimates
    
        Args:
            observation_function : function
                function characterizing how the observation at time t+1 is generated
            moments_pred : [n_dim_state] Moments
                mean and covariance of state at time t+1 given observations from time
                steps 0...t
            points_pred : [2*n_dim_state+1, n_dim_state] SigmaPoints
                sigma points corresponding to moments_pred
            observation : [n_dim_state] array
                observation at time t+1. If masked, treated as missing.
            points_observation : [2*n_dim_state, n_dim_obs] SigmaPoints
                sigma points corresponding to predicted observation at time t+1 given
                observations from times 0...t, if available. If not, noise is assumed
                to be additive.
            sigma_observation : [n_dim_obs, n_dim_obs] array
                covariance matrix corresponding to additive noise in observation at
                time t+1, if available. If missing, noise is assumed to be non-linear.
        
        Returns:
            moments_filt : [n_dim_state] Moments
            mean and covariance of state at time t+1 given observations from time steps 0...t+1
        
        """
        from pykalman.sqrt.unscented import _unscented_transform
        # Calculate E[z_t | z_{0:t-1}], Var(z_t | z_{0:t-1})
        (obs_points_pred, obs_moments_pred) = (
            _unscented_transform(
                points_pred, observation_function,
                points_noise=points_observation, sigma2_noise=sigma2_observation
            )
        )
    
        # Calculate Cov(x_t, z_t | z_{0:t-1})
        sigma_pair = (
            ((points_pred.points - moments_pred.mean).T)
            .dot(np.diag(points_pred.weights_mean))
            .dot(obs_points_pred.points - obs_moments_pred.mean)
        )
    
        # Calculate E[x_t | z_{0:t}], Var(x_t | z_{0:t})
        moments_filt = self._unscented_correct(sigma_pair, moments_pred, obs_moments_pred, observation)
        
        return moments_filt
    
    
    def filter_update(self,
                      filtered_state_mean, filtered_state_covariance,
                      observation=None,
                      transition_function=None, transition_covariance=None,
                      observation_function=None, observation_covariance=None):
        r"""Update a Kalman Filter state estimate

        Perform a one-step update to estimate the state at time :math:`t+1`
        give an observation at time :math:`t+1` and the previous estimate for
        time :math:`t` given observations from times :math:`[0...t]`.  This
        method is useful if one wants to track an object with streaming
        observations.

        Args:
            filtered_state_mean : [n_dim_state] array
                mean estimate for state at time t given observations from times
                [1...t]
            filtered_state_covariance : [n_dim_state, n_dim_state] array
                covariance of estimate for state at time t given observations from
                times [1...t]
            observation : [n_dim_obs] array or None
                observation from time t+1.  If `observation` is a masked array and
                any of `observation`'s components are masked or if `observation` is
                None, then `observation` will be treated as a missing observation.
            transition_function : optional, function
                state transition function from time t to t+1.  If unspecified,
                `self.transition_functions` will be used.
            transition_covariance : optional, [n_dim_state, n_dim_state] array
                state transition covariance from time t to t+1.  If unspecified,
                `self.transition_covariance` will be used.
            observation_function : optional, function
                observation function at time t+1.  If unspecified,
                `self.observation_functions` will be used.
            observation_covariance : optional, [n_dim_obs, n_dim_obs] array
                observation covariance at time t+1.  If unspecified,
                `self.observation_covariance` will be used.

        Returns:
            next_filtered_state_mean : [n_dim_state] array
                mean estimate for state at time t+1 given observations from times
                [1...t+1]
            next_filtered_state_covariance : [n_dim_state, n_dim_state] array
                covariance of estimate for state at time t+1 given observations
                from times [1...t+1]
                
        """
        from pykalman.standard import _arg_or_default
        from pykalman.sqrt.unscented import unscented_filter_predict,moments2points,_reconstruct_covariances
        from pykalman.unscented import Moments
        # initialize parameters
        (transition_functions, observation_functions,
         transition_cov, observation_cov,
         _, _) = (
            self._initialize_parameters()
        )

        def default_function(f, arr):
           if f is None:
              assert len(arr) == 1
              f = arr[0]
           return f

        transition_function = default_function(
            transition_function, transition_functions
        )
        observation_function = default_function(
            observation_function, observation_functions
        )
        transition_covariance = _arg_or_default(
            transition_covariance, transition_cov,
            2, "transition_covariance"
        )
        observation_covariance = _arg_or_default(
            observation_covariance, observation_cov,
            2, "observation_covariance"
        )

        # Make a masked observation if necessary
        if observation is None:
           n_dim_obs = observation_covariance.shape[0]
           observation = np.ma.array(np.zeros(n_dim_obs))
           observation.mask = True
        else:
           observation = np.ma.asarray(observation)

        # preprocess covariance matrices
        filtered_state_covariance2 = linalg.cholesky(filtered_state_covariance)
        transition_covariance2 = linalg.cholesky(transition_covariance)
        observation_covariance2 = linalg.cholesky(observation_covariance)

        # make sigma points
        moments_state = Moments(filtered_state_mean, filtered_state_covariance2)
        points_state = moments2points(moments_state)

        # predict
        (_, moments_pred) = (
            unscented_filter_predict(
                transition_function, points_state,
                sigma2_transition=transition_covariance2
            )
        )
        points_pred = moments2points(moments_pred)

        # correct
        (next_filtered_state_mean, next_filtered_state_covariance2) = (
            self.unscented_filter_correct(
                observation_function, moments_pred, points_pred,
                observation, sigma2_observation=observation_covariance2
            )
        )

        next_filtered_state_covariance = (
            _reconstruct_covariances(next_filtered_state_covariance2)
        )

        return (next_filtered_state_mean, next_filtered_state_covariance)
    
if __name__ == '__main__':
    """Test of unscented Kalman filter."""
    import pylab as pl
    
    # initialize parameters
    def transition_function(state, noise):
        a = np.sin(state[0]) + state[1] * noise[0]
        b = state[1] + noise[1]
        return np.array([a, b])
    
    def observation_function(state, noise):
        C = np.array([[-1, 0.5], [0.2, 0.1]])
        return np.dot(C, state) + noise
    
    transition_covariance = np.eye(2)
    random_state = np.random.RandomState(0)
    observation_covariance = np.eye(2) + random_state.randn(2, 2) * 0.1
    initial_state_mean = [0, 0]
    initial_state_covariance = [[1, 0.1], [-0.1, 1]]
    
    # sample from model
    kf = MyUnscentedKalmanFilter(
        transition_function, observation_function,
        transition_covariance, observation_covariance,
        initial_state_mean, initial_state_covariance,
        random_state=random_state
    )
    states, observations = kf.sample(50, initial_state_mean)
    
    # estimate state with filtering and smoothing
    filtered_state_estimates = kf.filter(observations)[0]
    smoothed_state_estimates = kf.smooth(observations)[0]
    
    # draw estimates
    pl.figure()
    lines_true = pl.plot(states, color='b')
    lines_filt = pl.plot(filtered_state_estimates, color='r', ls='-')
    lines_smooth = pl.plot(smoothed_state_estimates, color='g', ls='-.')
    pl.legend((lines_true[0], lines_filt[0], lines_smooth[0]),
              ('true', 'filt', 'smooth'),
              loc='lower left'
    )
    pl.show()
    