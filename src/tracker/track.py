from tracker.utils import convert_bbox_to_xywh, convert_xywh_to_bbox, speed_direction
import numpy as np

def calculate_direction(vector1, vector2):
    # Convert lists to numpy arrays for easier calculations
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    
    # Calculate the direction vector
    direction_vector = v2 - v1
    
    # Calculate the norm (length) of the direction vector
    norm = np.linalg.norm(direction_vector)
    
    # Check if the norm is zero to avoid division by zero
    if norm == 0:
        return direction_vector, direction_vector
    
    # Normalize the direction vector
    normalized_direction = direction_vector / norm
    
    return direction_vector, normalized_direction

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, score, delta_t=3, kf_name="Fading"):
        """
        Initialises a tracker using initial bounding box.

        """
        self._std_weight_position = 1. / 20 #20
        self._std_weight_velocity = 1. / 160 #160

        # define constant velocity model
        if kf_name == "New":
          from tracker.kalman_filter import KalmanFilterNew as KalmanFilter
          self.kf = KalmanFilter(dim_x=8, dim_z=4)
        elif kf_name == "Fading":
          from filterpy.kalman import FadingKalmanFilter
          self.kf = FadingKalmanFilter(alpha=1.50, dim_x=8, dim_z=4)
        else:
          from filterpy.kalman import KalmanFilter
          self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        dt = 1/20
        self.kf.F = np.array([[1, 0., 0., 0., dt, 0., 0., 0.],
                              [0., 1, 0., 0., 0., dt, 0., 0.],
                              [0., 0., 1, 0., 0., 0., dt, 0.],
                              [0., 0., 0., 1, 0., 0., 0., dt],
                              [0., 0., 0., 0., 1, 0., 0., 0.],
                              [0., 0., 0., 0., 0., 1, 0., 0.],
                              [0., 0., 0., 0., 0., 0., 1, 0.],
                              [0., 0., 0., 0., 0., 0., 0., 1]])
        self.kf.H = np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 1., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 1., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 1., 0., 0., 0., 0.]])

        
        measurement = convert_bbox_to_xywh(bbox) 
        
        # ====================================================================
        # Measurement Noise Covariance Process noise refers to the random variations or disturbances that affect the evolution of a dynamic system over time
        self.kf.R *= self._std_weight_position
        # Covariance Prediction
        std = [
            2*self._std_weight_position * measurement[2][0], #2
            2*self._std_weight_position * measurement[3][0], #2
            2*self._std_weight_position * measurement[2][0], #2
            2*self._std_weight_position * measurement[3][0], #2
            10*self._std_weight_velocity * measurement[2][0], #10
            10*self._std_weight_velocity * measurement[3][0], #10
            10*self._std_weight_velocity * measurement[2][0], #10
            10*self._std_weight_velocity * measurement[3][0]] #10
        init_P = np.diag(np.square(std))
        self.kf.P = init_P
        
        # Matrix of process noise (8x8)
        std_pos = [
            self._std_weight_position * measurement[2][0],
            self._std_weight_position * measurement[3][0],
            self._std_weight_position * measurement[2][0],
            self._std_weight_position * measurement[3][0]
        ]
        std_vel = [
            self._std_weight_velocity * measurement[2][0],
            self._std_weight_velocity * measurement[3][0],
            self._std_weight_velocity * measurement[2][0],
            self._std_weight_velocity * measurement[3][0]
        ]
        init_Q = np.diag(np.square(np.r_[std_pos, std_vel]))
        self.kf.Q = init_Q

        self.kf.x[:4] = convert_bbox_to_xywh(bbox)[:4]
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t
        self.score = score
        self.feature_mask = None 
        self.feature = None

    def update(self, bbox, confidence=0.0):
        """
        Updates the state vector with observed bbox.
        """
        if bbox is not None:
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age-dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)
            
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_xywh(bbox))#, confidence=bbox[-1])
        else:
            self.kf.update(bbox)#, confidence=0.0)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_xywh_to_bbox(self.kf.x, self.score))
        #print("self.kf.x: ", self.kf.x)
        return self.history[-1]
    
    # Method from BoTSORT
    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            result = []
            multi_mean = np.asarray([st.kf.x.copy() for st in stracks])
            multi_covariance = np.asarray([st.kf.P for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = mean.reshape((1,8))[0]
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())
                stracks[i].kf.x = mean.reshape((8, 1))
                stracks[i].kf.P = cov
                result.append(stracks[i])
            return result
        return stracks

    @staticmethod
    def multi_gmc_xy(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            result = []
            multi_mean = np.asarray([st.kf.x.copy() for st in stracks])
            multi_covariance = np.asarray([st.kf.P for st in stracks])

            R = H[:2, :2]  # Rotation/scale matrix
            t = H[:2, 2]   # Translation vector

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = mean.reshape((1, 8))[0]  # Flatten the state vector to 1D
                
                # Transform x center and y center (first two elements of the state vector)
                xy = mean[:2]  # Extract x and y center
                transformed_xy = R.dot(xy) + t  # Apply rotation and translation
                mean[:2] = transformed_xy  # Update x and y center
                
                # Transform covariance matrix (only related to x and y)
                R2x2 = R  # 2x2 transformation for covariance
                cov[:2, :2] = R2x2.dot(cov[:2, :2]).dot(R2x2.T)
                
                # Update the Kalman filter state and covariance
                stracks[i].kf.x = mean.reshape((8, 1))  # Reshape back to column vector
                result.append(stracks[i])

            return result
        
        return stracks

        
    @staticmethod
    def multi_gmc_xy_buffer(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            result = []
            multi_mean = np.asarray([st.kf.x.copy() for st in stracks])
            multi_covariance = np.asarray([st.kf.P for st in stracks])

            R = H[:2, :2]  # Rotation/scale matrix
            t = H[:2, 2]   # Translation vector

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = mean.reshape((1, 8))[0]  # Flatten the state vector to 1D
                
                # Transform x center and y center (first two elements of the state vector)
                xy = mean[:2]  # Extract x and y center
                transformed_xy = R.dot(xy) + t  # Apply rotation and translation
                mean[:2] = transformed_xy  # Update x and y center
                
                # Transform covariance matrix (only related to x and y)
                R2x2 = R  # 2x2 transformation for covariance
                cov[:2, :2] = R2x2.dot(cov[:2, :2]).dot(R2x2.T)
                
                # Update the Kalman filter state and covariance
                stracks[i].kf.x = mean.reshape((8, 1))  # Reshape back to column vector
                result.append(stracks[i])
            return result
        return stracks
    
    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_xywh_to_bbox(self.kf.x, score=self.score)