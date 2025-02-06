import gymnasium as gym
import numpy as np
import cv2


class GrayScaleObservation(gym.ObservationWrapper):
    """
    Convert RGB observations to grayscale
    """
    def __init__(self, env, keep_dim=True):
        super().__init__(env)
        self.keep_dim = keep_dim
        old_shape = self.observation_space.shape  # (210,160,3)

        new_shape = (old_shape[0], old_shape[1], 1)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # shape (H, W)
        if self.keep_dim:
            gray = np.expand_dims(gray, axis=-1)      # shape (H, W, 1)
        return gray


class ResizeObservation(gym.ObservationWrapper):
    """
    Resize observations to (84,84) while keeping the channel dimension.
    """
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        old_shape = self.observation_space.shape 
        # new shape (84,84,1)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shape[0], shape[1], old_shape[2]),
            dtype=np.uint8
        )

    def observation(self, obs):
        resized = cv2.resize(obs, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)
        if len(resized.shape) == 2:
            resized = np.expand_dims(resized, axis=-1)
        return resized