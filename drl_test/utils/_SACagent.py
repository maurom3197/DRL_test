import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from drl_test.utils.tfp_gaussian_actor import GaussianActor
from drl_test.utils.tfp_convmix_gaussian_actor import ConvGaussianActor


class SACAgent():
    """
    Soft Actor-Critic (SAC) Agent: https://arxiv.org/abs/1801.01290
    """
    def __init__(
            self,
            state_shape,
            action_dim,
            image_shape=(112,112,1),
            name="SAC",
            max_action=(0.5,1.0),
            min_action=(0.,-1.),
            actor_units=(256, 256),
            network='mlp',
            log_level = 20,
            **kwargs):
        """
        Initialize SAC

        Args:
            state_shape (iterable of int):
            action_dim (int):
            name (str): Name of network. The default is ``"SAC"``
            max_action (float):
            lr (float): Learning rate. The default is ``3e-4``.
            lr_alpha (alpha): Learning rate for alpha. The default is ``3e-4``.
            actor_units (iterable of int): Numbers of units at hidden layers of actor. The default is ``(256, 256)``.
            critic_units (iterable of int): Numbers of units at hidden layers of critic. The default is ``(256, 256)``.
            tau (float): Target network update rate.
            alpha (float): Temperature parameter. The default is ``0.2``.
            auto_alpha (bool): Automatic alpha tuning.
            init_temperature (float): Initial temperature
            n_warmup (int): Number of warmup steps before training. The default is ``int(1e4)``.
            memory_capacity (int): Replay Buffer size. The default is ``int(1e6)``.
            batch_size (int): Batch size. The default is ``256``.
            discount (float): Discount factor. The default is ``0.99``.
            max_grad (float): Maximum gradient. The default is ``10``.
            gpu (int): GPU id. ``-1`` disables GPU. The default is ``0``.
        """
        super().__init__(
            name=name, **kwargs)

        self.log_level = log_level
        self._setup_actor(state_shape, image_shape, action_dim, actor_units, max_action, min_action, network)
        self.state_ndim = len(state_shape)

    def _setup_actor(self, state_shape, image_shape, action_dim, actor_units, max_action, min_action, network='mlp'):
        if network=='mlp':
            self.actor = GaussianActor(
                state_shape, action_dim, max_action, min_action, squash=True, units=actor_units)
        elif network=='conv':
            self.actor = ConvGaussianActor(
                state_shape, image_shape, action_dim, max_action, min_action, squash=True, units=actor_units)
        #self.actor.model().summary()

    def get_action(self, state, test=False):
        """
        Get action

        Args:
            state: Observation state
            test (bool): When ``False`` (default), policy returns exploratory action.

        Returns:
            tf.Tensor or float: Selected action
        """
            
        assert isinstance(state, np.ndarray)
        is_single_state = len(state.shape) == self.state_ndim

        state = np.expand_dims(state, axis=0) if is_single_state else state
        action = self._get_action_body(tf.constant(state), test)
        return action.numpy()[0] if is_single_state else action

    @tf.function
    def _get_action_body(self, state, test):
        actions, _ = self.actor(state, test)
        return actions
    
    def load_weights(self, model_path):
        self.actor.load_weights(model_path)

