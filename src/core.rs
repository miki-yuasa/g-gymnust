use crate::utils::seeding;

/// The main Gymnust `Env` trait implementing Reinforcement Learning Agents environments.
///
/// The trait encapsulates an environment with arbitrary behind-the-scenes dynamics though the `step` and `reset` functions.
/// An environment can be partially or fully observed by single agents.
/// Multi-agent environments are future work.
///
/// The main API methods that users of this trait need to know are:
/// * `step` - Run one timestep of the environment's dynamics.
/// When end of episode is reached, you are responsible for calling `reset` to reset this environment's state.
/// * `reset` - Reset the environment's state. Returns the initial observation.
/// * `render` - Render the environment to help visualize what the agent see, example modes are human, rgb_array, ansi, etc.
/// * `close` - Cleanup any resources.
///
/// The structs that implement this trait need have additional attributes for users to understand the implementation.
/// * `action_space` - The action space of the environment.
/// * `observation_space` - The observation space of the environment.
/// * `spec` - An environment specification that contains the information used to initialize the environment from `gymnust::make()`.
/// * `metadata` - Additional information about the environment i.e. render modes, render fps, etc.
/// * `rs_random` - A random number generator and a seed that corresponds to `np_random` in Gymnasium.
/// This is automatically assigned during `reset()` and when assessing `rs_random`.
///
/// Note:
///     To get reproducible sampling of actions, a seed can be set with ``action_space.seed(seed)``.
pub trait Env {}
