/// The main Gymnust `Env` trait implementing Reinforcement Learning Agents environments.
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
struct Env<ActSpace, ObsSpace, Metadata> {
    action_space: ActSpace,
    observation_space: ObsSpace,
    spec: Option<String>,
    metadata: Metadata,
    rs_random: Option<Generator>,
}

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
pub trait Dynamics<ObsType, ActType> {
    /// Run one timestep of the environment's dynamics using the agent action.
    ///
    /// When the end of an episode is reached (``terminated`` or ``truncated``), ut us necessary to call `reset` to reset the environment's state for the next episode.
    /// For bootstrapping in reinforcement learning algorithms, ``terminated`` and ``truncated`` are used to indicate the end of an episode to make it clear to users how an episode ended.
    ///
    /// # Arguments
    /// * `action` - The action to be executed in the environment.
    ///
    /// # Returns
    /// * `observation` - The agent's observation of the current environment.
    /// * `reward` - The reward from the environment for the action taken.
    /// * `terminated` - A boolean indicating if the episode has ended.
    /// * `truncated` - A boolean indicating if the episode was truncated.
    /// * `info` - A dictionary containing additional information about the environment.
    fn step<T>(&mut self, action: ActType) -> (ObsType, f32, bool, bool, T);

    /// Reset the environment to an initial internal state, returning an initial observation and info.
    ///
    /// This method generates a new starting state often with some randomness to ensure that the agent explores the state space and learns a generalized policy about the environment.
    /// This randomness is controlled by the `seed` parameter otherwise if the environment already has a random number generator and `reset` is called without a seed, the RNG is not reset.
    ///
    /// Therefore, `reset` should be called (in the typical use case) with a seed right after the initialization of the environment and never agin.
    ///
    /// For custom environments, implement this seeding behavior in the `reset` method.
    ///
    /// # Arguments
    /// * `seed` - An optional seed for the random number generator to initialize the environment's PRNG (`rs_random`) and the read-only attribute `rs_random_seed`.
    ///     If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed, a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
    ///     However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset and the env's :attr:`np_random_seed` will *not* be altered.
    ///     If you pass an integer, the PRNG will be reset even if it already exists.
    ///     Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
    ///     Please refer to the minimal example above to see this paradigm in action.
    /// * `options` - Additional options to be passed to specify how the environment is reset (optional, depending on the specific environment)
    ///
    /// # Returns
    /// * `observation` - The initial observation of the environment.
    /// * `info` - A dictionary containing additional information about the environment.
    #[allow(unused_variables)]
    fn reset<T, U>(&mut self, seed: Option<u32>, options: Option<T>) -> (ObsType, U);
    /// Compute the render frame(s) as specified by the `render_mode` during initialization of the environment.
    ///
    /// The environment's :attr:`metadata` render modes (`env.metadata["render_modes"]`) should contain the possible  ways to implement the render modes.
    /// In addition, list versions for most render modes is achieved through `gymnust::make` which automatically applies a wrapper to collect rendered frames.
    /// Note:
    ///     As the :attr:`render_mode` is known during initialization, the objects used to render the environment state should be initialized in initialization.
    /// By convention, if the :attr:`render_mode` is:
    ///
    /// - None (default): no render is computed.
    /// - "human": The environment is continuously rendered in the current display or terminal, usually for human consumption.
    ///   This rendering should occur during :meth:`step` and :meth:`render` doesn't need to be called. Returns ``None``.
    /// - "rgb_array": Return a single frame representing the current state of the environment.
    ///   A frame is a vector with shape ``(x, y, 3)`` representing RGB values for an x-by-y pixel image.
    /// - "ansi": Return a strings (``str``) or ``StringIO.StringIO`` containing a terminal-style text representation
    ///   for each time step. The text can include newlines and ANSI escape sequences (e.g. for colors).
    /// - (TBD) "rgb_array_list" and "ansi_list": List based version of render modes are possible (except Human) through the
    ///   wrapper, :py:class:`gymnust.wrappers.RenderCollection` that is automatically applied during ``gymnasium.make(..., render_mode="rgb_array_list")``.
    ///   The frames collected are popped after :meth:`render` is called or :meth:`reset`.
    ///
    /// Note:
    ///    Make sure that your class's :attr:`metadata` ``"render_modes"`` key includes the list of supported modes.
    fn render<RenderFrame>(&self) -> Option<RenderFrame>;

    /// Close the environment and free resources.
    /// This method should be called when the environment is no longer needed.
    fn close(&self);

    /// Return the base non-wrapped environment.
    /// This method should be implemented to return `Self`.
    fn unwrapped(&self);

    fn to_string(&self) -> String;
    // {
    //     let spec = self._get_spec_id();
    //     let out_str = match spec {
    //         Some(spec) => format!("{}<{}>", std::any::type_name::<Self>(), spec),
    //         None => format!("{}", std::any::type_name::<Self>()),
    //     };
    //     out_str
    // }

    // fn _get_spec_id(&self) -> Option<String>;

    // fn rs_random_seed(&self) -> u32;

    // // fn rs_random(&self) -> Generator;

    // /// Get the observation of the current state.
    // fn _get_obs(&self) -> ObsType;

    // /// Get additional information about the current state.
    // fn _get_info<T>(&self) -> HashMap<String, T>;

    // /// Set the PRNG
    // ///
    // /// # Arguments
    // /// * `rng` - A random number generator
    // fn _set_rs_random(&mut self, rng: Generator);

    // /// Set the PRNG seed
    // ///
    // /// # Arguments
    // /// * `seed` - A seed for the random number generator
    // fn _set_rs_random_seed(&mut self, seed: u32);

    // /// Assign a random number generator and seed to the environment.
    // ///
    // /// # Arguments
    // /// * `seed` - An optional seed for the random number generator
    // fn _assign_randomness(&mut self, seed: Option<u32>) {
    //     let (mut rng, rs_seed) = rs_random(seed);
    //     self._set_rs_random(rng);
    //     self._set_rs_random_seed(rs_seed);
    // }
}

// To do: Implement the Dynamics trait for the Env struct
// impl Dynamics for Env {}
