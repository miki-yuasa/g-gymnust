#[derive(Debug, Clone)]
pub struct WrapperSpec<WrapperSpecArgs> {
    name: String,
    entry_point: String,
    kwargs: Option<WrapperSpecArgs>,
}

#[derive(Debug, Clone)]
pub struct EnvSpec<EnvSpecArgs, WrapperSpecArgs> {
    pub id: String,
    pub entry_point: String,
    // Environment attributes
    pub reward_threshold: Option<f64>,
    pub nondeterministic: bool,
    // Wrapper
    pub max_episode_steps: Option<usize>,
    pub order_enforce: bool,
    pub disable_env_checker: bool,
    // Environment arguments
    pub kwargs: Option<EnvSpecArgs>,
    // Post-init attributes
    pub namespace: Option<String>,
    pub name: String,
    pub version: Option<usize>,
    // Applied wrappers
    pub applied_wrappers: Option<Vec<WrapperSpec<WrapperSpecArgs>>>,
    // Todo: implement VectorEnvCreator
    // vector_entry_point
}

impl<EnvSpecArgs, WrapperSpecArgs> EnvSpec<EnvSpecArgs, WrapperSpecArgs> {
    pub fn to_string(&self) -> String {
        let out_str = format!("{}<{}>", std::any::type_name::<Self>(), self.id);
        out_str
    }
}
