use crate::{tensor::Device, utils::seeding::Generator};
/// Struct that is used to define observation and action spaces.
#[derive(Debug, Clone)]
pub struct Spacial {
    pub shape: Option<Vec<u32>>,
    pub dtype: Option<String>,
    pub rs_random: Generator,
    pub device: Option<Device>,
}

/// Spaces are crucially used in Gym to define the format of valid actions and observations.
/// They serve various purposes:
///
/// * They clearly define how to interact with environments, i.e. they specify what actions need to look like
///   and what observations will look like
/// * They allow us to work with highly structured data (e.g. in the form of elements of :class:`Dict` spaces)
///   and painlessly transform them into flat arrays that can be used in learning code
/// * They provide a method to sample random elements. This is especially useful for exploration and debugging.
///
/// Different spaces can be combined hierarchically via container spaces (:class:`Tuple` and :class:`Dict`) to build a more expressive space.
///
/// Warning:
///     Custom observation & action spaces can inherit from the ``Space`` class.
///     However, most use-cases should be covered by the existing space classes (e.g. :class:`Box`, :class:`Discrete`, etc...), and container classes (:class:`Tuple` & :class:`Dict`).
///     Note that parametrized probability distributions (through the :meth:`Space.sample()` method), and batching functions (in :class:`gym.vector.VectorEnv`), are only well-defined for instances of spaces provided in gym by default.
///     Moreover, some implementations of Reinforcement Learning algorithms might not handle custom spaces properly. Use custom spaces with care.
pub trait Space<DType> {
    fn is_flattenable(&self) -> bool;
    fn sample<Mask>(&self, mask: Option<Mask>) -> DType;
    fn seed(&mut self, seed: Option<u32>) -> Vec<u32>;
    fn contains<T>(&self, x: T) -> bool;
}

pub enum Bound {
    F64,
    Tensor,
}
