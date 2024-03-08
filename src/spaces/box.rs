//! Implementation of a space that represents closed boxes in euclidean space.
use crate::spaces::space::Space;
use crate::utils::seeding::{rs_random, Generator};

/// Struct that is used to define observation and action spaces.
#[derive(Debug, Clone)]
pub struct Box {
    pub shape: Option<Vec<u32>>,
    pub dtype: Option<String>,
    pub rs_random: Generator,
}

impl<DType> Space<DType> for Box {
    fn is_flattenable(&self) -> bool {
        true
    }

    fn sample<T>(&self, mask: Option<T>) -> DType {
        todo!()
    }

    fn seed(&mut self, seed: Option<u32>) -> Vec<u32> {
        let rs_seed;
        (self.rs_random, rs_seed) = rs_random(seed);

        // Return the seed in a vec
        vec![rs_seed]
    }

    fn contains<T>(&self, x: T) -> bool {
        true
    }
}
