use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

pub type Generator = Xoshiro256Plus;

/// Generate a random number generator and a seed.
/// If a seed is not provided, the seed is generated from the OS entropy.
///
/// # Arguments
/// * `seed` - An optional seed for the random number generator
///
/// # Returns
/// * `rng` - A random number generator
/// * `rs_seed` - The seed used to generate `rng``
pub fn rs_random(seed: Option<u32>) -> (Generator, u32) {
    // if seed is None, used the entropy from the OS
    let rs_seed: u32 = match seed {
        Some(seed) => seed,
        None => rand::random(),
    };

    let mut rng: Generator = rand_xoshiro::Xoshiro256Plus::seed_from_u64(rs_seed as u64);

    // Return the seed and random generator
    (rng, rs_seed)
}
