use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

type Generator = Xoshiro256Plus;

pub fn rs_random(seed: Option<u32>) -> (Generator, u32) {
    // if seed is None, used the entropy from the OS
    let rs_seed: u32 = match seed {
        Some(seed) => seed,
        None => rand::random(),
    };

    let mut rng = rand_xoshiro::Xoshiro256Plus::seed_from_u64(rs_seed as u64);

    // Return the seed and random generator
    (rng, rs_seed)
}
