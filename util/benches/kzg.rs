use std::time::Instant;

use ark_bn254::Bn254;
use ark_ec::pairing::Pairing;
use ark_ff::UniformRand;
use rand::thread_rng;
use util::{
    kzg::{LOG_CHUNK_SIZE, Mkzg, MkzgProveParams, MkzgVerParams},
    poly::MlPoly,
};

fn main() {
    let mut rng = thread_rng();
    let (pp, vp) = Mkzg::<Bn254>::gen_srs(LOG_CHUNK_SIZE, &mut rng);
    bench_kzg(&pp, &vp, 20);
}

fn bench_kzg<E: Pairing>(pp: &MkzgProveParams<E>, vp: &MkzgVerParams<E>, nv: usize) {
    let mut rng = thread_rng();
    let pp = pp.trim(LOG_CHUNK_SIZE);
    let vp = vp.trim(LOG_CHUNK_SIZE);
    let poly = MlPoly(
        (0..1 << nv)
            .map(|_| <E::ScalarField as UniformRand>::rand(&mut rng))
            .collect(),
    );
    let point = (0..nv)
        .map(|_| <E::ScalarField as UniformRand>::rand(&mut rng))
        .collect::<Vec<_>>();
    let start = Instant::now();
    let commit = Mkzg::commit(&pp, &poly);
    let (proof, value) = Mkzg::open(&pp, poly.clone(), point.clone());
    let prover_time = start.elapsed().as_millis();
    let start = Instant::now();
    assert!(Mkzg::verify(&vp, point, &commit, value, proof));
    let verifier_time = start.elapsed().as_millis();
    println!(
        "nv = {}, prover time = {}, verifier_time = {}",
        nv, prover_time, verifier_time
    );
}
