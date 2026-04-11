use ark_ff::Field;
use ark_serialize::CanonicalSerialize;
use rs_merkle::{Hasher, MerkleProof, MerkleTree};

#[derive(Debug, Clone)]
pub struct Blake32 {}

impl Hasher for Blake32 {
    type Hash = [u8; 32];

    fn hash(data: &[u8]) -> [u8; 32] {
        blake3::hash(data).as_bytes().clone()
    }
}

#[derive(Clone)]
pub struct MerkleTreeProver {
    pub merkle_tree: MerkleTree<Blake32>,
    leave_num: usize,
}

pub struct Serialize;

impl Serialize {
    pub fn serialize_fields<F: Field>(v: &[F]) -> Vec<u8> {
        let mut bytes = vec![];
        v.iter()
            .for_each(|x| <F as CanonicalSerialize>::serialize_compressed(&x, &mut bytes).unwrap());
        bytes
    }
}

#[derive(Debug, Clone)]
pub struct MerkleTreeVerifier {
    pub merkle_root: <Blake32 as Hasher>::Hash,
    pub leave_number: usize,
}

impl MerkleTreeProver {
    pub fn new(leaf_values: &Vec<Vec<u8>>) -> Self {
        let leaves = leaf_values
            .iter()
            .map(|x| Blake32::hash(x))
            .collect::<Vec<_>>();
        let merkle_tree = MerkleTree::<Blake32>::from_leaves(&leaves);
        Self {
            merkle_tree,
            leave_num: leaf_values.len(),
        }
    }

    pub fn leave_num(&self) -> usize {
        self.leave_num
    }

    pub fn commit(&self) -> <Blake32 as Hasher>::Hash {
        self.merkle_tree.root().unwrap()
    }

    pub fn open(&self, leaf_indices: &[usize]) -> Vec<u8> {
        self.merkle_tree.proof(leaf_indices).to_bytes()
    }
}

impl MerkleTreeVerifier {
    pub fn new(leave_number: usize, merkle_root: &<Blake32 as Hasher>::Hash) -> Self {
        Self {
            leave_number,
            merkle_root: merkle_root.clone(),
        }
    }

    pub fn verify(
        &self,
        proof_bytes: Vec<u8>,
        indices: &Vec<usize>,
        leaves: &Vec<Vec<u8>>,
    ) -> bool {
        let proof = MerkleProof::<Blake32>::try_from(proof_bytes).unwrap();
        let leaves_to_prove = leaves.iter().map(|x| Blake32::hash(x)).collect::<Vec<_>>();
        proof.verify(
            self.merkle_root,
            indices,
            &leaves_to_prove,
            self.leave_number,
        )
    }

    pub fn get_root(
        proof_bytes: Vec<u8>,
        index: usize,
        leaf: Vec<u8>,
        leave_number: usize,
    ) -> <Blake32 as Hasher>::Hash {
        let proof = MerkleProof::<Blake32>::try_from(proof_bytes).unwrap();
        let leaf_hashes = vec![Blake32::hash(&leaf)];
        proof
            .root(&vec![index], &leaf_hashes, leave_number)
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_ff::{Field, UniformRand};
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use rand::thread_rng;

    use super::*;

    #[test]
    fn commit_and_open() {
        let leaf_values = (0..8)
            .map(|x| Serialize::serialize_fields(&[Fr::from(x * 2), Fr::from(x * 2 + 1)]))
            .collect::<Vec<_>>();
        let leave_number = leaf_values.len();
        let prover = MerkleTreeProver::new(&leaf_values);
        let root = prover.commit();
        let verifier = MerkleTreeVerifier::new(leave_number, &root);
        let leaf_indices = vec![2, 3];
        let proof_bytes = prover.open(&leaf_indices);
        let open_values = vec![
            Serialize::serialize_fields(&[Fr::from(2 * 2), Fr::from(2 * 2 + 1)]),
            Serialize::serialize_fields(&[Fr::from(3 * 2), Fr::from(3 * 2 + 1)]),
        ];
        assert!(verifier.verify(proof_bytes, &leaf_indices, &open_values));
    }

    #[test]
    fn serialize() {
        let mut rng = thread_rng();
        for _ in 0..100 {
            let x = <Fr as UniformRand>::rand(&mut rng);
            let mut bytes = vec![];
            <Fr as CanonicalSerialize>::serialize_compressed(&x, &mut bytes).unwrap();
            let y = <Fr as CanonicalDeserialize>::deserialize_compressed(bytes.as_slice()).unwrap();
            assert_eq!(x, y);
        }

        for _ in 0..10000 {
            let bytes = (0..30).map(|_| rand::random::<u8>()).collect::<Vec<_>>();
            let _ = <Fr as Field>::from_random_bytes(&bytes).unwrap();
        }
    }
}
