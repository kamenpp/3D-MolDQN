from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os

from absl import app
from absl import flags

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import run_dqn_rxn3Ddqn_modified as run_dqn
import deep_q_networks_rxn3Ddqn as deep_q_networks
import molecules_rxn3Ddqn as molecules_mdp
import core
import pickle

FLAGS = flags.FLAGS


class MultiObjectiveRewardMolecule(molecules_mdp.Molecule):
    """Defines the subclass of generating a molecule with a specific reward.
  The reward is defined as a scalar
    reward = weight * similarity_score + (1 - weight) *  qed_score
  """

    def __init__(self, target_molecule, similarity_weight, discount_factor, weighting_vec,
                 **kwargs):
        """Initializes the class.
    Args:
      target_molecule: SMILES string. The target molecule against which we
        calculate the similarity.
      similarity_weight: Float. The weight applied similarity_score.
      discount_factor: Float. The discount factor applied on reward.
      **kwargs: The keyword arguments passed to the parent class.
    """
        super(MultiObjectiveRewardMolecule, self).__init__(**kwargs)
        target_molecule = Chem.MolFromSmiles(target_molecule)
        self._target_mol_fingerprint = self.get_fingerprint(target_molecule)
        self._sim_weight = similarity_weight
        self._weighting_vec = weighting_vec
        self._discount_factor = discount_factor
        self.cache = {}

    def get_fingerprint(self, molecule):
        """Gets the morgan fingerprint of the target molecule.
    Args:
      molecule: Chem.Mol. The current molecule.
    Returns:
      rdkit.ExplicitBitVect. The fingerprint of the target.
    """
        return AllChem.GetMorganFingerprint(molecule, radius=2, useChirality=True)

    def get_similarity(self, smiles):
        """Gets the similarity between the current molecule and the target molecule.
    Args:
      smiles: String. The SMILES string for the current molecule.
    Returns:
      Float. The Tanimoto similarity.
    """

        structure = Chem.MolFromSmiles(smiles)
        if structure is None:
            return 0.0
        fingerprint_structure = self.get_fingerprint(structure)

        return DataStructs.TanimotoSimilarity(self._target_mol_fingerprint,
                                              fingerprint_structure)

    def _reward(self, hparams, episode, steps_left):
        """Calculates the reward of the current state.
    The reward is defined as a tuple of the similarity and QED value.
    Returns:
      A tuple of the similarity and qed value
    """
        # calculate similarity.
        # if the current molecule does not contain the scaffold of the target,
        # similarity is zero.
        if self._state is None:
            return 0.0
        mol = Chem.MolFromSmiles(self._state)
        if mol is None:
            return 0.0

        if FLAGS.cache == 'True':  # with cache -- to be changed to a boolean
            InChI_key = Chem.MolToInchiKey(mol)
            if InChI_key in self.cache.keys():
                similarity_score, docking_score, envelope_penalty, sucos, interactions_penalty, total_number_of_interactions, sascore, qed, logp, ligand_file_name, reward = \
                self.cache[InChI_key]
                print('Molecule is repeated ' + self._state + ' in episode ', episode, 'and steps left', steps_left)
            else:
                similarity_score = self.get_similarity(self._state)
                docking_score, envelope_penalty, sucos, interactions_penalty, total_number_of_interactions, sascore, qed, logp, ligand_file_name = run_dqn.run_docking(
                    self, episode, hparams, mol, steps_left, similarity_score)
                if docking_score < 0.:
                    docking_score = 0.
                if envelope_penalty < -100.:
                    envelope_penalty = -100.
                if interactions_penalty < -25.:
                    interactions_penalty = -25.
                reward = 0.5 * (docking_score + envelope_penalty)

                self.cache[InChI_key] = (similarity_score, docking_score, envelope_penalty, sucos, interactions_penalty,
                                         total_number_of_interactions, sascore, qed, logp, ligand_file_name, reward)
        else:  # without cache
            similarity_score = self.get_similarity(self._state)
            docking_score, envelope_penalty, sucos, interactions_penalty, total_number_of_interactions, sascore, qed, logp, ligand_file_name = run_dqn.run_docking(
                self, episode, hparams, mol, steps_left, similarity_score)
            if docking_score < 0.:
                docking_score = 0.
            if envelope_penalty < -100.:
                envelope_penalty = -100.
            if interactions_penalty < -25.:
                interactions_penalty = -25.
            reward = 0.5 * (docking_score + envelope_penalty)
        discount = self._discount_factor ** (self.max_steps - self._counter)
        reward_discount = reward * discount
        num_heavy_atoms = float(Chem.Mol.GetNumHeavyAtoms(mol))
        normalised_interactions_penalty = interactions_penalty/total_number_of_interactions
        print('Episode: ' + str(episode) + ' steps left ' + str(steps_left) + ' the similarity is ' + str(
            similarity_score) +
              ' the docking score is ' + str(docking_score) + ' the penalty is ' +
              str(envelope_penalty) + ' sucos is ' + str(sucos) +
              ' interactions_penalty ' + str(interactions_penalty) + ' total num interactions ' + str(
            total_number_of_interactions) +
              ' the normalised interactions penalty ' + str(normalised_interactions_penalty) + ' the sascore is ' + str(
            sascore) + ' ligand efficiency is ' + str(
            docking_score / num_heavy_atoms) + ' penalised_negative_docking_score is ' + str(
            docking_score - sascore) +
              ' docking_env_penalty is ' + str(0.5 * (docking_score + envelope_penalty)) +
              ' the reward is ' + str(reward) + ' the discounted reward is ' + str(reward_discount) +
              ' the smiles ' + self._state + ' the qed is ' + str(qed) + ' the logp is ' + str(
            logp) + ' ligand file name is ' + ligand_file_name)

        return reward_discount


def main(argv):
    del argv  # unused.
    if FLAGS.hparams is not None:
        with open(FLAGS.hparams, 'r') as f:
            hparams = deep_q_networks.get_hparams(**json.load(f))
    else:
        hparams = deep_q_networks.get_hparams()

    environment = MultiObjectiveRewardMolecule(
        target_molecule=FLAGS.target_molecule,
        similarity_weight=FLAGS.similarity_weight,
        discount_factor=hparams.discount_factor,
        weighting_vec=FLAGS.weighting_vec,
        atom_types=set(hparams.atom_types),
        init_mol=FLAGS.start_molecule,
        allow_removal=hparams.allow_removal,
        allow_no_modification=hparams.allow_no_modification,
        allow_bonds_between_rings=hparams.allow_bonds_between_rings,
        allowed_ring_sizes=set(hparams.allowed_ring_sizes),
        max_steps=hparams.max_steps_per_episode)

    dqn = deep_q_networks.DeepQNetwork(
        input_shape=(hparams.batch_size, hparams.fingerprint_length + 1),  # + 1 because dqn gets info on the steps left
        q_fn=functools.partial(
            deep_q_networks.multi_layer_model, hparams=hparams),
        optimizer=hparams.optimizer,
        grad_clipping=hparams.grad_clipping,
        num_bootstrap_heads=hparams.num_bootstrap_heads,
        gamma=hparams.gamma,
        epsilon=1.0)
    td_errors = []
    rewards = []

    run_dqn.run_training(hparams=hparams, environment=environment, dqn=dqn, td_errors=td_errors, rewards=rewards)
    # SAVE THE TD_ERRORS AND REWARDS IN A FILE
    core.write_hparams(hparams, os.path.join(FLAGS.model_dir, 'config_multiobj_kamen_mol3Ddqn.json'))
    with open(FLAGS.model_dir + '/errors.txt', 'wb') as fp:
        pickle.dump(td_errors, fp)
        print('Done writing list into a binary file')

    with open(FLAGS.model_dir + '/rewards.txt', 'wb') as fp2:
        pickle.dump(rewards, fp2)
        print('Done writing list into a binary file')


if __name__ == '__main__':
    app.run(main)
