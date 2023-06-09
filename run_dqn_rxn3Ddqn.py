#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Created on Thu Oct 17 16:14:53 2019

# @author: goto
# """

# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Executor for deep Q network models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os
import time
import sys  # added for docking
import subprocess  # added for docking

from absl import app
from absl import flags
from absl import logging
import pandas as pd
# from baselines.common import schedules
from stable_baselines.common import schedules
# from baselines.deepq import replay_buffer
from stable_baselines.common import buffers  # replay_buffer

import numpy as np

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

import tensorflow as tf
from tensorflow import gfile

# from mol_dqn.chemgraph.dqn import deep_q_networks
# from mol_dqn.chemgraph.dqn import molecules as molecules_mdp
# from mol_dqn.chemgraph.dqn.py import molecules
# from mol_dqn.chemgraph.dqn.tensorflow_core import core
import deep_q_networks_rxn3Ddqn as deep_q_networks
# import deep_q_networks
# import molecules1 as molecules
import core

# add more below or create a dictionary so that the users could control what kind of reactions they want to perform, etc.
#flags.DEFINE_string('fpath', None, 'Filename for starting drug molecule.')
#flags.DEFINE_string('fpath1', None, 'Filename for reagents for acyl halide.')
#flags.DEFINE_string('fpath2', None, 'Filename for reagents for amine.')
#flags.DEFINE_string('fpath3', None, 'Name of model for docking intermediates.')
#flags.DEFINE_string('fpath4', None, 'Name of the starting material for docking intermediates.')
#flags.DEFINE_string('fpath5', None, 'Directory to place the docking intermediates.')
#flags.DEFINE_string('fpath6', None, 'Path to AutoDock Vina.')
#flags.DEFINE_string('fpath7', None, 'Path to pdbqt file for receptor.')
#flags.DEFINE_float('fpath8', None, 'Value of center_x for AutoDock Vina.')
#flags.DEFINE_float('fpath9', None, 'Value of center_y for AutoDock Vina.')
#flags.DEFINE_float('fpath10', None, 'Value of center_z for AutoDock Vina.')
#flags.DEFINE_string('fpath11', None, 'Value of size_x for AutoDock Vina.')
#flags.DEFINE_string('fpath12', None, 'Value of size_y for AutoDock Vina.')
#flags.DEFINE_string('fpath13', None, 'Value of size_z for AutoDock Vina.')
#flags.DEFINE_string('fpath14', None, 'Value of exhaustiveness for AutoDock Vina.')
#flags.DEFINE_string('cache', None, 'Specify whether cache is True or False.')
#flags.DEFINE_string('SMARTS', None, 'SMARTS expression for fragment masking.')
flags.DEFINE_string('model_dir',
                    # '/namespace/gas/primary/zzp/dqn/r=3/exp2_bs_dqn',
                    # '/data/hoodedcrow/goto/MolDQN',
                    None,
                    'The directory to save data to.')
flags.DEFINE_string('target_molecule', 'C1CCC2CCCCC2C1',
                    'The SMILES string of the target molecule.')
flags.DEFINE_string('start_molecule', None,
                    'The SMILES string of the start molecule.')
flags.DEFINE_float(
    'similarity_weight', 0.5,
    'The weight of the similarity score in the reward function.')
flags.DEFINE_float('target_weight', None,
                   'The target molecular weight of the molecule.')
flags.DEFINE_string('hparams', None, 'Filename for serialized HParams.')
flags.DEFINE_boolean('multi_objective', True,
                     'Whether to run multi objective DQN.')
flags.DEFINE_string('docking', None, '')
flags.DEFINE_string('smarts_mask', None, 'SMARTS expression for fragment masking.')
flags.DEFINE_string('cache', None, 'Specify whether cache is True or False.')

FLAGS = flags.FLAGS

import molecules_rxn3Ddqn as molecules_mdp
from SA_Score import sascorer
from rdkit.Chem import rdDistGeom
import molecules1_rxn3Ddqn as molecules
from scipy import spatial
from rdkit.Chem import QED
from pymol2.cmd2 import global_cmd as pymol_cmd
from plip.structure.preparation import PDBComplex
from plip.exchange.report import BindingSiteReport

envelope = pd.read_csv('../utils/celecoxib_envelope_02_with_01_resolution.csv', )
kd_tree_envelope = spatial.cKDTree(envelope)

def retrieve_plip_interactions(pdb_file):
    """
    Retrieves the interactions from PLIP.

    Parameters
    ----------
    pdb_file :
        The PDB file of the complex.

    Returns
    -------
    dict :
        A dictionary of the binding sites and the interactions.
    """
    protlig = PDBComplex()
    protlig.load_pdb(pdb_file)  # load the pdb file
    for ligand in protlig.ligands:
        protlig.characterize_complex(ligand)  # find ligands and analyze interactions
    sites = {}
    # loop over binding sites
    interaction_sites = []
    for key, site in sorted(protlig.interaction_sets.items()):
        print(key)
        interaction_sites.append(key) # KP since there is only one site in our case
        # keep track of that site's key for convenience
        binding_site = BindingSiteReport(site)  # collect data about interactions
        # tuples of *_features and *_info will be converted to pandas DataFrame
        keys = (
            "hydrophobic",
            "hbond",
            "waterbridge",
            "saltbridge",
            "pistacking",
            "pication",
            "halogen",
            "metal",
        )
        # interactions is a dictionary which contains relevant information for each
        # of the possible interactions: hydrophobic, hbond, etc. in the considered
        # binding site. Each interaction contains a list with
        # 1. the features of that interaction, e.g. for hydrophobic:
        # ('RESNR', 'RESTYPE', ..., 'LIGCOO', 'PROTCOO')
        # 2. information for each of these features, e.g. for hydrophobic
        # (residue nb, residue type,..., ligand atom 3D coord., protein atom 3D coord.)
        interactions = {
            k: [getattr(binding_site, k + "_features")] + getattr(binding_site, k + "_info")
            for k in keys
        }
        sites[key] = interactions
    return sites, interaction_sites[0]

def run_docking(self, episode, hparams, molecule, steps_left, similarity_score):
    """Save the PDBQT format of the target molecule.
    Args:
      molecule: Chem.Mol. The current molecule.
    Returns: 
      String file. PDBQT file of the current molecule.
    """
    with open(FLAGS.docking, 'r') as docking_file:
        json_str = docking_file.read()

    docking = json.loads(json_str)
    docking_file.close()
    fpath3 = docking["naming1"]
    fpath4 = docking["naming2"]
    fpath5 = docking["file_path_pdbqt"]
    fpath6 = docking["vina"]
    fpath7 = docking["protein_pdbqt"]
    fpath8_stripped = 14.469
    fpath9_stripped = 0.268
    fpath10_stripped = 15.256
    ligand_for_sucos = docking["for_SuCOS"]

    ##ligand_file_name = "ligand_%s_%04d_%s_%s.pdbqt" % (episode, self.num_steps_taken, fpath3, fpath4)
    if steps_left == 1:
        ligand_file_name = "episode_outs/ligand_%s_%04d_%s_%s.pdbqt" % (episode, self.num_steps_taken, fpath3, fpath4)
    else:
        ligand_file_name = "ligand_%s_%s_%04d_%s_%s.pdbqt" % (episode, hparams.num_episodes, self.num_steps_taken, fpath3, fpath4)
        # ligand_file_name = "ligand_%s_%04d_%s_%s.pdbqt" % (episode, self.num_steps_taken, fpath3, fpath4)
    ligand_file_path = fpath5 + ligand_file_name
    sascore = sascorer.calculateScore(molecule)
    #sdf_writer = Chem.SDWriter(ligand_file_path[:-6] + '.sdf')
    try:
        molecule = AllChem.AddHs(molecule)
        ps = rdDistGeom.ETKDG()
        res = rdDistGeom.EmbedMolecule(molecule, ps)

        #sdf_writer.write(molecule)
        #sdf_writer.close()

        #pymol_cmd.load(ligand_file_path[:-6] + '.sdf', object='celecoxib')
        #pymol_cmd.save(ligand_file_path[:-2], 'celecoxib')
        #pymol_cmd.reinitialize()
        #AllChem.EmbedMolecule(molecule)
        if res != -1:
            Chem.MolToPDBFile(molecule, ligand_file_path[:-2])
        #ligand_pdb = molecules.Mol_into_PDB_ligand(molecule)
        #with open(ligand_file_path[:-2], 'w') as f:
        #    f.write(ligand_pdb + os.linesep)

            _ = subprocess.getoutput(
                "python %s -l %s -o %s -R %r " % (
                    '/Users/kamen/Dev/AutoDockTools_py3/prep_ligand4.py', ligand_file_path[:-2], ligand_file_path, 0))  # just taking the first atom as the root because it shouldn't matter
            print('output from prep_ligand4 is ', _)
    except:
        #sdf_writer.close()
        print('failed to prepare molecule ', Chem.MolToSmiles(molecule), ' with synthetic accessibility ', sascore, ' episode ', episode, steps_left)

    try:
        docking_output = subprocess.getoutput(
            "%s --receptor=%s --ligand=%s --center_x=%r --center_y=%r --center_z=%r --size_x=%s --size_y=%s --size_z=%s --exhaustiveness=%s --out=%s --seed 42| grep -H '^   1' | awk '{print $4}'" % (
                fpath6, fpath7, ligand_file_path, fpath8_stripped, fpath9_stripped, fpath10_stripped, 25, 25,
                25, 1, ligand_file_path[:-6] + '_out.pdbqt'))
    except:
        docking_output = 0.
        print('failed to dock molecule ', Chem.MolToSmiles(molecule), ' with synthetic accessibility ', sascore, ' episode ', episode, steps_left)
    sucos_score = ''
    penalty = 0.

    try:
        with open(ligand_file_path[:-6]+'_out.pdbqt', 'r') as f:
            xs, ys, zs = [], [], []
            for line in f.readlines():
                if line == 'ENDMDL\n':
                    break
                line_li = line.split()
                if len(line_li) == 12:
                    xs.append(float(line_li[5]))
                    ys.append(float(line_li[6]))
                    zs.append(float(line_li[7]))
        f.close()

        ligand = pd.DataFrame(data={'x':xs, 'y':ys, 'z':zs})  # probably can skip the creation of a df object
        penalty = 0.
        for i in range(len(ligand)):
            crd = np.array(ligand.iloc[i])
            # this will return the ligand atoms that are within r from at least 1 point in the pocket
            dist_to_neighbour, _ = kd_tree_envelope.query(crd, k=1)
            if dist_to_neighbour < 1.:
                continue
            elif dist_to_neighbour < 2.:
                penalty -= dist_to_neighbour / 2
            else:
                penalty -= dist_to_neighbour

        _ = subprocess.getoutput(
            "%s %s -O %s" % ('obabel', ligand_file_path[:-6] + '_out.pdbqt', ligand_file_path[:-6] + '_out.sdf'))

        #sucos_score = ''
        #try:
        sucos_score = subprocess.getoutput(
            "%s %s --lig1 %s --lig2 %s | grep 'SuCOS score:' | awk '{print $3}' | head -1" % (
            'python', '../utils/calc_SuCOS_normalized.py', ligand_for_sucos,
            ligand_file_path[:-6] + '_out.sdf'))
        sucos_score = float(sucos_score)
    except:
        print('SuCOS failed! because ',sucos_score)
        sucos_score = 0.

    try:
        pymol_cmd.load(ligand_file_path[:-6] + '_out.pdbqt', object='docked_ligand')
        pymol_cmd.load(fpath7, object='1oq5')
        pymol_cmd.create('lig_and_prot', 'all')
        pymol_cmd.save(fpath5 + 'combined.pdb', selection='lig_and_prot')
        pymol_cmd.reinitialize()

        interactions_by_site, site = retrieve_plip_interactions(fpath5 + 'combined.pdb')
        keys = (
            "hydrophobic",
            "hbond",
            "waterbridge",
            "saltbridge",
            "pistacking",
            "pication",
            "halogen",
            "metal",
        )
        hydrophobic_interacting_resis_flexres = [67, 121, 131, 198, 198]
        hbond_interacting_resis_flexres = [199, 94]
        pistacking_interacting_resis_flexres = {131}
        #site = 'UNL:Z:1'  # because of the Zn there are two sites
        # but this is the correct name every time anyway

        interactions_penalty = 0
        total_number_of_interactions = 0
        for key in keys:
            # subtracting one because the first one is the name of the columns
            total_number_of_interactions += len(interactions_by_site[site][key]) - 1
            if key == "hydrophobic":
                # the number of hydrophobic interactions for nirmatrelvir-binding site is 2
                # and 1 is subtracted for the indices in the dictionary
                hydrophobic_interacting_resis_ligand = []
                if len(interactions_by_site[site][key]) > 1:
                    for row in interactions_by_site[site][key][1:]:
                        hydrophobic_interacting_resis_ligand.append(row[0])
                    while len(hydrophobic_interacting_resis_flexres) != 0:
                        if hydrophobic_interacting_resis_flexres[0] not in hydrophobic_interacting_resis_ligand:
                            interactions_penalty -= 1
                            hydrophobic_interacting_resis_flexres.remove(hydrophobic_interacting_resis_flexres[0])
                        elif hydrophobic_interacting_resis_flexres[0] in hydrophobic_interacting_resis_ligand:
                            hydrophobic_interacting_resis_ligand.remove(hydrophobic_interacting_resis_flexres[0])
                            hydrophobic_interacting_resis_flexres.remove(hydrophobic_interacting_resis_flexres[0])
                    interactions_penalty -= len(hydrophobic_interacting_resis_ligand)
                else:
                    interactions_penalty -= 4
            elif key == "hbond":
                hbond_interacting_resis_ligand = []
                if len(interactions_by_site[site][key]) > 1:
                    for row in interactions_by_site[site][key][1:]:
                        hbond_interacting_resis_ligand.append(row[0])
                    while len(hbond_interacting_resis_flexres) != 0:
                        if hbond_interacting_resis_flexres[0] not in hbond_interacting_resis_ligand:
                            interactions_penalty -= 1
                            hbond_interacting_resis_flexres.remove(hbond_interacting_resis_flexres[0])
                        elif hbond_interacting_resis_flexres[0] in hbond_interacting_resis_ligand:
                            hbond_interacting_resis_ligand.remove(hbond_interacting_resis_flexres[0])
                            hbond_interacting_resis_flexres.remove(hbond_interacting_resis_flexres[0])
                    interactions_penalty -= len(hbond_interacting_resis_ligand)
                else:
                    interactions_penalty -= 3
                # interactions_penalty -= abs(len(interactions_by_site[site][key]) - 9)
            elif key == "pistacking":
                pistacking_interacting_resis_ligand = set()
                if len(interactions_by_site[site][key]) > 1:
                    for row in interactions_by_site[site][key][1:]:
                        pistacking_interacting_resis_ligand.update([row[0]])
                    interactions_penalty -= len(
                        pistacking_interacting_resis_ligand ^ pistacking_interacting_resis_flexres)
                else:
                    interactions_penalty -= 2
                    # interactions_penalty -= abs(len(interactions_by_site[site][key]) - 2)
            else:
                # the number of all other interactions in nirmatrelvir-binding site is 0
                interactions_penalty -= abs(len(interactions_by_site[site][key]) - 1)

        #subprocess.getoutput("rm " + ligand_file_path)
        f_docking_score = (-1.0) * float(docking_output)
    except:
        print('IT IS THROWING SOME EXCEPTION for molecule with sascore ',sascore, ' smiles ', Chem.MolToSmiles(molecule), ' episode ', episode, steps_left)
        total_number_of_interactions = 1
        f_docking_score = 0.0
        sucos_score = 0.
        interactions_penalty = -25.
        penalty = -50.  # basically means the molecule is regarded as good
        # so it will be put to the buffer with priority, hence will be replayed and hopefully
        # the second docking will be successful
        #else:
        #print('The SAScore was ', sascore, ' so the molecule was not docked in ', episode, steps_left, Chem.MolToSmiles(molecule))
        #f_docking_score = 0.0
        #sucos_score = 0.
        #interactions_penalty = -25.
        #penalty = -50.

    qed = QED.qed(molecule)
    logp = molecules.penalized_logp(molecule)

    return f_docking_score, penalty, float(sucos_score), interactions_penalty, total_number_of_interactions,qed, logp, ligand_file_name


class TargetWeightMolecule(molecules_mdp.Molecule):
    """Defines the subclass of a molecule MDP with a target molecular weight."""

    def __init__(self, target_weight, **kwargs):  # same for new action space
        """Initializes the class.
    Args:
      target_weight: Float. the target molecular weight.
      **kwargs: The keyword arguments passed to the parent class.
    """
        super(TargetWeightMolecule, self).__init__(**kwargs)
        self.target_weight = target_weight

    def _reward(self):
        """Calculates the reward of the current state.
    The reward is defined as the negative l1 distance between the current
    molecular weight and target molecular weight range.
    Returns:
      Float. The negative distance.
    """
        print('*' * 70)
        print(self._state)
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return -self.target_weight ** 2
        if FLAGS.cache == 'True':  # with cache
            cache = dict()
            InChI_key = Chem.MolToInchiKey(molecule)
            if cache is not None and InChI_key in cache.keys():
                return cache[InChI_key]
            value = []
            value.append(self._state)
            if cache is not None:
                cache[InChI_key] = value
                lower, upper = self.target_weight - 25, self.target_weight + 25
                mw = Descriptors.MolWt(molecule)
                if lower <= mw <= upper:
                    return 1
                return -min(abs(lower - mw), abs(upper - mw))
        if FLAGS.cache == 'False':  # without cache
            lower, upper = self.target_weight - 25, self.target_weight + 25
            mw = Descriptors.MolWt(molecule)
            if lower <= mw <= upper:
                return 1
            return -min(abs(lower - mw), abs(upper - mw))


class MultiObjectiveRewardMolecule(molecules_mdp.Molecule):
    """Defines the subclass of generating a molecule with a specific reward.
  The reward is defined as a 1-D vector with 2 entries: similarity and QED
    reward = (similarity_score, qed_score)
  """

    def __init__(self, target_molecule, **kwargs):  # same for new action space
        """Initializes the class.
    Args:
      target_molecule: SMILES string. the target molecule against which we
        calculate the similarity.
      **kwargs: The keyword arguments passed to the parent class.
    """
        super(MultiObjectiveRewardMolecule, self).__init__(**kwargs)
        target_molecule = Chem.MolFromSmiles(target_molecule)
        self._target_mol_fingerprint = self.get_fingerprint(target_molecule)
        self._target_mol_scaffold = molecules.get_scaffold(target_molecule)
        self.reward_dim = 2

    def get_fingerprint(self, molecule):  # same for new action space
        """Gets the morgan fingerprint of the target molecule.
    Args:
      molecule: Chem.Mol. The current molecule.
    Returns:
      rdkit.ExplicitBitVect. The fingerprint of the target.
    """
        return AllChem.GetMorganFingerprint(molecule, radius=2)

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

    def _reward(self, hparams):
        """Calculates the reward of the current state.
    The reward is defined as a tuple of the similarity and QED value.
    Returns:
      A tuple of the similarity and qed value
    """
        # calculate similarity.
        # if the current molecule does not contain the scaffold of the target,
        # similarity is zero.
        if self._state is None:
            return 0.0, 0.0
        mol = Chem.MolFromSmiles(self._state)
        if mol is None:
            return 0.0, 0.0
        if FLAGS.cache == 'True':  # with cache
            cache = dict()
            InChI_key = Chem.MolToInchiKey(mol)
            if cache is not None and InChI_key in cache.keys():
                return cache[InChI_key]
            value = []
            value.append(self._state)
            if cache is not None:
                cache[InChI_key] = value
                if molecules.contains_scaffold(mol, self._target_mol_scaffold):
                    similarity_score = self.get_similarity(self._state)
                else:
                    similarity_score = 0.0
                    # calculate QED
                docking_score = run_docking(self, hparams, mol)
                return similarity_score, docking_score
        if FLAGS.cache == 'False':  # without cache
            if molecules.contains_scaffold(mol, self._target_mol_scaffold):
                similarity_score = self.get_similarity(self._state)
            else:
                similarity_score = 0.0
            docking_score = run_docking(self, hparams, mol)
            return similarity_score, docking_score


'''
class MultiObjectiveRewardMolecule(molecules_mdp.Molecule):
  """Defines the subclass of generating a molecule with a specific reward.
  The reward is defined as a 1-D vector with 2 entries: similarity and QED
    reward = (similarity_score, qed_score)
  """

  def __init__(self, target_molecule, **kwargs): #same for new action space
    """Initializes the class.
    Args:
      target_molecule: SMILES string. the target molecule against which we
        calculate the similarity.
      **kwargs: The keyword arguments passed to the parent class.
    """
    super(MultiObjectiveRewardMolecule, self).__init__(**kwargs)
    target_molecule = Chem.MolFromSmiles(target_molecule)
    self._target_mol_fingerprint = self.get_fingerprint(target_molecule)
    self._target_mol_scaffold = molecules.get_scaffold(target_molecule)
    self.reward_dim = 2

  def get_fingerprint(self, molecule): #same for new action space
    """Gets the morgan fingerprint of the target molecule.
    Args:
      molecule: Chem.Mol. The current molecule.
    Returns:
      rdkit.ExplicitBitVect. The fingerprint of the target.
    """
    return AllChem.GetMorganFingerprint(molecule, radius=2)

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


  def _reward(self):
    """Calculates the reward of the current state.
    The reward is defined as a tuple of the similarity and QED value.
    Returns:
      A tuple of the similarity and qed value
    """
    # calculate similarity.
    # if the current molecule does not contain the scaffold of the target,
    # similarity is zero.
    if self._state is None:
      return 0.0, 0.0
    mol = Chem.MolFromSmiles(self._state)
    if mol is None:
      return 0.0, 0.0
    if molecules.contains_scaffold(mol, self._target_mol_scaffold):
      similarity_score = self.get_similarity(self._state)
    else:
      similarity_score = 0.0
    # calculate QED
    qed_value = QED.qed(mol)
    return similarity_score, qed_value
'''
'''
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
'''
'''
  def _reward(self):
    """Calculates the reward of the current state.
    The reward is defined as a tuple of the similarity and QED value.
    Returns:
      A tuple of the similarity and qed value
    """
    # calculate similarity.
    # if the current molecule does not contain the scaffold of the target,
    # similarity is zero.
    if self._state is None:
      return 0.0, 0.0
    mol = Chem.MolFromSmiles(self._state)
    if mol is None:
      return 0.0, 0.0
    if molecules.contains_scaffold(mol, self._target_mol_scaffold):
      similarity_score = self.get_similarity(self._state)
    else:
      similarity_score = 0.0
    # calculate QED
    qed_value = QED.qed(mol)
    return similarity_score, qed_value
'''


# TODO(zzp): use the tf.estimator interface.
def run_training(hparams, environment, dqn, td_errors, rewards):  # same for new action space
    """Runs the training procedure.
  Briefly, the agent runs the action network to get an action to take in
  the environment. The state transition and reward are stored in the memory.
  Periodically the agent samples a batch of samples from the memory to
  update(train) its Q network. Note that the Q network and the action network
  share the same set of parameters, so the action network is also updated by
  the samples of (state, action, next_state, reward) batches.
  Args:
    hparams: tf.contrib.training.HParams. The hyper parameters of the model.
    environment: molecules.Molecule. The environment to run on.
    dqn: An instance of the DeepQNetwork class.
  Returns:
    None
  """
    summary_writer = tf.compat.v1.summary.FileWriter(FLAGS.model_dir)
    tf.reset_default_graph()
    with tf.Session() as sess:
        dqn.build()
        model_saver = tf.train.Saver(max_to_keep=hparams.max_num_checkpoints)
        # The schedule for the epsilon in epsilon greedy policy.
        exploration = schedules.PiecewiseSchedule(
            [(0, 1.0), (int(hparams.num_episodes / 2), 0.1),
             (hparams.num_episodes, 0.01)],
            outside_value=0.01)
        if hparams.prioritized:
            memory = buffers.PrioritizedReplayBuffer(hparams.replay_buffer_size,
                                                     hparams.prioritized_alpha)
            beta_schedule = schedules.LinearSchedule(
                hparams.num_episodes, initial_p=hparams.prioritized_beta, final_p=0)
        else:
            memory = buffers.ReplayBuffer(hparams.replay_buffer_size)
            beta_schedule = None
        sess.run(tf.global_variables_initializer())
        sess.run(dqn.update_op)
        global_step = 0
        for episode in range(hparams.num_episodes):
            global_step, episode_td_errors, episode_rewards = _episode(
                environment=environment,
                dqn=dqn,
                memory=memory,
                episode=episode,
                global_step=global_step,
                hparams=hparams,
                summary_writer=summary_writer,
                exploration=exploration,
                beta_schedule=beta_schedule,
            )
            td_errors.append(episode_td_errors)
            rewards.append(episode_rewards)
            if (episode + 1) % hparams.update_frequency == 0:
                sess.run(dqn.update_op)
            if (episode + 1) % hparams.save_frequency == 0:
                model_saver.save(
                    sess,
                    os.path.join(FLAGS.model_dir, 'ckpt'),
                    global_step=global_step)

'''KP 2022
def run_eval(environment, hparams, dqn, checkpoint): # KP 2022, not used for now
    tf.compat.v1.reset_default_graph()
    with tf.Session() as sess:
        dqn.build()
        sess.run(tf.global_variables_initializer())
        sess.run(dqn.update_op)
        # print('before import ', tf.trainable_variables())
        saver = tf.compat.v1.train.import_meta_graph(checkpoint)
        saver.restore(sess=sess, save_path=tf.train.latest_checkpoint('./outputs/save'))

        for episode in range(hparams.num_episodes):
            global_step, episode_td_errors, episode_rewards = _episode_eval(
                environment=environment,
                dqn=dqn,
                memory=memory,
                episode=episode,
                global_step=global_step,
                hparams=hparams,
                summary_writer=summary_writer,
                exploration=exploration,
                beta_schedule=beta_schedule,
            )'''


def _episode_eval(environment, dqn, memory, episode, global_step, hparams,
                  summary_writer, exploration, beta_schedule):   # KP 2022
    episode_start_time = time.time()
    environment.initialize()
    '''
    if hparams.num_bootstrap_heads:
        head = np.random.randint(hparams.num_bootstrap_heads)
    else:
        head = 0'''
    head = 0
    for step in range(hparams.max_steps_per_episode):
        result = _step(
            environment=environment,
            dqn=dqn,
            memory=memory,
            episode=episode,
            hparams=hparams,
            exploration=exploration,
            head=head)
        if step == hparams.max_steps_per_episode - 1:
            episode_summary = dqn.log_result(result.state, result.reward)
            summary_writer.add_summary(episode_summary, global_step)
            logging.info('Episode %d/%d took %gs', episode + 1, hparams.num_episodes,
                         time.time() - episode_start_time)
            logging.info('SMILES: %s\n', result.state)
            # Use %s since reward can be a tuple or a float number.
            logging.info('The reward is: %s', str(result.reward))
        if (episode > min(50, hparams.num_episodes / 10)) and (
                global_step % hparams.learning_frequency == 0):
            if hparams.prioritized:
                (state_t, _, reward_t, state_tp1, done_mask, weight,
                 indices) = memory.sample(
                    hparams.batch_size, beta=beta_schedule.value(episode))
            else:
                (state_t, _, reward_t, state_tp1,
                 done_mask) = memory.sample(hparams.batch_size)
                weight = np.ones([reward_t.shape[0]])
            # np.atleast_2d cannot be used here because a new dimension will
            # be always added in the front and there is no way of changing this.
            if reward_t.ndim == 1:
                reward_t = np.expand_dims(reward_t, axis=1)
    return reward_t


def _episode(environment, dqn, memory, episode, global_step, hparams,
             summary_writer, exploration, beta_schedule):  # same for new action space
    """Runs a single episode.
  Args:
    environment: molecules.Molecule; the environment to run on.
    dqn: DeepQNetwork used for estimating rewards.
    memory: ReplayBuffer used to store observations and rewards.
    episode: Integer episode number.
    global_step: Integer global step; the total number of steps across all
      episodes.
    hparams: HParams.
    summary_writer: FileWriter used for writing Summary protos.
    exploration: Schedule used for exploration in the environment.
    beta_schedule: Schedule used for prioritized replay buffers.
  Returns:
    Updated global_step.
  """
    episode_td_errors, episode_rewards = [], []
    episode_start_time = time.time()
    environment.initialize()
    if hparams.num_bootstrap_heads:
        head = np.random.randint(hparams.num_bootstrap_heads)
    else:
        head = 0
    for step in range(hparams.max_steps_per_episode):
        result = _step(
            environment=environment,
            dqn=dqn,
            memory=memory,
            episode=episode,
            hparams=hparams,
            exploration=exploration,
            head=head)
        if step == hparams.max_steps_per_episode - 1:
            episode_summary = dqn.log_result(result.state, result.reward)
            summary_writer.add_summary(episode_summary, global_step)
            logging.info('Episode %d/%d took %gs', episode + 1, hparams.num_episodes,
                         time.time() - episode_start_time)
            logging.info('SMILES: %s\n', result.state)
            # Use %s since reward can be a tuple or a float number.
        if (episode > min(50, hparams.num_episodes / 10)) and (
                global_step % hparams.learning_frequency == 0):
            if hparams.prioritized:
                (state_t, _, reward_t, state_tp1, done_mask, weight,
                 indices) = memory.sample(
                    hparams.batch_size, beta=beta_schedule.value(episode))
            else:
                (state_t, _, reward_t, state_tp1,
                 done_mask) = memory.sample(hparams.batch_size)
                weight = np.ones([reward_t.shape[0]])
            # np.atleast_2d cannot be used here because a new dimension will
            # be always added in the front and there is no way of changing this.
            if reward_t.ndim == 1:
                reward_t = np.expand_dims(reward_t, axis=1)
            td_error, error_summary, _ = dqn.train(
                states=state_t,
                rewards=reward_t,
                next_states=state_tp1,
                done=np.expand_dims(done_mask, axis=1),
                weight=np.expand_dims(weight, axis=1))
            episode_td_errors.append(td_error)  # KP 2022
            episode_rewards.append(reward_t)  # KP 2022
            summary_writer.add_summary(error_summary, global_step)
            logging.info('Current TD error (from inside the episode function): %.4f', np.mean(np.abs(td_error)))
            if hparams.prioritized:
                memory.update_priorities(
                    indices,
                    np.abs(np.squeeze(td_error) + hparams.prioritized_epsilon).tolist())
        global_step += 1
    return global_step, episode_td_errors, episode_rewards  # KP 2022

def _step(environment, dqn, memory, episode, hparams, exploration, head):  # same for new action space
    """Runs a single step within an episode.
  Args:
    environment: molecules.Molecule; the environment to run on.
    dqn: DeepQNetwork used for estimating rewards.
    memory: ReplayBuffer used to store observations and rewards.
    episode: Integer episode number.
    hparams: HParams.
    exploration: Schedule used for exploration in the environment.
    head: Integer index of the DeepQNetwork head to use.
  Returns:
    molecules.Result object containing the result of the step.
  """
    # Compute the encoding for each valid action from the current state.
    steps_left = hparams.max_steps_per_episode - environment.num_steps_taken
    valid_actions = list(environment.get_valid_actions())
    observations = np.vstack([
        np.append(deep_q_networks.get_fingerprint(act, hparams), steps_left)
        for act in valid_actions
    ])
    action = valid_actions[dqn.get_action(
        observations, head=head, update_epsilon=exploration.value(episode))]
    action_t_fingerprint = np.append(
        deep_q_networks.get_fingerprint(action, hparams), steps_left)
    steps_left = hparams.max_steps_per_episode - environment.num_steps_taken
    result = environment.step(action, hparams, episode, steps_left)
    action_fingerprints = np.vstack([
        np.append(deep_q_networks.get_fingerprint(act, hparams), steps_left)
        for act in environment.get_valid_actions()
    ])
    # we store the fingerprint of the action in obs_t so action
    # does not matter here.
    memory.add(
        obs_t=action_t_fingerprint,
        action=0,
        reward=result.reward,
        obs_tp1=action_fingerprints,
        done=float(result.terminated))  # KP the action is assigned to 0 as it is not used later
    return result


def run_dqn(multi_objective=False):
    """Run the training of Deep Q Network algorithm.
  Args:
    multi_objective: Boolean. Whether to run the multiobjective DQN.
  """
    if FLAGS.hparams is not None:
        with gfile.Open(FLAGS.hparams, 'r') as f:
            hparams = deep_q_networks.get_hparams(**json.load(f))
    else:
        hparams = deep_q_networks.get_hparams()
    logging.info(
        'HParams:\n%s', '\n'.join([
            '\t%s: %s' % (key, value)
            for key, value in sorted(hparams.values().items())
        ]))

    # Changed this part for docking
    # TODO(zzp): merge single objective DQN to multi objective DQN.
    if multi_objective:
        environment = MultiObjectiveRewardMolecule(
            target_molecule=FLAGS.target_molecule,
            atom_types=set(hparams.atom_types),
            init_mol=FLAGS.start_molecule,
            allow_removal=hparams.allow_removal,
            allow_no_modification=hparams.allow_no_modification,
            allow_bonds_between_rings=False,
            allowed_ring_sizes={3, 4, 5, 6},
            max_steps=hparams.max_steps_per_episode)

        dqn = deep_q_networks.MultiObjectiveDeepQNetwork(
            objective_weight=np.array([[FLAGS.similarity_weight],
                                       [1 - FLAGS.similarity_weight]]),
            input_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
            q_fn=functools.partial(
                deep_q_networks.multi_layer_model, hparams=hparams),
            optimizer=hparams.optimizer,
            grad_clipping=hparams.grad_clipping,
            num_bootstrap_heads=hparams.num_bootstrap_heads,
            gamma=hparams.gamma,
            epsilon=1.0)
    else:
        environment = TargetWeightMolecule(
            target_weight=FLAGS.target_weight,
            atom_types=set(hparams.atom_types),
            init_mol=FLAGS.start_molecule,
            allow_removal=hparams.allow_removal,
            allow_no_modification=hparams.allow_no_modification,
            allow_bonds_between_rings=hparams.allow_bonds_between_rings,
            allowed_ring_sizes=set(hparams.allowed_ring_sizes),
            max_steps=hparams.max_steps_per_episode)

        dqn = deep_q_networks.DeepQNetwork(
            input_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
            q_fn=functools.partial(
                deep_q_networks.multi_layer_model, hparams=hparams),
            optimizer=hparams.optimizer,
            grad_clipping=hparams.grad_clipping,
            num_bootstrap_heads=hparams.num_bootstrap_heads,
            gamma=hparams.gamma,
            epsilon=1.0)

    run_training(
        hparams=hparams,
        environment=environment,
        dqn=dqn,
    )

    core.write_hparams(hparams, os.path.join(FLAGS.model_dir, 'config_rxn3Ddqn.json'))


'''
def run_dqn(multi_objective=False):
  """Run the training of Deep Q Network algorithm.
  Args:
    multi_objective: Boolean. Whether to run the multiobjective DQN.
  """
  if FLAGS.hparams is not None:
    with gfile.Open(FLAGS.hparams, 'r') as f:
      hparams = deep_q_networks.get_hparams(**json.load(f))
  else:
    hparams = deep_q_networks.get_hparams()
  logging.info(
      'HParams:\n%s', '\n'.join([
          '\t%s: %s' % (key, value)
          for key, value in sorted(hparams.values().items())
      ]))

  # TODO(zzp): merge single objective DQN to multi objective DQN.
  if multi_objective:
    environment = MultiObjectiveRewardMolecule(
        target_molecule=FLAGS.target_molecule,
        atom_types=set(hparams.atom_types),
        init_mol=FLAGS.start_molecule,
        allow_removal=hparams.allow_removal,
        allow_no_modification=hparams.allow_no_modification,
        allow_bonds_between_rings=False,
        allowed_ring_sizes={3, 4, 5, 6},
        max_steps=hparams.max_steps_per_episode)

    dqn = deep_q_networks.MultiObjectiveDeepQNetwork(
        objective_weight=np.array([[FLAGS.similarity_weight],
                                   [1 - FLAGS.similarity_weight]]),
        input_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
        q_fn=functools.partial(
            deep_q_networks.multi_layer_model, hparams=hparams),
        optimizer=hparams.optimizer,
        grad_clipping=hparams.grad_clipping,
        num_bootstrap_heads=hparams.num_bootstrap_heads,
        gamma=hparams.gamma,
        epsilon=1.0)
  else:
    environment = TargetWeightMolecule(
        target_weight=FLAGS.target_weight,
        atom_types=set(hparams.atom_types),
        init_mol=FLAGS.start_molecule,
        allow_removal=hparams.allow_removal,
        allow_no_modification=hparams.allow_no_modification,
        allow_bonds_between_rings=hparams.allow_bonds_between_rings,
        allowed_ring_sizes=set(hparams.allowed_ring_sizes),
        max_steps=hparams.max_steps_per_episode)

    dqn = deep_q_networks.DeepQNetwork(
        input_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
        q_fn=functools.partial(
            deep_q_networks.multi_layer_model, hparams=hparams),
        optimizer=hparams.optimizer,
        grad_clipping=hparams.grad_clipping,
        num_bootstrap_heads=hparams.num_bootstrap_heads,
        gamma=hparams.gamma,
        epsilon=1.0)

  run_training(
      hparams=hparams,
      environment=environment,
      dqn=dqn,
  )

  core.write_hparams(hparams, os.path.join(FLAGS.model_dir, 'config.json'))
'''


def main(argv):
    del argv  # unused.
    run_dqn(FLAGS.multi_objective)


if __name__ == '__main__':
    app.run(main)
