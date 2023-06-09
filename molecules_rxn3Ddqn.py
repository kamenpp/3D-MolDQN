#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Created on Thu Oct 17 14:32:13 2019

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

"""Defines the Markov decision process of generating a molecule.
The problem of molecule generation as a Markov decision process, the
state space, action space, and reward function are defined.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import sys
import itertools

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdmolops import FastFindRings
from chembl_structure_pipeline import standardizer

# from mol_dqn.chemgraph.dqn.py import molecules
# import molecules1_amide_test2 as molecules
import molecules1_rxn3Ddqn as molecules

# state = molecules.initial_state()
# for i in state1:
#    state = Chem.MolToSmiles(i)
# print(type(state))

# Add more SMARTS expressions below in the future when adding more reactions
AMIDES = "[NX3][CX3](=[OX1])[#6]"
ACYL_HALIDES = "[CX3](=[OX1])[F,Cl,Br,I]"
PRIMARY_OR_SECONDARY_AMINE = "[NX3;H2,H1;!$(NC=O)]"
# TERTIARY_AMINE = "[NX3;H0;!$(NC=O)]"
CARBOXYLATE = "[CX3](=O)[O-]"
AMIDE_DISCONNECTION = '[#7:1][C;x0:2]=[O:3]>>[#7:1].Cl[C:2]=[O:3]'
PRIMARY_AMINE_AMIDE_REACTION = '[NH2:1].Cl[C:2]=[O:3]>>[#7:1][C:2]=[O:3]'
SECONDARY_AMINE_AMIDE_REACTION = '[NH1:1].Cl[C:2]=[O:3]>>[#7:1][C:2]=[O:3]'
ACID_CHLORIDE_AMIDE_REACTION = '[Cl,Br,I][C:2]=[O:3].[#7:1]>>[#7:1][C:2]=[O:3]'
CARBOXYLIC_ACID_AMIDE_REACTION = '[OH1][C:2]=[O:3].[#7:1]>>[#7:1][C:2]=[O:3]'

from absl import flags
FLAGS = flags.FLAGS


# lg = RDLogger.logger()
# lg.setLevel(RDLogger.CRITICAL)


class Result(
    collections.namedtuple('Result', ['state', 'reward', 'terminated'])):
    # same for the new action space
    """A namedtuple defines the result of a step for the molecule class.
    The namedtuple contains the following fields:
      state: Chem.RWMol. The molecule reached after taking the action.
      reward: Float. The reward get after taking the action.
      terminated: Boolean. Whether this episode is terminated.
  """


# Add more methods once expanding the action space
def get_valid_actions(state, atom_types, allow_removal, allow_no_modification,
                      allowed_ring_sizes, allow_bonds_between_rings):
    """Computes the set of valid actions for a given state.
  Args:
    state: String SMILES; the current state. If None or the empty string, we 
    read in the molecules. 
    atom_types: Set of string atom types, e.g. {'C', 'O'}.
    allow_removal: Boolean whether to allow actions that remove atoms and bonds.
    allow_no_modification: Boolean whether to include a "no-op" action.
    allowed_ring_sizes: Set of integer allowed ring sizes; used to remove some
      actions that would create rings with disallowed sizes.
    allow_bonds_between_rings: Boolean whether to allow actions that add bonds
      between atoms that are both in rings
    allow_disconnect: Boolean whether to allow actions that disconnect a compound
    by recognizing existence of particular functional group. 
    allow_find_handle_and_react: Boolean whether to allow actions that recognize 
    whether a compound has a particular functional group and after disconnection reaction, 
    react with the reagents that could perform the reaction (ex. amide reaction)
  Returns:
    Set of string SMILES containing the valid actions (technically, the set of
    all states that are acceptable from the given state).
  Raises:
    ValueError: If state does not represent a valid molecule.
  """
    # smiles = Chem.MolToSmiles(state)
    if not state:
        fpath = sys.argv[1]
        # fpath = "/data/hoodedcrow/goto/MolDQN/test_3compounds_all.smi"
        with open(fpath, 'r') as f:
            states = f.readlines()
            return states
        # fpath = "/Users/angoto/Downloads/MolDQN_Smiles/test_10compounds/test_10compounds_all.smi"
        # with open(fpath, 'r') as f:
        #    for state in set_state:
        #        f.readlines(state)

    #  raise ValueError('Received invalid state: %s' % smiles)
    # if not state:
    # Available actions are adding a node of each type.
    #  return copy.deepcopy(atom_types)
    # for state in states:
    # for s in state:
    #    print("108",s)
    mol = Chem.MolFromSmiles(state)
    if mol is None:
        fpath = sys.argv[1]
        with open(fpath, 'r') as f:
            states = f.readlines()
            return states

    # from original MolDQN
    atom_valences = dict(
        zip(sorted(atom_types), molecules.atom_valences(sorted(atom_types))))
    atoms_with_free_valence = {
        i: [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            # Only atoms that allow us to replace at least one H with a new bond
            # are enumerated here.
            if atom.GetNumImplicitHs() >= i
        ] for i in range(1, max(atom_valences.values()))
    }
    valid_actions = set()

    # from original MolDQN
    #valid_actions.update(
    #    _atom_addition(
    #        mol,
    #        atom_types=atom_types,
    #        atom_valences=atom_valences,
    #        atoms_with_free_valence=atoms_with_free_valence))
    '''bond_addition_set = _bond_addition(
        mol,
        atoms_with_free_valence=atoms_with_free_valence,
        allowed_ring_sizes=allowed_ring_sizes,
        allow_bonds_between_rings=allow_bonds_between_rings)
    if bond_addition_set is not None and len(bond_addition_set) != 0:
        valid_actions.update(bond_addition_set)  # KP this should not be necessary anymore as voids can't be returned'''
    #valid_actions.update(
    #    _bond_addition(
    #        mol,
    #        atoms_with_free_valence=atoms_with_free_valence,
    #        allowed_ring_sizes=allowed_ring_sizes,
    #        allow_bonds_between_rings=allow_bonds_between_rings)
    #)
    #if allow_removal:
    #    valid_actions.update(_bond_removal(mol))
    if allow_no_modification:
        valid_actions.add(Chem.MolToSmiles(mol))

    return valid_actions


'''
def get_valid_actions(state, atom_types, allow_removal, allow_no_modification,
                      allowed_ring_sizes, allow_bonds_between_rings):
  """Computes the set of valid actions for a given state.
  Args:
    state: String SMILES; the current state. If None or the empty string, we
      assume an "empty" state with no atoms or bonds.
    atom_types: Set of string atom types, e.g. {'C', 'O'}.
    allow_removal: Boolean whether to allow actions that remove atoms and bonds.
    allow_no_modification: Boolean whether to include a "no-op" action.
    allowed_ring_sizes: Set of integer allowed ring sizes; used to remove some
      actions that would create rings with disallowed sizes.
    allow_bonds_between_rings: Boolean whether to allow actions that add bonds
      between atoms that are both in rings.
  Returns:
    Set of string SMILES containing the valid actions (technically, the set of
    all states that are acceptable from the given state).
  Raises:
    ValueError: If state does not represent a valid molecule.
  """
  if not state:
    # Available actions are adding a node of each type.
    return copy.deepcopy(atom_types)
  mol = Chem.MolFromSmiles(state)
  if mol is None:
    raise ValueError('Received invalid state: %s' % state)
  atom_valences = dict(
      zip(sorted(atom_types), molecules.atom_valences(sorted(atom_types))))
  atoms_with_free_valence = {
      i: [
          atom.GetIdx()
          for atom in mol.GetAtoms()
          # Only atoms that allow us to replace at least one H with a new bond
          # are enumerated here.
          if atom.GetNumImplicitHs() >= i
      ] for i in range(1, max(atom_valences.values()))
  }
  valid_actions = set()
  valid_actions.update(
      _atom_addition(
          mol,
          atom_types=atom_types,
          atom_valences=atom_valences,
          atoms_with_free_valence=atoms_with_free_valence))
  valid_actions.update(
      _bond_addition(
          mol,
          atoms_with_free_valence=atoms_with_free_valence,
          allowed_ring_sizes=allowed_ring_sizes,
          allow_bonds_between_rings=allow_bonds_between_rings))
  if allow_removal:
    valid_actions.update(_bond_removal(mol))
  if allow_no_modification:
    valid_actions.add(Chem.MolToSmiles(mol))
  return valid_actions
'''


def smiles_to_clean_mol(smiles):
    """converts SMILES string to RDKit mol, then perform sanitization, uncharging, and standardizing"""
    # params = Chem.SmilesParserParams()
    # params.removeHs = False
    # molecule = Chem.MolFromSmiles(smiles, params)
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    # molecule.UpdatePropertyCache()
    # FastFindRings(molecule)
    try:
        mol = mol_to_clean_mol(molecule)
    except:
        print('failed to clean', molecule)
        print('throwing an exception from molecules/smiles_to_clean_mol')
        mol = molecule # KP 2022, defining it in case exception is thrown
    return mol


def mol_to_clean_mol(molecule):
    """converts RDKit mol to RDKit molblock, use ChEMBL standarizer, and convert back to RDKit mol
    If not possible, returns the original molecule"""
    standardize_mol_H = None
    try:
        standardize_mol = standardizer.standardize_mol(molecule)
        # Update property and add implicit H
        # standardize_mol.UpdatePropertyCache(strict=False)
        standardize_mol_H = Chem.AddHs(standardize_mol, addCoords=True)
        standardize_mol_H.UpdatePropertyCache()
        FastFindRings(standardize_mol_H)
    except (ValueError, RuntimeError):
        # except:
        print('this is the error')
        print("461", Chem.MolToSmiles(molecule))
        # Chem.SanitizeMol(molecule)
        # molecule.UpdatePropertyCache()
        # FastFindRings(molecule)
        # standardize_mol_H = molecule
        fpath = sys.argv[1]
        # fpath = "/data/hoodedcrow/goto/MolDQN/test_3compounds_all.smi"
        with open(fpath, 'r') as f:
            states = f.readlines()
            states_str = ''.join(states)
            states_mol = Chem.MolFromSmiles(states_str)

    if standardize_mol_H is not None:
        return standardize_mol_H
    else:
        return states_mol


def performing_rxn(reaction_smarts, frag1_cleanup, block):
    """performs the reaction using RDKit mol, use mol_to_clean_mol then populates the product to set_of_products - 
    the difference between performing_rxn1 is the ordering of reagents to produce product of the reaction"""

    set_of_products = set()
    try:
        rxn = AllChem.ReactionFromSmarts(reaction_smarts)
        product = rxn.RunReactants((frag1_cleanup, Chem.MolFromSmiles(block)))
        # product = rxn.RunReactants((Chem.MolFromSmiles(frag1_cleanup), Chem.MolFromSmiles(block)))
        for frag2 in product:
            # Update property and add implicit H
            # frag2.UpdatePropertyCache(strict=False)
            # frag2_H = Chem.AddHs(frag2, addCoords=True)
            # frag2_cleanup = mol_to_clean_mol(frag2_H[0])
            frag2_cleanup = mol_to_clean_mol(frag2[0])
            # When sanitization fails
            # if frag2_cleanup:
            #  continue
            new_state = Chem.MolToSmiles(frag2_cleanup, canonical=True)
            # When smiles generation fails
            # if new_state:
            #  continue
            set_of_products.add(new_state)
    except:
        print('unable to perform rxn')
    return set_of_products


def performing_rxn1(reaction_smarts, frag1_cleanup, block):
    """performs the reaction using RDKit mol, use mol_to_clean_mol then populates the product to set_of_products"""
    set_of_products = set()
    try:
        rxn = AllChem.ReactionFromSmarts(reaction_smarts)
        product = rxn.RunReactants((Chem.MolFromSmiles(block), frag1_cleanup))
        for frag2 in product:
            # Update property and add implicit H
            # frag2.UpdatePropertyCache(strict=False)
            # frag2_H = Chem.AddHs(frag2, addCoords=True)
            # frag2_cleanup = mol_to_clean_mol(frag2_H[0])
            frag2_cleanup = mol_to_clean_mol(frag2[0])
            # When sanitization fails
            # if frag2_cleanup:
            #  continue
            new_state = Chem.MolToSmiles(frag2_cleanup, canonical=True)
            # When smiles generation fails
            # if new_state:
            #  continue
            set_of_products.add(new_state)
    except:
        print('unable to perform rxn1')
    return set_of_products


def _allow_substitute_R1(state):
    """Computes valid actions that involve substituting particular functional groups to the graph.
      Actions:
        * Substitute functional group (by disconnecting a bond that consists of particular functional groups
        to the existing graph)
      Args: 
          state: RDKit Mol. 
      Returns:
       Set of string SMILES; the available actions.
    """
    global AMIDES, ACYL_HALIDES, AMIDE_DISCONNECTION, PRIMARY_AMINE_AMIDE_REACTION, SECONDARY_AMINE_AMIDE_REACTION
    ss = Chem.MolFromSmarts(AMIDES)
    aa = Chem.MolFromSmarts(ACYL_HALIDES)
    keep_R2 = set()
    allfrags_allow_substitute_R1 = set()

    # print("243", state)
    # if not state:
    #    fpath = "/data/shrike/goto/MolDQN/test_3compounds_all.smi"
    #    with open(fpath, 'r') as f:
    #        states = f.readlines( )
    #        return states
    # for state in states:
    smol1 = mol_to_clean_mol(state)
    # smi_smol1 = Chem.MolToSmiles(smol1)
    if smol1.HasSubstructMatch(ss):
        # print(Chem.MolToSmiles(smol1))      #w/o prints out compounds that cannot perform reaction
        rxn = AllChem.ReactionFromSmarts(AMIDE_DISCONNECTION)
        # try:
        #    product = rxn.RunReactants([smol1])[0]
        # print([Chem.MolToSmiles(p) for p in product])
        # except IndexError as e:
        #    print(e)
        try:
            product = rxn.RunReactants([smol1])[0]
            # count number of products after disconnection
            if len(product) != 1:
                for frag in product:
                    frag_cleanup = mol_to_clean_mol(frag)
                    if frag_cleanup.HasSubstructMatch(aa):
                        filtered_product = Chem.MolToSmiles(frag_cleanup)
                        keep_R2.add(filtered_product)
                        for block in keep_R2:
                            # print("255", block)
                            for frag1 in molecules.amide_reaction_reagent_amine():
                                # print("253", frag1)
                                frag1_cleanup = smiles_to_clean_mol(frag1)
                                # print("254", frag1_cleanup)
                                # yes = performing_rxn(PRIMARY_AMINE_AMIDE_REACTION, frag1_cleanup, block)
                                # print("254", yes)
                                allfrags_allow_substitute_R1.update(
                                    performing_rxn(PRIMARY_AMINE_AMIDE_REACTION, frag1_cleanup, block))
                                allfrags_allow_substitute_R1.update(
                                    performing_rxn(SECONDARY_AMINE_AMIDE_REACTION, frag1_cleanup, block))
        except IndexError as e:
            print(e)
        # except IndexError as e:
        #   print(e)
    allfrags_allow_substitute_R1 = [x for x in allfrags_allow_substitute_R1 if x]
    # print("279", allfrags_allow_substitute_R1)
    return allfrags_allow_substitute_R1


def _allow_substitute_R2(state):
    """Computes valid actions that involve substituting particular functional groups to the graph.
      Actions:
        * Substitute functional group (by disconnecting a bond that consists of particular functional groups
        to the existing graph)
      Args: 
          state: RDKit Mol. 
      Returns:
       Set of string SMILES; the available actions.
    """
    global AMIDES, ACYL_HALIDES, PRIMARY_OR_SECONDARY_AMINE, AMIDE_DISCONNECTION, ACID_CHLORIDE_AMIDE_REACTION, CARBOXYLIC_ACID_AMIDE_REACTION
    ss = Chem.MolFromSmarts(AMIDES)
    aa = Chem.MolFromSmarts(ACYL_HALIDES)
    bb = Chem.MolFromSmarts(PRIMARY_OR_SECONDARY_AMINE)
    keep_R1 = set()
    allfrags_allow_substitute_R2 = set()

    # fpath = "/data/shrike/goto/MolDQN/test_3compounds_all.smi"
    # with open(fpath, 'r') as f:
    #    states = f.readlines( )
    # for state in states:
    # print("299", Chem.MolToSmiles(state))
    smol1 = mol_to_clean_mol(state)
    # smi_smol1 = Chem.MolToSmiles(smol1)
    if smol1.HasSubstructMatch(ss):
        # print(Chem.MolToSmiles(smol1))      #w/o prints out compounds that cannot perform reaction
        rxn = AllChem.ReactionFromSmarts(AMIDE_DISCONNECTION)
        try:
            product = rxn.RunReactants([smol1])[0]
            # count number of products after disconnection
            if len(product) != 1:
                for frag in product:
                    frag_cleanup = mol_to_clean_mol(frag)
                    frag_cleanup_one_reaction_center = frag_cleanup.GetSubstructMatches(bb)
                    if len(frag_cleanup_one_reaction_center) == 1:
                        continue
                        if frag_cleanup.HasSubstructMatch(aa):
                            continue
                        else:
                            filtered_product = Chem.MolToSmiles(frag_cleanup)
                            keep_R1.add(filtered_product)
                            for block in keep_R1:
                                for frag1 in molecules.amide_reaction_reagent_acyl_halide():
                                    frag1_cleanup = smiles_to_clean_mol(frag1)
                                allfrags_allow_substitute_R2.update(
                                    performing_rxn(ACID_CHLORIDE_AMIDE_REACTION, frag1_cleanup, block))
                                allfrags_allow_substitute_R2.update(
                                    performing_rxn(CARBOXYLIC_ACID_AMIDE_REACTION, frag1_cleanup, block))
            # products = rxn.RunReactants([smol1])
            # product = products[0]
            # for p in products:
            #    print("309", Chem.MolToSmiles(p[0]))
            #    print("310", Chem.MolToSmiles(p[1]))
            # print("307", [Chem.MolToSmiles(p) for p in products])
            # print([Chem.MolToSmiles(p) for p in product])
        except IndexError as e:
            print(e)

    allfrags_allow_substitute_R2 = [x for x in allfrags_allow_substitute_R2 if x]
    # print("340", allfrags_allow_substitute_R2)
    return allfrags_allow_substitute_R2


def _allow_disconnect(state):
    """Computes valid actions that involve disconnecting the bond by recognizing particular functional groups to the graph.
      Actions:
        * Disconnect a bond (by recognizing particular functional groups to the existing graph)
      Args: 
          state: RDKit Mol. 
      Returns:
       Set of string SMILES; the available actions.
    """
    global AMIDES, AMIDE_DISCONNECTION
    ss = Chem.MolFromSmarts(AMIDES)
    allfrags_disconnection = set()

    # fpath = "/data/shrike/goto/MolDQN/test_3compounds_all.smi"
    # with open(fpath, 'r') as f:
    #    states = f.readlines( )
    # for state in states:
    smol1 = mol_to_clean_mol(state)
    if smol1.HasSubstructMatch(ss):
        # print(Chem.MolToSmiles(molecule))
        ##print(Chem.MolToSmiles(state))      #w/o prints out compounds that cannot perform reaction 
        rxn = AllChem.ReactionFromSmarts(AMIDE_DISCONNECTION)
        try:
            product = rxn.RunReactants([smol1])[0]
        except IndexError as e:
            print(e)
        for frag in product:
            frag_cleanup = mol_to_clean_mol(frag)
            new_state = Chem.MolToSmiles(frag_cleanup)
            allfrags_disconnection.update(new_state)
    allfrags_disconnection = [x for x in allfrags_disconnection if x]
    return allfrags_disconnection


def _allow_find_handle_and_react(state):
    """Computes valid actions that involve recognizing reaction center and disconnecting the bond to the graph.
      Actions:
        * Recognize the functional group and disconnect to perform the reaction (by recognizing particular functional groups to the existing graph)
      Args: 
          state: RDKit Mol. 
      Returns:
       Set of string SMILES; the available actions.
    """
    global PRIMARY_OR_SECONDARY_AMINE, ACYL_HALIDES, CARBOXYLATE, PRIMARY_AMINE_AMIDE_REACTION, SECONDARY_AMINE_AMIDE_REACTION, ACID_CHLORIDE_AMIDE_REACTION, CARBOXYLIC_ACID_AMIDE_REACTION
    ss = Chem.MolFromSmarts(PRIMARY_OR_SECONDARY_AMINE)  # SMARTS for primary or secondary amine
    aa = Chem.MolFromSmarts(ACYL_HALIDES)  # SMARTS for acyl halide
    bb = Chem.MolFromSmarts(CARBOXYLATE)  # SMARTS for carboxylate
    allfrags_allow_find_handle_and_react = set()
    keep_amine = set()
    keep_acyl_halide = set()
    keep_carboxylate = set()

    # fpath = "/data/shrike/goto/MolDQN/test_3compounds_all.smi"
    # with open(fpath, 'r') as f:
    #    states = f.readlines( )
    # for state in states:
    smol1 = mol_to_clean_mol(state)
    if smol1.HasSubstructMatch(ss):
        amine = Chem.MolToSmiles(smol1)
        keep_amine.add(amine)
        for block in keep_amine:
            for frag1 in molecules.amide_reaction_reagent_acyl_halide():
                frag1_cleanup = smiles_to_clean_mol(frag1)
                allfrags_allow_find_handle_and_react.update(
                    performing_rxn1(PRIMARY_AMINE_AMIDE_REACTION, block, frag1_cleanup))
                allfrags_allow_find_handle_and_react.update(
                    performing_rxn1(SECONDARY_AMINE_AMIDE_REACTION, block, frag1_cleanup))
    if smol1.HasSubstructMatch(aa):
        acyl_halide = Chem.MolToSmiles(smol1)
        keep_acyl_halide.add(acyl_halide)
        for block in keep_acyl_halide:
            for frag1 in molecules.amide_reaction_reagent_amine():
                frag1_cleanup = smiles_to_clean_mol(frag1)
                allfrags_allow_find_handle_and_react.update(
                    performing_rxn1(ACID_CHLORIDE_AMIDE_REACTION, block, frag1_cleanup))
    if smol1.HasSubstructMatch(bb):
        carboxylate = Chem.MolToSmiles(smol1)
        keep_carboxylate.add(carboxylate)
        for block in keep_carboxylate:
            for frag1 in molecules.amide_reaction_reagent_amine():
                frag1_cleanup = smiles_to_clean_mol(frag1)
                allfrags_allow_find_handle_and_react.update(
                    performing_rxn1(CARBOXYLIC_ACID_AMIDE_REACTION, block, frag1_cleanup))
    allfrags_allow_find_handle_and_react = [x for x in allfrags_allow_find_handle_and_react if x]
    return allfrags_allow_find_handle_and_react


# parts of code from original MolDQN

def _atom_addition(state, atom_types, atom_valences, atoms_with_free_valence): # KP refactored
    """Computes valid actions that involve adding atoms to the graph.
  Actions:
    * Add atom (with a bond connecting it to the existing graph)
  Each added atom is connected to the graph by a bond. There is a separate
  action for connecting to (a) each existing atom with (b) each valence-allowed
  bond type. Note that the connecting bond is only of type single, double, or
  triple (no aromatic bonds are added).
  For example, if an existing carbon atom has two empty valence positions and
  the available atom types are {'C', 'O'}, this section will produce new states
  where the existing carbon is connected to (1) another carbon by a double bond,
  (2) another carbon by a single bond, (3) an oxygen by a double bond, and
  (4) an oxygen by a single bond.
  Args:
    state: RDKit Mol.
    atom_types: Set of string atom types.
    atom_valences: Dict mapping string atom types to integer valences.
    atoms_with_free_valence: Dict mapping integer minimum available valence
      values to lists of integer atom indices. For instance, all atom indices in
      atoms_with_free_valence[2] have at least two available valence positions.
  Returns:
    Set of string SMILES; the available actions.
  """
    bond_order = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }

    # Added for fragment masking
    core_to_keep = set()
    #if FLAGS.smarts_mask != 'False':
    #    core_to_keep_patt = Chem.MolFromSmarts(FLAGS.smarts_mask)
    smol1 = mol_to_clean_mol(state)
    #print('the type of smol1 is ', type(smol1))
    #print('the smol1 smiles is ', Chem.MolToSmiles(smol1))
    if FLAGS.smarts_mask != 'False':
        # Added for fragment masking
        core_to_keep_patt = Chem.MolFromSmarts(FLAGS.smarts_mask)
        core_to_keep = set(state.GetSubstructMatch(core_to_keep_patt)) # was using smol1
        if len(core_to_keep) == 0:
            print('THE LENGTH OF THE ATOM ADDITION SET IS 0!')
            return set()

    atom_addition = set()
    for i in bond_order:
        # remove indices of core_to_keep from atom_with_free_valence
        # same way that we just talked about
        for atom in atoms_with_free_valence[i]:

            if atom in core_to_keep:  # i.e. if the atom with free valence is from the fragment, skip trying to add atoms to it
                #print('index in atom skipped', atom)
                continue
            for element in atom_types:
                if atom_valences[element] >= i:
                    new_state = Chem.RWMol(state)
                    idx = new_state.AddAtom(Chem.Atom(element))
                    #if idx in atoms_with_free_valence[i]:
                    #    print('WAIT, IT CAN BE ONE THAT WAS ALREADY IN THE MOLECULE')
                    #if idx in core_to_keep:  # KP 2022 this never actually gets executed
                    #    return
                    new_state.AddBond(atom, idx, bond_order[i])
                    # new_state1 = mol_to_clean_mol(new_state)
                    # standardization_result = standardizer.standardize_mol(new_state)
                    sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
                    # When standardization fails - KP I guess sanitisation
                    if sanitization_result:  # KP - so it doesn't matter whether sanitization fails?
                        continue
                    # Check for molecular weight
                    #if MolWt(new_state) <= 500:
                    try:
                        smiles = Chem.MolToSmiles(new_state)
                        atom_addition.add(smiles)
                    except (ValueError, RuntimeError):  # KP completely proper way of handling that...
                        print('An error occured while trying to convert the molecule to smiles! (_atom_addition)')  # KP 2022
                        continue
    print('the length of the atom addition set ', len(atom_addition))

    return atom_addition


def _bond_addition(state, atoms_with_free_valence, allowed_ring_sizes,
                   allow_bonds_between_rings):  # KP 2022 refactored here
    """Computes valid actions that involve adding bonds to the graph.
  Actions (where allowed):
    * None->{single,double,triple}
    * single->{double,triple}
    * double->{triple}
  Note that aromatic bonds are not modified.
  Args:
    state: RDKit Mol.
    atoms_with_free_valence: Dict mapping integer minimum available valence
      values to lists of integer atom indices. For instance, all atom indices in
      atoms_with_free_valence[2] have at least two available valence positions.
    allowed_ring_sizes: Set of integer allowed ring sizes; used to remove some
      actions that would create rings with disallowed sizes.
    allow_bonds_between_rings: Boolean whether to allow actions that add bonds
      between atoms that are both in rings.
  Returns:
    Set of string SMILES; the available actions.
  """
    bond_orders = [
        None,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
    ]

    # Added for fragment masking
    core_to_keep = set()
    #if FLAGS.smarts_mask != 'False':
    #    core_to_keep_patt = Chem.MolFromSmarts(FLAGS.smarts_mask)
    #    flag_bond_in_core = False

    smol1 = mol_to_clean_mol(state)
    if FLAGS.smarts_mask != 'False':
        # Added for fragment masking
        core_to_keep_patt = Chem.MolFromSmarts(FLAGS.smarts_mask)
        core_to_keep = set(state.GetSubstructMatch(core_to_keep_patt)) # was using smol1
        if len(core_to_keep) == 0:
            print('THE LENGTH OF THE BOND ADDITION SET IS 0!')

    bond_addition = set()
    for valence, atoms in atoms_with_free_valence.items():
        for atom1, atom2 in itertools.combinations(atoms, 2):
            # Get the bond from a copy of the molecule so that SetBondType() doesn't
            # modify the original state.
            if atom1 in core_to_keep or atom2 in core_to_keep:  # KP if any of the
                #print("did not make a bond between", atom1, ' and ', atom2)
                continue
            bond = Chem.Mol(state).GetBondBetweenAtoms(atom1, atom2)
            # print('2000', core_to_keep)
            new_state = Chem.RWMol(state)
            # Kekulize the new state to avoid sanitization errors; note that bonds
            # that are aromatic in the original state are not modified (this is
            # enforced by getting the bond from the original state with
            # GetBondBetweenAtoms()).
            Chem.Kekulize(new_state, clearAromaticFlags=True)
            #print('will be making a bond between ', atom1, ' and ', atom2)
            if bond is not None:
                if bond.GetBondType() not in bond_orders:
                    continue  # Skip aromatic bonds.
                idx = bond.GetIdx()
                #if idx in core_to_keep: # KP this is not meaningful because the core_to_keep contains atoms not bond indices
                #    flag_bond_in_core = True
                #    print('BOND IN THE MASKED FRAGMENT')
                #    print('there should be a chance to get this actually!')
                #    return
                # Compute the new bond order as an offset from the current bond order.
                bond_order = bond_orders.index(bond.GetBondType())
                bond_order += valence
                if bond_order < len(bond_orders):
                    idx = bond.GetIdx()  # that is the bond index I don't know why you're trying to find it in the atoms
                    #if idx in core_to_keep:  # KP this returns void sometimes and breaks the step
                    #    return
                    bond.SetBondType(bond_orders[bond_order])
                    new_state.ReplaceBond(idx, bond)
                else:
                    continue
            # If do not allow new bonds between atoms already in rings.
            elif (not allow_bonds_between_rings and
                  (state.GetAtomWithIdx(atom1).IsInRing() and
                   state.GetAtomWithIdx(atom2).IsInRing())):
                continue
            # If the distance between the current two atoms is not in the
            # allowed ring sizes
            elif (allowed_ring_sizes is not None and
                  len(Chem.rdmolops.GetShortestPath(
                      state, atom1, atom2)) not in allowed_ring_sizes):
                continue
            else:
                #if atom1 in core_to_keep or atom2 in core_to_keep:
                #    return
                new_state.AddBond(atom1, atom2, bond_orders[valence])
            # standardization_result = standardizer.standardize_mol(new_state)
            # new_state1 = mol_to_clean_mol(new_state)
            sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
            # When standardization fails
            if sanitization_result:
                continue
            #if MolWt(new_state) <= 500: KP commented that because it makes no sense to be constrained
            try:
                smiles = Chem.MolToSmiles(new_state)
                bond_addition.add(smiles)
            except (ValueError, RuntimeError):
                print('An error occured while trying to convert the molecule to smiles! (_bond_addition)')  # KP 2022
                continue
    print('the length of the bond addition set ', len(bond_addition))
    return bond_addition


def _bond_removal(state):  # KP 2022 refactored here
    """Computes valid actions that involve removing bonds from the graph.
  Actions (where allowed):
    * triple->{double,single,None}
    * double->{single,None}
    * single->{None}
  Bonds are only removed (single->None) if the resulting graph has zero or one
  disconnected atom(s); the creation of multi-atom disconnected fragments is not
  allowed. Note that aromatic bonds are not modified.
  Args:
    state: RDKit Mol.
  Returns:
    Set of string SMILES; the available actions.
  """
    bond_orders = [
        None,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
    ]

    # Added for fragment masking
    #core_to_keep = set()
    #if FLAGS.smarts_mask != 'False':
    #    core_to_keep_patt = Chem.MolFromSmarts(FLAGS.smarts_mask)
    core_to_keep = set()
    smol1 = mol_to_clean_mol(state)
    if FLAGS.smarts_mask != 'False':
        # Added for fragment masking
        core_to_keep_patt = Chem.MolFromSmarts(FLAGS.smarts_mask)
        core_to_keep = set(state.GetSubstructMatch(core_to_keep_patt)) # was using smol1
        if len(core_to_keep) == 0:
            print('THE LENGTH OF THE BOND REMOVAL SET IS 0!')
        #for idx in tuple_of_atom_indices:
        #    core_to_keep.add(idx)

    bond_removal = set()
    bond_removal_500 = set()
    for valence in [1, 2, 3]:
        for bond in state.GetBonds():
            # Get the bond from a copy of the molecule so that SetBondType() doesn't
            # modify the original state.
            if bond.GetBeginAtomIdx() in core_to_keep or bond.GetEndAtomIdx() in core_to_keep:
                #print('skipped bond between ', bond.GetBeginAtomIdx(), ' and ', bond.GetEndAtomIdx())
                continue
            bond = Chem.Mol(state).GetBondBetweenAtoms(bond.GetBeginAtomIdx(),
                                                       bond.GetEndAtomIdx())
            if bond.GetBondType() not in bond_orders:
                continue  # Skip aromatic bonds.
            new_state = Chem.RWMol(state)
            # Kekulize the new state to avoid sanitization errors; note that bonds
            # that are aromatic in the original state are not modified (this is
            # enforced by getting the bond from the original state with
            # GetBondBetweenAtoms()).
            Chem.Kekulize(new_state, clearAromaticFlags=True)
            # Compute the new bond order as an offset from the current bond order.
            bond_order = bond_orders.index(bond.GetBondType())
            bond_order -= valence
            if bond_order > 0:  # Downgrade this bond.
                idx = bond.GetIdx()
                #if idx in core_to_keep:
                #    print('this should also have a chance to get triggered!')
                #    print('THE BOND TO REMOVE WAS IN THE MASKED FRAGMENT')
                #    return
                bond.SetBondType(bond_orders[bond_order])
                new_state.ReplaceBond(idx, bond)
                # standardization_result = standardizer.standardize_mol(new_state)
                sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
                # When standardization fails
                if sanitization_result:
                    continue
                # Check for molecular weight
                # new_state1 = mol_to_clean_mol(new_state)
                # if new_state1 != None:
                #if MolWt(new_state) <= 500: KP 2022
                try:
                    smiles = Chem.MolToSmiles(new_state)
                    bond_removal_500.add(smiles)
                except (ValueError, RuntimeError):
                    continue
            elif bond_order == 0:  # Remove this bond entirely.
                atom1 = bond.GetBeginAtom().GetIdx()
                atom2 = bond.GetEndAtom().GetIdx()
                #if atom1 in core_to_keep or atom2 in core_to_keep: # KP I don't think this can ever be reached
                #    return
                new_state.RemoveBond(atom1, atom2)
                # new_state1 = mol_to_clean_mol(new_state)
                # standardization_result = standardizer.standardize_mol(new_state)
                sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
                # When standardizaton fails
                if sanitization_result:
                    continue
                # When smiles generation fails
                #if MolWt(new_state) <= 500: KP commented that out
                try:
                    smiles = Chem.MolToSmiles(new_state)
                    parts = sorted(smiles.split('.'), key=len)
                    # We define the valid bond removing action set as the actions
                    # that remove an existing bond, generating only one independent
                    # molecule, or a molecule and an atom.
                    if len(parts) == 1 or len(parts[0]) == 1:
                        bond_removal_500.add(parts[-1])
                except (ValueError, RuntimeError):
                    continue
    print('the length of the atom removal set ', len(bond_removal_500))
    return bond_removal_500


class Molecule(object):
    """Defines the Markov decision process of generating a molecule."""

    def __init__(self,
                 atom_types,
                 init_mol=None,
                 allow_removal=True,
                 allow_no_modification=True,
                 allow_bonds_between_rings=True,
                 allowed_ring_sizes=None,
                 max_steps=10,
                 target_fn=None,
                 record_path=False):
        """Initializes the parameters for the MDP.
    Internal state will be stored as SMILES strings.
    Args:
      atom_types: The set of elements the molecule may contain.
      init_mol: String, Chem.Mol, or Chem.RWMol. If string is provided, it is
        considered as the SMILES string. The molecule to be set as the initial
        state. If None, an empty molecule will be created.
      allow_removal: Boolean. Whether to allow removal of a bond.
      allow_no_modification: Boolean. If true, the valid action set will
        include doing nothing to the current molecule, i.e., the current
        molecule itself will be added to the action set.
      allow_bonds_between_rings: Boolean. If False, new bonds connecting two
        atoms which are both in rings are not allowed.
        DANGER Set this to False will disable some of the transformations eg.
        c2ccc(Cc1ccccc1)cc2 -> c1ccc3c(c1)Cc2ccccc23
        But it will make the molecules generated make more sense chemically.
      allowed_ring_sizes: Set of integers or None. The size of the ring which
        is allowed to form. If None, all sizes will be allowed. If a set is
        provided, only sizes in the set is allowed.
      allow_disconnect: Boolean. Whether to allow disconnection of a bond. 
      allow_find_handle_and_react: Boolean. If true, the valid action set will 
      also include actions where the reaction center (i.e. functional group) has been
      recognized and after performing the disconnection reaction, the synthon reacts
      with the corresponding reagent. 
      max_steps: Integer. The maximum number of steps to run.
      target_fn: A function or None. The function should have Args of a
        String, which is a SMILES string (the state), and Returns as
        a Boolean which indicates whether the input satisfies a criterion.
        If None, it will not be used as a criterion.
      record_path: Boolean. Whether to record the steps internally.
    """
        if isinstance(init_mol, Chem.Mol):
            init_mol = Chem.MolToSmiles(init_mol)
        self.init_mol = init_mol
        self.atom_types = atom_types
        self.allow_removal = allow_removal
        self.allow_no_modification = allow_no_modification
        self.allow_bonds_between_rings = allow_bonds_between_rings
        self.allowed_ring_sizes = allowed_ring_sizes
        self.max_steps = max_steps
        self._state = None
        self._valid_actions = []
        # The status should be 'terminated' if initialize() is not called.
        self._counter = self.max_steps
        self._target_fn = target_fn
        self.record_path = record_path
        self._path = []
        self._max_bonds = 4
        atom_types = list(self.atom_types)
        self._max_new_bonds = dict(
            zip(atom_types, molecules.atom_valences(atom_types)))

    @property
    def state(self):  # same for new action space
        return self._state

    @property
    def num_steps_taken(self):  # same for new action space
        return self._counter

    def get_path(self):  # same for new action space
        return self._path

    def initialize(self):
        """Resets the MDP to its initial state."""  # same for the new ation space
        self._state = self.init_mol
        if self.record_path:
            self._path = [self._state]
        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._counter = 0

    def get_valid_actions(self, state=None, force_rebuild=False):
        """Gets the valid actions for the state.
    In this design, we do not further modify a aromatic ring. For example,
    we do not change a benzene to a 1,3-Cyclohexadiene. That is, aromatic
    bonds are not modified.
    Args:
      state: String, Chem.Mol, or Chem.RWMol. If string is provided, it is
        considered as the SMILES string. The state to query. If None, the
        current state will be considered.
      force_rebuild: Boolean. Whether to force rebuild of the valid action
        set.
    Returns:
      A set contains all the valid actions for the state. Each action is a
        SMILES string. The action is actually the resulting state.
    """
        if state is None:
            if self._valid_actions and not force_rebuild:
                return copy.deepcopy(self._valid_actions)
            state = self._state
        if isinstance(state, Chem.Mol):
            state = Chem.MolToSmiles(state)
        self._valid_actions = get_valid_actions(
            state,
            atom_types=self.atom_types,
            allow_removal=self.allow_removal,
            allow_no_modification=self.allow_no_modification,
            allowed_ring_sizes=self.allowed_ring_sizes,
            allow_bonds_between_rings=self.allow_bonds_between_rings)
        return copy.deepcopy(self._valid_actions)

    def _reward(self):  # same for new action space
        """Gets the reward for the state.
    A child class can redefine the reward function if reward other than
    zero is desired.
    Returns:
      Float. The reward for the current state.
    """
        return 0.0

    def _goal_reached(self):  # same for new action space
        """Sets the termination criterion for molecule Generation.
    A child class can define this function to terminate the MDP before
    max_steps is reached.
    Returns:
      Boolean, whether the goal is reached or not. If the goal is reached,
        the MDP is terminated.
    """
        if self._target_fn is None:
            return False
        return self._target_fn(self._state)

    def step(self, action, hparams, episode, steps_left):
        """Takes a step forward according to the action.
    Args:
      action: Chem.RWMol. The action is actually the target of the modification.
    Returns:
      results: Namedtuple containing the following fields:
        * state: The molecule reached after taking the action.
        * reward: The reward get after taking the action.
        * terminated: Whether this episode is terminated.
    Raises:
      ValueError: If the number of steps taken exceeds the preset max_steps, or
        the action is not in the set of valid_actions.
    """
        if self._counter >= self.max_steps or self._goal_reached():
            raise ValueError('This episode is terminated.')
        if action not in self._valid_actions:
            raise ValueError('Invalid action.')
        self._state = action
        if self.record_path:
            self._path.append(self._state)
        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._counter += 1
        reward = self._reward(hparams, episode, steps_left)  # KP 2022
        #print('Episode: ',episode,' steps left: ', steps_left, ' the reward is ', reward)
        #print('Episode: ',episode,' steps left: ', steps_left, ' the state smiles is ', self._state)

        result = Result(
            state=self._state,
            reward=reward,
            terminated=(self._counter >= self.max_steps) or self._goal_reached())
        return result

    def visualize_state(self, state=None, **kwargs):  # same for new action space
        """Draws the molecule of the state.
    Args:
      state: String, Chem.Mol, or Chem.RWMol. If string is prov ided, it is
        considered as the SMILES string. The state to query. If None, the
        current state will be considered.
      **kwargs: The keyword arguments passed to Draw.MolToImage.
    Returns:
      A PIL image containing a drawing of the molecule.
    """
        if state is None:
            state = self._state
        if isinstance(state, str):
            state = Chem.MolFromSmiles(state)
        return Draw.MolToImage(state, **kwargs)


'''
  def __init__(self,
               atom_types,
               init_mol=None,
               allow_removal=True,
               allow_no_modification=True,
               allow_bonds_between_rings=True,
               allowed_ring_sizes=None,
               max_steps=10,
               target_fn=None,
               record_path=False):
    """Initializes the parameters for the MDP.
    Internal state will be stored as SMILES strings.
    Args:
      atom_types: The set of elements the molecule may contain.
      init_mol: String, Chem.Mol, or Chem.RWMol. If string is provided, it is
        considered as the SMILES string. The molecule to be set as the initial
        state. If None, an empty molecule will be created.
      allow_removal: Boolean. Whether to allow removal of a bond.
      allow_no_modification: Boolean. If true, the valid action set will
        include doing nothing to the current molecule, i.e., the current
        molecule itself will be added to the action set.
      allow_bonds_between_rings: Boolean. If False, new bonds connecting two
        atoms which are both in rings are not allowed.
        DANGER Set this to False will disable some of the transformations eg.
        c2ccc(Cc1ccccc1)cc2 -> c1ccc3c(c1)Cc2ccccc23
        But it will make the molecules generated make more sense chemically.
      allowed_ring_sizes: Set of integers or None. The size of the ring which
        is allowed to form. If None, all sizes will be allowed. If a set is
        provided, only sizes in the set is allowed.
      max_steps: Integer. The maximum number of steps to run.
      target_fn: A function or None. The function should have Args of a
        String, which is a SMILES string (the state), and Returns as
        a Boolean which indicates whether the input satisfies a criterion.
        If None, it will not be used as a criterion.
      record_path: Boolean. Whether to record the steps internally.
    """
    if isinstance(init_mol, Chem.Mol):
      init_mol = Chem.MolToSmiles(init_mol)
    self.init_mol = init_mol
    self.atom_types = atom_types
    self.allow_removal = allow_removal
    self.allow_no_modification = allow_no_modification
    self.allow_bonds_between_rings = allow_bonds_between_rings
    self.allowed_ring_sizes = allowed_ring_sizes
    self.max_steps = max_steps
    self._state = None
    self._valid_actions = []
    # The status should be 'terminated' if initialize() is not called.
    self._counter = self.max_steps
    self._target_fn = target_fn
    self.record_path = record_path
    self._path = []
    self._max_bonds = 4
    atom_types = list(self.atom_types)
    self._max_new_bonds = dict(
        zip(atom_types, molecules.atom_valences(atom_types)))
'''
'''
  def get_valid_actions(self, state=None, force_rebuild=False):
    """Gets the valid actions for the state.
    In this design, we do not further modify a aromatic ring. For example,
    we do not change a benzene to a 1,3-Cyclohexadiene. That is, aromatic
    bonds are not modified.
    Args:
      state: String, Chem.Mol, or Chem.RWMol. If string is provided, it is
        considered as the SMILES string. The state to query. If None, the
        current state will be considered.
      force_rebuild: Boolean. Whether to force rebuild of the valid action
        set.
    Returns:
      A set contains all the valid actions for the state. Each action is a
        SMILES string. The action is actually the resulting state.
    """
    if state is None:
      if self._valid_actions and not force_rebuild:
        return copy.deepcopy(self._valid_actions)
      state = self._state
    if isinstance(state, Chem.Mol):
      state = Chem.MolToSmiles(state)
    self._valid_actions = get_valid_actions(
        state,
        atom_types=self.atom_types,
        allow_removal=self.allow_removal,
        allow_no_modification=self.allow_no_modification,
        allowed_ring_sizes=self.allowed_ring_sizes,
        allow_bonds_between_rings=self.allow_bonds_between_rings)
    return copy.deepcopy(self._valid_actions)
'''
