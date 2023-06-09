#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Created on Thu Oct 17 16:09:03 2019

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

"""Tools for manipulating graphs and converting from atom and pair features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Crippen import MolLogP  # KP 2022 added import
from rdkit.Chem.Scaffolds import MurckoScaffold
from molvs import Standardizer, charge
# from rdkit.Contrib.SA_Score import sascorer
import pandas as pd
import numpy
import sys
from openbabel import openbabel  # added for docking
from openbabel import pybel  # added for docking
import os  # added for docking
import subprocess  # added for docking

import SA_Score.sascorer as sascorer


# the first two definitions are not working
# define more methods once needing more reagents to perform other reactions in the action space

def initial_state():
    fpath = "/data/hoodedcrow/goto/MolDQN/test_10compounds_all.smi"
    with open(fpath, 'r') as f:
        smiles_amide_reaction_starting_material1 = f.readline()
    return smiles_amide_reaction_starting_material1


def amide_reaction_starting_material():
    fpath = "/data/hoodedcrow/goto/MolDQN/test_3compounds_all.smi"
    with open(fpath, 'r') as f:
        smiles_amide_reaction_starting_material = f.readlines()
    return smiles_amide_reaction_starting_material


def amide_reaction_reagent_amine():
    fpath2 = sys.argv[3]
    # fpath2 = "/data/hoodedcrow/goto/MolDQN/intersection.smi"
    # fpath2 = "/data/hoodedcrow/goto/MolDQN/test_fragmentB_all_mini.smi"
    with open(fpath2, 'r') as f:
        smiles_amide_reaction_reagent_amine = f.readlines()
    return smiles_amide_reaction_reagent_amine


def amide_reaction_reagent_acyl_halide():
    fpath1 = sys.argv[2]
    # fpath1 = "/data/hoodedcrow/goto/MolDQN/intersection1.smi"
    # fpath1 = "/data/hoodedcrow/goto/MolDQN/test_fragmentA_all_mini.smi"
    with open(fpath1, 'r') as f:
        smiles_amide_reaction_reagent_acyl_halide = f.readlines()
    return smiles_amide_reaction_reagent_acyl_halide


'''
def SMILES_drugbank():
    data1 = pd.read_csv('/data/hoodedcrow/goto/MolDQN/drug_bank_approved.csv')
    b = data1['SMILES'].values
    approved_drug_smiles_list = b.tolist()
    approved_drug_smiles = {*approved_drug_smiles_list}
    return approved_drug_smiles

def search_amide_from_drugbank():
    ss = Chem.MolFromSmarts("[NX3][CX3](=[OX1])[#6]")
    approved_drug_with_amide_bond = set()
    for drug in SMILES_drugbank():
        molecule = Chem.MolFromSmiles(drug)
        if molecule.HasSubstructMatch(ss):
            approved_drug_with_amide_bond.add(drug)
        else:
            continue
    return approved_drug_with_amide_bond

def reactant_1_smiles_restricted_yield():
    data = pd.read_csv('data/hoodedcrow/goto/MolDQN/restricted_yield.csv)
    a = data['Reactant_1_SMILES'].values
    reactant_amine_smiles_list = a.tolist()
    reactant_amine_smiles = {*reactant_amine_smiles_list}
    return reactant_amine_smiles
    
def reactant_2_smiles_restricted_yield():
    data = pd.read_csv('data/hoodedcrow/goto/MolDQN/restricted_yield.csv)
    c = data['Reactant_2_SMILES'].values
    reactant_acyl_halide_smiles_list = c.tolist()
    reactant_acyl_halide_smiles = {*reactant_acyl_halide_smiles_list}
    return reactant_acyl_halide_smiles
 
def amide_reaction_starting_material():
    fpath = "/Users/angoto/Downloads/MolDQN_Smiles/test_10compounds/test_10compounds_all.smi"
    with open(fpath, 'r') as f:
        smiles_amide_reaction_starting_material = f.readlines( )
    return smiles_amide_reaction_starting_material

def amide_reaction_reagent_amine():
    fpath2 = "/Users/angoto/Downloads/MolDQN_Smiles/test_fragmentB/test_fragmentB_all.smi"
    with open(fpath2, 'r') as f:
        smiles_amide_reaction_reagent_amine = f.readlines( )
    return smiles_amide_reaction_reagent_amine

def amide_reaction_reagent_acyl_halide():
    fpath1 = "/Users/angoto/Downloads/MolDQN_Smiles/test_fragmentA/test_fragmentA_all.smi"
    with open(fpath1, 'r') as f:
        smiles_amide_reaction_reagent_acyl_halide = f.readlines( )
    return smiles_amide_reaction_reagent_acyl_halide
'''
'''
def amide_reaction_starting_material():
  fpath = "/Users/angoto/Downloads/MolDQN_Smiles/test_10compounds/test_10compounds_all.smi"
  with open(fpath, 'r') as f:
      smiles_amide_reaction_starting_material = f.readlines( )
  return smiles_amide_reaction_starting_material

def amide_reaction_reagent_amine():
  fpath2 = "/Users/angoto/Downloads/MolDQN_Smiles/test_fragmentB/test_fragmentB_all.smi"
  with open(fpath2, 'r') as f:
      smiles_amide_reaction_reagent_amine = f.readlines( )
  return smiles_amide_reaction_reagent_amine

def amide_reaction_reagent_acyl_halide():
  fpath1 = "/Users/angoto/Downloads/MolDQN_Smiles/test_fragmentA/test_fragmentA_all.smi"
  with open(fpath1, 'r') as f:
      smiles_amide_reaction_reagent_acyl_halide = f.readlines( )
  return smiles_amide_reaction_reagent_acyl_halide
'''


def atom_valences(atom_types):
    """Creates a list of valences corresponding to atom_types.
  Note that this is not a count of valence electrons, but a count of the
  maximum number of bonds each element will make. For example, passing
  atom_types ['C', 'H', 'O'] will return [4, 1, 2].
  Args:
    atom_types: List of string atom types, e.g. ['C', 'H', 'O'].
  Returns:
    List of integer atom valences.
  """
    periodic_table = Chem.GetPeriodicTable()
    return [
        max(list(periodic_table.GetValenceList(atom_type)))
        for atom_type in atom_types
    ]


def get_scaffold(mol):
    """Computes the Bemis-Murcko scaffold for a molecule.
  Args:
    mol: RDKit Mol.
  Returns:
    String scaffold SMILES.
  """
    return Chem.MolToSmiles(
        MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=True)


def contains_scaffold(mol, scaffold):
    """Returns whether mol contains the given scaffold.
  NOTE: This is more advanced than simply computing scaffold equality (i.e.
  scaffold(mol_a) == scaffold(mol_b)). This method allows the target scaffold to
  be a subset of the (possibly larger) scaffold in mol.
  Args:
    mol: RDKit Mol.
    scaffold: String scaffold SMILES.
  Returns:
    Boolean whether scaffold is found in mol.
  """
    pattern = Chem.MolFromSmiles(scaffold)
    matches = mol.GetSubstructMatches(pattern)
    return bool(matches)


def get_largest_ring_size(molecule):
    """Calculates the largest ring size in the molecule.
  Refactored from
  https://github.com/wengong-jin/icml18-jtnn/blob/master/bo/run_bo.py
  Args:
    molecule: Chem.Mol. A molecule.
  Returns:
    Integer. The largest ring size.
  """
    cycle_list = molecule.GetRingInfo().AtomRings()
    if cycle_list:
        cycle_length = max([len(j) for j in cycle_list])
    else:
        cycle_length = 0
    return cycle_length


def penalized_logp(molecule):
    """Calculates the penalized logP of a molecule.
  Refactored from
  https://github.com/wengong-jin/icml18-jtnn/blob/master/bo/run_bo.py
  See Junction Tree Variational Autoencoder for Molecular Graph Generation
  https://arxiv.org/pdf/1802.04364.pdf
  Section 3.2
  Penalized logP is defined as:
   y(m) = logP(m) - SA(m) - cycle(m)
   y(m) is the penalized logP,
   logP(m) is the logP of a molecule,
   SA(m) is the synthetic accessibility score,
   cycle(m) is the largest ring size minus by six in the molecule.
  Args:
    molecule: Chem.Mol. A molecule.
  Returns:
    Float. The penalized logP value.
  """
    #log_p = Descriptors.MolLogP(molecule)
    log_p = MolLogP(molecule) # KP 2022 MolLogP migrated to somewhere else not Descriptors any more
    sas_score = sascorer.calculateScore(molecule)
    largest_ring_size = get_largest_ring_size(molecule)
    cycle_score = max(largest_ring_size - 6, 0)
    return log_p - sas_score - cycle_score


def SMILES_add_hydrogens(molecule):
    """Read in molecules using OBConversion framework.
    Read in SMILES, adds hydrogens, and writes new SMILES.
    Args:
      molecule: Chem.Mol. The current molecule.
    Returns:
      smiles: String. The SMILES string for the current molecule.
    """
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("smi", "smi")
    mol_without_H = openbabel.OBMol()
    obConversion.ReadString(mol_without_H, Chem.MolToSmiles(molecule))
    mol_without_H.AddHydrogens()
    mol_H = obConversion.WriteString(mol_without_H)
    return mol_H


def SMILES_into_PDBQT_ligand(molecule):
    """Convert SMILES with hydrogen into PDBQT format of the target molecule.
    Args:
      molecule: Chem.Mol. The current molecule.
    Returns:
      PDBQT: String. The PDBQT string for the current molecule with hydrogen. 
    """
    mol_H_string = pybel.readstring("smi", SMILES_add_hydrogens(molecule))
    mol_H_string.make3D()
    #pcs = mol_H_string.calccharges(model="qtpie")
    #print('the partial charges that should have been assigned are ', pcs)
    mol_H_PDBQT = mol_H_string.write("pdbqt")
    return mol_H_PDBQT

def Mol_into_PDB_ligand(molecule):
    """Convert SMILES with hydrogen into PDBQT format of the target molecule.
    Args:
      molecule: Chem.Mol. The current molecule.
    Returns:
      PDBQT: String. The PDBQT string for the current molecule with hydrogen.
    """
    mol_H_string = pybel.readstring("smi", SMILES_add_hydrogens(molecule))
    mol_H_string.make3D()
    mol_H_PDB = mol_H_string.write("pdb")
    return mol_H_PDB


def SMILES_into_SDF_ligand(molecule):
    """Convert SMILES with hydrogen into SDF format of the target molecule.
    Args:
      molecule: Chem.Mol. The current molecule.
    Returns:
      SDF: String. The PDBQT string for the current molecule with hydrogen. 
    """
    mol_H_string = pybel.readstring("smi", SMILES_add_hydrogens(molecule))
    mol_H_string.make3D()
    mol_H_SDF = mol_H_string.write("sdf")
    return mol_H_SDF
