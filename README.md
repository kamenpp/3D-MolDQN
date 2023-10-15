# 3D-MolDQN

To complement the thesis submitted for Part II project in Chemistry.

## How to calculate the protrusion of an arbitrary ligand after being docked using AutoDock Vina?

```python calculate_protrusion.py --envelope "celecoxib_envelope_02_with_01_resolution.csv" --ligand "ligand_61_300_0002_MPro_docking_10step_out.pdbqt"```

This would also work with any other 3D structural format of the ligand after a small modification in the script to read the correct columnds in the file (e.g. a PDB file)

## How to Build your own Envelope?

Execute (something similar to) the following snippet in a Terminal:

```/opt/schrodinger/suites2022-3/run python3 /path/to/compute_substrate_envelope_custom.py  *.pdb  -a -r 0.1 -jobname hiv1 -o  sub_envelope_hiv1.vis```

```/opt/schrodinger/suites2022-3/run python3 /path/to/compute_substrate_envelope_custom.py  nat_sub_1.pdb nat_sub_2.pdb nat_sub_3.pdb  -a -r 0.1 -jobname hiv1 -o  sub_envelope_hiv1.csv```

Obviously you need to have installed Schroedinger Suite in order to execute the script through Schroedinger.

## Set up environemnt

```conda env create -f environment.yml```

## Execution

Execute the following in order to experiment with darunavir as the starting molecule:

```python docking_env_penalty.py  --docking="./config/docking_specs.json"  --cache="False" --smarts_mask="S(=O)(=O)"  --model_dir="./outputs_24_01_dep/save"  --hparams="./config/multi_obj_dqn_rxn3Ddqn_remote.json" --start_molecule="CC1(C)[C@@H]2[C@@H](C(=O)S)NC[C@@H]21"  --target_molecule="CC1(C)[C@@H]2[C@@H](C(=O)S)NC[C@@H]21"```
