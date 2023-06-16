# 3D-MolDQN

To complement the thesis submitted for Part II project in Chemistry.

## How to Build your own Envelope?

Execute (something similar to) the following snippet in a Terminal:

/opt/schrodinger/suites2022-3/run python3 /path/to/compute_substrate_envelope_custom.py  *.pdb  -a -r 0.1 -jobname hiv1 -o  sub_envelope_hiv1.vis

/opt/schrodinger/suites2022-3/run python3 /path/to/compute_substrate_envelope_custom.py  nat_sub_1.pdb nat_sub_2.pdb nat_sub_3.pdb  -a -r 0.1 -jobname hiv1 -o  sub_envelope_hiv1.csv

Obviously you need to have installed Schroedinger Suit in order to execute the script through Schroedinger.
