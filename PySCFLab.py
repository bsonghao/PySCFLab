###                                 Welcome to PySCFLab! 
###     This program is developed to porvide a new electronic structure package for the
### CHEM 400/740 Introduction to Computational Quantum Chemistry course offered by Professor Marcel Nooijen.

###     Package Information: The Python-based simulations of chemistry framework(PySCF) is a general-purpose electronic
### structure platform designed from the ground up to emphasize code simplicity, so as to facilitate new method
### development and enable flexible computational workflows.
###     See full documentations for PySCF from: https://sunqm.github.io/pyscf/index.html

###     Purpose: PySCF is a very large and complex program that has many functions and subroutines to complete quantum
### chemical calculations as other packages like Gaussian/Gaussview, ORCA, MOLPRO or ACES2. It is important to provide
### an easy-to-use and user-interface-friendly program(or helper program) that make full use of the PySCF package.
###     Hence, this program is named PySCFLab. You can always import this module to run your calculation.

###     update log:
### Version 0.1: added run
### Version 0.2: added run_batch, parse_kwargs
### Version 0.3: added read_molecule
### Version 0.4: added XYZtoMF
### Version 0.5: enable pandas for running mutiple calculations at once (ref: https://github.com/nmardirossian)
### Version 0.6: Included a huge molecule library

###     Bug report or Questions: Contact Haobo Liu at h349liu@uwaterloo.ca
### Before you start running this program, make sure you installed the PySCF package and the pandas package in your computer.
### If you are not sure, try to install them through command "pip3 install pyscf" and "pip3 install pandas"

### How to use:

###     Make sure your calculation at least have a 'method', a 'molecule' and a 'basis' to run. For example,
### run('HF', '''O 0,0,0; H 0,1,0; H 0,0,1''', 'cc-pvtz') will run the Hartree-Fock Calculation for H2O molecule.
### create input file(stand alone)

###     Note that the "run" function will not produce any output/print, you need to add your own print function. For example,
### print("My result is ", run('HF', '''O 0,0,0; H 0,1,0; H 0,0,1''', 'cc-pvtz')) should produce:
### My result is -76.04562579701266
###     Where the '-76.04562579701266' an energy of -76.04562579701266 Hartree.

### Determine your method in one of: 

### 1. HF = Hartree-Fock
### 2. KS = Kohn-Sham
### 3. DFT = Density functional theory
### 4. RHF = Non-relativistic Hartree-Fock analytical nuclear gradients
### 5. ROHF = Non-relativistic ROHF analytical nuclear gradients
### 6. UHF = Non-relativistic unrestricted Hartree-Fock analytical nuclear gradients
### 7. RKS = Non-relativistic restricted Kohn-Sham
### 8. ROKS = Non-relativistic ROKS analytical nuclear gradients
### 9. UKS = Non-relativistic Unrestricted Kohn-Sham
### 10. UDFT = Non-relativistic Unrestricted Density functional theory

### More features available see https://sunqm.github.io/pyscf/overview.html#features for detail.

### Determine your Geometry for the calculation:

### Define your molecule with coordinate sets, for example:
### '''O 0,0,0; H 0,1,0; H 0,0,1'''

### '''O 0 0 0; H  0 1 0; H 0 0 1'''

### '''O 0 0 0
### H  0 1 0
### H 0 0 1'''

###     I provide 5931 xyz files in the /data folder, feel free to test or create your own Geometry!
###     If you want to label an atom to distinguish it from the rest, you can prefix or suffix number or special characters, for example
### 1234567890~!@#$%^&*()_+.?:<>[]{}|

###     You can always define your molecule with other methods, see https://sunqm.github.io/pyscf/tutorial.html#geometry for detail.

###     Basis sets available in PySCF now: https://sunqm.github.io/pyscf/_modules/pyscf/gto/basis.html. Larger basis set will lead to
### longer calculation time and more accurate result.

###     By default, run will perform a restricted calculation if the molecule is closed-shell (no unpaired electrons or spin=0) 
### and an unrestricted calculation if the molecule is open-shell (spin>0). This default can be overwritten in several ways. 
### Either the method can be explicitly stated (i.e., RHF, ROHF, or UHF), or the scf_type input can be set explicitly (R, RO, or U).
###     You can determine your charge or spin by adding them to your 'run' function, for example:
### run('HF', '''O 0,0,0; H 0,1,0; H 0,0,1''', 'cc-pVDZ', charge=1, spin=1)

###     Another useful feature that can be turned on is prop. When prop=True, a variety of molecular properties (dipole moment, S^2
### and 2S+1 values, as well as two different population analysis results) are computed after the SCF converges. For example:
### oh="""O 0.0000000000 0.0000000000 0.0000000000; H 0.0000000000 0.0000000000 0.9706601900"""
### run('UHF', oh, 'cc-pVDZ', spin=1, prop=True) will produce:

### Dipole moment(X, Y, Z, Debye):  0.00000, -0.00000,  1.80400
### (S^2, 2S+1): (0.7546117327980264, 2.004606428003289)
### Mulliken population analysis
### O : -0.3232140255988618
### H : 0.32321402559886236
### Mulliken population analysis, based on meta-Lowdin AOs
### O : -0.1849955872165765
### H : 0.18499558721657794

###     When dealing with radicals, multireference systems, or systems that can break spin symmetry, it is possible to land on 
### an SCF solution that is unstable. For these notorious cases, the stable=True setting will attempt to resolve the 
### instability 3 times (a setting that can be modified by stable_cyc). For example:
### c2="""C 0.0 0.0 0.0;C 0.0 0.0 1.24"""
### run('RHF', c2, 'cc-pVDZ', stable=True) will produce:

### Initial SCF energy:  -75.386817114
### Performing Stability Analysis (up to  3 iterations)
### Unstable!
### Updated SCF energy:  -75.4159592748
### Stable!

### The function run also can read molecule files from disk, for example:
### print run('HF', 'your_file_name.xyz', 'cc-pVDZ', datapath='Your_folder_name/')

### If you want to run mutiple calculations at once on slurm, the function run_batch is very useful.
### For example, define your database at the beginning by 
### mydbase = pandas.DataFrame()
###     then run: 
### filenames=['11_Ar-Ar_dim_NC15.xyz', '11_Ar-Ar_monA_NC15.xyz', '11_Ar-Ar_monB_NC15.xyz']
### mydbase=run_batch(['UHF', 'RHF'], filenames, ['cc-pVDZ', 'def2-SVP'], datapath='data/')
###     and print your own charts! Enjoy!
### print(tabulate(mydbase[['molecule', 'method', 'basis', 'charge', 'spin', 'e_tot', 'converged']], headers='keys', tablefmt='psql'))

import pyscf
#import os
import sys
from pyscf import gto, scf, dft
import numpy
import itertools
import pandas
from tabulate import tabulate
pandas.set_option('display.max_columns', 500)

# assume we only take one argument from command-line
# and that argument is the path
path = sys.argv[1]
input_file = open(path, 'r')
input_str = input_file.read() # this contains everything

split_result = input_str.split('----------------------------------------------')
function_selection = split_result[0]
basis_selection = split_result[1]
method_selection = split_result[2]
geo_info = split_result[3]
Advanced_command = split_result[4]

charge = spin = 0
molecule = 0

# use split_result as buffer
split_result = function_selection.split('\n')
for rows in split_result:
    if "Please select your functions:" in rows:
        function_selection = int(rows.strip("Please select your functions:"))
        break
#success
#print(function_selection)

split_result = basis_selection.split('\n')
for rows in split_result:
    if "Please select your Basis set:" in rows:
        basis_selection = int(rows.strip("Please select your Basis set:"))
        break

split_result = method_selection.split('\n')
for rows in split_result:
    if "Please select your method:" in rows:
        method_selection = int(rows.strip("Please select your method:"))
        break

split_result = geo_info.split('\n')
split_result[:] = (value for value in split_result if value != '')
split_result.remove('Please provide the gerometry for the molecule:')
#print(split_result)

try:
    int(split_result[0])
except ValueError:
    try:
        charge = int(split_result[0].split(' ')[0])
        spin = int(split_result[0].split(' ')[1]) - 1
    except ValueError:
        molecule = split_result
    else:
        molecule = '\n'.join(split_result[1:])
else:
    if int(split_result[0]) == len(split_result) - 2:
        molecule = '\n'.join(split_result[2:])
        try:
            charge = int(split_result[1].split(' ')[0])
            spin = int(split_result[1].split(' ')[1])-1
        except ValueError:
            pass
    else:
        print("THIS IS NOT A VALID XYZ FILE")

def run(method, molecule, basis,
        charge=0, ecp={}, spin=0, unit='Angstrom',
        conv_tol=1e-12, conv_tol_grad=1e-8, direct_scf_tol=1e-13,
        init_guess='minao', level_shift=0, max_cycle=100, max_memory=8000,
        xc=None, nlc='', xc_grid=3, nlc_grid=1, small_rho_cutoff=1e-7,
        atomic_radii='BRAGG', becke_scheme='BECKE', prune='NWCHEM',
        radi_method='TREUTLER_AHLRICHS', radii_adjust='TREUTLER',
        algo='DIIS', datapath='', getdict=False, lin_dep_thresh=1e-8,
        prop=False, scf_type=None, stable=False, stable_cyc=3, verbose=0):

    # modify inputs arguments that are strings
    method = method.upper()
    atomic_radii = atomic_radii.upper()
    becke_scheme = becke_scheme.upper()
    if isinstance(prune, str):
        prune = prune.upper()
    radi_method = radi_method.upper()
    if isinstance(radii_adjust, str):
        radii_adjust = radii_adjust.upper()
    algo = algo.upper()

    # create molecule object
    mol = gto.Mole()

    # set mol object attributes
    try:
        gto.Mole(atom=molecule, charge=charge, spin=spin).build()
    except KeyError:
        (mol.atom, charge, spin) = read_molecule(datapath + molecule)
    else:
        mol.atom = molecule
    mol.basis = basis
    mol.charge = charge
    mol.ecp = ecp
    mol.spin = spin
    mol.unit = unit
    mol.verbose = verbose
    mol.build()

    # check method for density functional
    DFT = False
    if method != 'HF':
        try:
            dft.libxc.parse_xc(method)
        except KeyError:
            pass
        else:
            xc = method
            method = 'KS'

    # determine restricted/unrestricted if unspecified
    # automatically sets R if closed-shell and U if open-shell
    if method in ['HF', 'KS', 'DFT'] and scf_type is None:
        if spin == 0:
            scf_type = 'R'
        else:
            scf_type = 'U'

    # create HF/KS object
    if method in ['RHF', 'ROHF'] or (method == 'HF' and scf_type in ['R', 'RO']):
        mf = scf.RHF(mol)
        scf_type = 'R'
    elif method == 'UHF' or (method == 'HF' and scf_type == 'U'):
        mf = scf.UHF(mol)
        scf_type = 'U'
    elif method in ['RKS', 'ROKS', 'RDFT', 'RODFT'] or (method in ['KS', 'DFT'] and scf_type in ['R', 'RO']):
        mf = scf.RKS(mol)
        scf_type = 'R'
        DFT = True
    elif method in ['UKS', 'UDFT'] or (method in ['KS', 'DFT'] and scf_type == 'U'):
        mf = scf.UKS(mol)
        scf_type = 'U'
        DFT = True
    else:
        print( "CRASH 1")

    # set HF attributes
    mf.conv_check = False
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.direct_scf_tol = direct_scf_tol
    mf.init_guess = init_guess
    mf.level_shift = level_shift
    mf.max_cycle = max_cycle
    mf.max_memory = max_memory
    mf.verbose = verbose

    # set KS attributes
    if DFT:

        mf.xc = xc
        if mf.xc is None:
            print( "CRASH 2")
        mf.nlc = nlc

        if isinstance(xc_grid, int):
            mf.grids.level = xc_grid
        elif isinstance(xc_grid, tuple) or isinstance(xc_grid, dict):
            mf.grids.atom_grid = xc_grid
        else:
            print( "CRASH 3")

        if isinstance(nlc_grid, int):
            mf.nlcgrids.level = nlc_grid
        elif isinstance(nlc_grid, tuple) or isinstance(nlc_grid, dict):
            mf.nlcgrids.atom_grid = nlc_grid
        else:
            print( "CRASH 4")

        if atomic_radii == 'BRAGG':
            mf.grids.atomic_radii = dft.radi.BRAGG_RADII
            mf.nlcgrids.atomic_radii = dft.radi.BRAGG_RADII
        elif atomic_radii == 'COVALENT':
            mf.grids.atomic_radii = dft.radi.COVALENT_RADII
            mf.nlcgrids.atomic_radii = dft.radi.COVALENT_RADII
        else:
            print( "CRASH 5")

        if becke_scheme == 'BECKE':
            mf.grids.becke_scheme = dft.gen_grid.original_becke
            mf.nlcgrids.becke_scheme = dft.gen_grid.original_becke
        elif becke_scheme == 'STRATMANN':
            mf.grids.becke_scheme = dft.gen_grid.stratmann
            mf.nlcgrids.becke_scheme = dft.gen_grid.stratmann
        else:
            print( "CRASH 6")

        if prune == 'NWCHEM':
            mf.grids.prune = dft.gen_grid.nwchem_prune
            mf.nlcgrids.prune = dft.gen_grid.nwchem_prune
        elif prune == 'SG1':
            mf.grids.prune = dft.gen_grid.sg1_prune
            mf.nlcgrids.prune = dft.gen_grid.sg1_prune
        elif prune == 'TREUTLER':
            mf.grids.prune = dft.gen_grid.treutler_prune
            mf.nlcgrids.prune = dft.gen_grid.treutler_prune
        elif prune == 'NONE' or prune is None:
            mf.grids.prune = None
            mf.nlcgrids.prune = None
        else:
            print( "CRASH 7")

        if radi_method in ['TREUTLER_AHLRICHS', 'TREUTLER', 'AHLRICHS']:
            mf.grids.radi_method = dft.radi.treutler_ahlrichs
            mf.nlcgrids.radi_method = dft.radi.treutler_ahlrichs
        elif radi_method == 'DELLEY':
            mf.grids.radi_method = dft.radi.delley
            mf.nlcgrids.radi_method = dft.radi.delley
        elif radi_method in ['MURA_KNOWLES', 'MURA', 'KNOWLES']:
            mf.grids.radi_method = dft.radi.mura_knowles
            mf.nlcgrids.radi_method = dft.radi.mura_knowles
        elif radi_method in ['GAUSS_CHEBYSHEV', 'GAUSS', 'CHEBYSHEV']:
            mf.grids.radi_method = dft.radi.gauss_chebyshev
            mf.nlcgrids.radi_method = dft.radi.gauss_chebyshev
        else:
            print( "CRASH 8")

        if radii_adjust == 'TREUTLER':
            mf.grids.radii_adjust = dft.radi.treutler_atomic_radii_adjust
            mf.nlcgrids.radii_adjust = dft.radi.treutler_atomic_radii_adjust
        elif radii_adjust == 'BECKE':
            mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
            mf.nlcgrids.radii_adjust = dft.radi.becke_atomic_radii_adjust
        elif radii_adjust == 'NONE' or radii_adjust is None:
            mf.grids.radii_adjust = None
            mf.nlcgrids.radii_adjust = None
        else:
            print( "CRASH 9")

        mf.small_rho_cutoff = small_rho_cutoff

    # select optimizer
    if algo == 'DIIS':
        mf.diis = True
    elif algo == 'ADIIS':
        mf.diis = scf.diis.ADIIS()
    elif algo == 'EDIIS':
        mf.diis = scf.diis.EDIIS()
    elif algo == 'NEWTON':
        mf = mf.newton()
    else:
        print( "CRASH 10")

    # run SCF
    mf.kernel()

    # stability analysis (optional)
    if stable:
        print( 'Initial SCF energy: ', mf.e_tot)
        print( 'Performing Stability Analysis (up to ', stable_cyc, 'iterations)')
        for i in range(stable_cyc):
            new_mo_coeff = mf.stability(internal=True, external=False)[0]
            if numpy.linalg.norm(numpy.array(new_mo_coeff) - numpy.array(mf.mo_coeff)) < 10**-14:
                print( "Stable!")
                break
            else:
                print( "Unstable!")
                if scf_type == 'U':
                    n_alpha = numpy.count_nonzero(mf.mo_occ[0])
                    n_beta = numpy.count_nonzero(mf.mo_occ[1])
                    P_alpha = numpy.dot(new_mo_coeff[0][:, :n_alpha], new_mo_coeff[0].T[:n_alpha])
                    P_beta = numpy.dot(new_mo_coeff[1][:, :n_beta], new_mo_coeff[1].T[:n_beta])
                    mf.kernel(dm0=(P_alpha, P_beta))
                elif scf_type in ['R', 'RO']:
                    n_alpha = numpy.count_nonzero(mf.mo_occ)
                    P_alpha = 2*numpy.dot(new_mo_coeff[:, :n_alpha], new_mo_coeff.T[:n_alpha])
                    mf.kernel(dm0=(P_alpha))
                else:
                    print( "CRASH 11")
                print( 'Updated SCF energy: ', mf.e_tot)

    # properties
    if prop:

        # dipole moment
        mf.dip_moment()

        # S^2
        if scf_type == 'U':
            (ssq, mult) = mf.spin_square()
            print( '(S^2, 2S+1):', (ssq, mult))
        elif scf_type in ['R', 'RO']:
            S = float(mol.spin)/2.
            (ssq, mult) = (S*(S+1.), 2.*S+1.)
            print( '(S^2, 2S+1):', (ssq, mult))
        else:
            print( "CRASH 12")

        # population analysis
        print( "Mulliken population analysis")
        for ia in range(mol.natm):
            symb = mol.atom_symbol(ia)
            print( symb, ':', mf.mulliken_meta(verbose=verbose)[1][ia])

        print( "Mulliken population analysis, based on meta-Lowdin AOs")
        for ia in range(mol.natm):
            symb = mol.atom_symbol(ia)
            print( symb, ':', mf.mulliken_pop(verbose=verbose)[1][ia])

    # return either just an energy or (energy,dict)
    if getdict:
        mydict = dict()
        mydict.update(mol.__dict__)
        mydict.update(mf.__dict__)
        if DFT:
            mydict.update(mf.grids.__dict__)
        return mf.e_tot, mydict
    else:
        return mf.e_tot


def run_batch(method, molecule, basis, database=pandas.DataFrame(), **kwargs):
    mand_args = ['method', 'molecule', 'basis']
    num_mand_args = len(mand_args)
    if not isinstance(method, list):
        method = [method]
    if not isinstance(molecule, list):
        molecule = [molecule]
    if not isinstance(basis, list):
        basis = [basis]
    for i in itertools.product(method, molecule, basis, list(parse_kwargs(**kwargs))):
        mydict = dict()
        mydict.update(i[num_mand_args:][0])
        (en, rundict) = run(*i[:num_mand_args], getdict=True, **i[num_mand_args:][0])
        mydict.update(rundict)
        for num, val in enumerate(mand_args):
            if val == 'molecule':
                try:
                    read_molecule(mydict['datapath'] + i[num])
                except IOError:
                    mydict.update({val: XYZtoMF(i[num])})
                else:
                    mydict.update({val: i[num]})
            else:
                mydict.update({val: i[num]})
        database = database.append(mydict, ignore_index=True)

    return database


def parse_kwargs(**kwargs):
    items = kwargs.items()
    keys = [key for key, value in items]
    sets = [value for key, value in items]
    for index, item in enumerate(sets):
        if not isinstance(item, list):
            sets[index] = [sets[index]]
    for values in itertools.product(*sets):
        yield dict(zip(keys, values))


def read_inpfile(path):
    ### read from .inp file
    ### use the standard inp file format to make everything works
    
    ### read-only first
    open(path, 'r')

def read_molecule(path):

    charge = spin = 0
    with open(path, 'r') as myfile:
        output = myfile.read()
        output = output.lstrip()
        output = output.rstrip()
        output = output.split('\n')

    try:
        int(output[0])
    except ValueError:
        try:
            charge = int(output[0].split(' ')[0])
            spin = int(output[0].split(' ')[1]) - 1
        except ValueError:
            molecule = output
        else:
            molecule = '\n'.join(output[1:])
    else:
        if int(output[0]) == len(output) - 2:
            molecule = '\n'.join(output[2:])
            try:
                charge = int(output[1].split(' ')[0])
                spin = int(output[1].split(' ')[1])-1
            except ValueError:
                pass
        else:
            print("THIS IS NOT A VALID XYZ FILE")

    return (molecule, charge, spin)


def XYZtoMF(inp):

    return filter(lambda x: x.isalpha(), inp)

output_file = open("result.txt","w")

output_file.write("Welcome to PySCFLab!\nThis is your output file, your input file is : ")
output_file.write(path)
output_file.write("\n--------------------------------------\n")

if function_selection == 1:
    #energy calculation
    output_file.write("Function selected: Energy\n")
    output_file.write("Method used: ")
    
    if method_selection == 1:
        method_str = "HF"
    elif method_selection == 2:
        method_str = "KS"
    elif method_selection == 3:
        method_str = "DFT"
    elif method_selection == 4:
        method_str = "RHF"
    elif method_selection == 5:
        method_str = "ROHF"
    elif method_selection == 6:
        method_str = "UHF"
    elif method_selection == 7:
        method_str = "RKS"
    elif method_selection == 8:
        method_str = "ROKS"
    elif method_selection == 9:
        method_str = "UKS"
    elif method_selection == 10:
        method_str = "UDFT"
    
    output_file.write(method_str + "\n")
    output_file.write("Basis sets: ")

    if basis_selection == 1:
        basis_str = "STO-3G"
    elif basis_selection == 2:
        basis_str = "3-21G"
    elif basis_selection == 3:
        basis_str = "6-31G"
    elif basis_selection == 4:
        basis_str = "6-311G"
    elif basis_selection == 5:
        basis_str = "cc-pVDZ"
    elif basis_selection == 6:
        basis_str = "cc-pVTZ"
    elif basis_selection == 7:
        basis_str = "cc-pvQz"
    elif basis_selection == 8:
        basis_str = "LanL2DZ"
    elif basis_selection == 9:
        basis_str = "LanL2TZ"
    
    output_file.write(basis_str + "\n")
    output_file.write("Input geometry: \n" + molecule + "\n")
    run(method_str, molecule, basis_str, stable = True)


#if function_selection == 2:
    #optimization
    #mf = scf.RHF(mol)
    #mol_eq = optimize(mf)
    #print(mol_eq.atom_coords()) #this will produce the xyz file

#if function_selection == 3:
    #frequency
    #mf = scf.RHF(mol).run()
    #h = mf.Hessian().kernel()
    #results = harmonic_analysis(mol, h)
    #results = thermo(mf, results['freq_au'], 298.15, 101325)
    #for key in (results):
        #print (key,results[key])


###-----------------------------------------------------------
#h2="""
#H 0.0 0.0 0.0
#H 0.0 0.0 0.74"""
#print(run('HF', h2, 'cc-pVDZ'))
#oh="""
#O 0.0000000000 0.0000000000 0.0000000000
#H 0.0000000000 0.0000000000 0.9706601900"""

 
# unrestricted
#print ('HF: ', run('HF', oh, 'cc-pVDZ', spin=1))
#print ('UHF: ', run('UHF', oh, 'cc-pVDZ', spin=1))
#print ('HF/U: ', run('HF', oh, 'cc-pVDZ', spin=1, scf_type='U'))


# restricted
#print ('RHF: ', run('RHF', oh, 'cc-pVDZ', spin=1))
#print ('ROHF: ', run('ROHF', oh, 'cc-pVDZ', spin=1))
#print ('HF/R: ', run('HF', oh, 'cc-pVDZ', spin=1, scf_type='R'))
#print ('HF/RO: ', run('HF', oh, 'cc-pVDZ', spin=1, scf_type='RO'))

#run('UHF', oh, 'cc-pVDZ', spin=1, prop=True)
#print("result = ",run('HF',C2H4, 'cc-pvtz'))

#mydbase = pandas.DataFrame()
#mydbase = run_batch(['UHF', 'RHF', 'ROHF', 'HF'], ['01a_water_dimAB_3B-69.xyz','h2o_SW49.xyz','20_He-He_monB_NC15.xyz'], ['3-21G', '6-31G'], database=mydbase, datapath='data/')
#print(tabulate(mydbase[['molecule', 'method', 'basis', 'charge', 'spin', 'e_tot', 'converged']], headers='keys', tablefmt='psql'))

#os.system("rm tmp*")