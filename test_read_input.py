# use sys to enable system access
import sys

# assume we only take one argument from command-line
# and that argument is the path
path = sys.argv[1]
input_file = open(path, 'r')
input_str = input_file.read() # this contains everything
#print("the file is:\n", input_file.read())
# success
#print(input_str.split('----------------------------------------------')[1])
# success
split_result = input_str.split('----------------------------------------------')
function_selection = split_result[0]
basis_selection = split_result[1]
geo_info = split_result[2]
Advanced_command = split_result[3]

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

split_result = geo_info.split('\n')
split_result[:] = (value for value in split_result if value != '')
split_result.remove('Please provide the gerometry for the molecule:')
#print(split_result)
charge = spin = 0
molecule = 0
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
print(molecule, charge, spin)


#read_molecule("01_water-ammonia_0p9_dim_A21x12.xyz")
#print(read_molecule("01_water-ammonia_0p9_dim_A21x12.xyz"))