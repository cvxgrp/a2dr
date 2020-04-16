"""
Copyright 2019 Anqi Fu, Junzi Zhang

This file is part of A2DR.

A2DR is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

A2DR is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with A2DR. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import codecs
import os.path

# code for single sourcing versions
# reference: https://packaging.python.org/guides/single-sourcing-package-version/
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

def solve_vec(x, var_type='vec'):
	# reshape to vector format for input to a2dr
	if var_type == 'vec':
		return x, x.shape
	elif var_type == 'mat_C':
		return x.ravel(order='C'), x.shape
	elif var_type == 'mat_F':
		return x.ravel(order='F'), x.shape
	elif var_type == 'mat_symm':
		# lower triangular part, row-wise stacking
		if x.shape[0] != x.shape[1] or np.linalg.norm(x-x.T) != 0:
			raise ValueError("input must be square and symmetric")
		mask = np.ones(Q.shape, dtype=bool)
		np.fill_diagonal(mask, 0)
		x[mask] *= np.sqrt(2)
		ind = np.tril_indices(x.shape[0])
		return x[ind], x.shape
	else:
		raise ValueError("var_type = must be vec, mat_C, mat_F or mat_symm")

def solve_mat(x, shape=None, var_type='vec'):
	# reshape back to the original format after running a2dr
	if var_type == 'vec':
		return x
	elif var_type == 'mat_C':
		if shape == None:
			raise ValueError("shape must be provided for var_type = mat_C")
		return x.reshape(shape, order='C')
	elif var_type == 'mat_F':
		if shape == None:
			raise ValueError("shape must be provided for var_type = mat_F")
		return x.reshape(shape, order='F')
	elif var_type == 'mat_symm':
		# lower triangular part, row-wise stacking
		if x.shape[0] != x.shape[1] or np.linalg.norm(x-x.T) != 0:
			raise ValueError("input must be square and symmetric")
		if shape == None:
			raise ValueError("shape must be provided for var_type = mat_symm")
		ind = np.tril_indices(shape[0])
		ind_u = np.triu_indices(shape[0])
		y = np.zeros(shape)
		y[ind] = x
		y[ind_u] = y.T[ind_u]
		return y
	else:
		raise ValueError("var_type = must be vec, mat_C, mat_F or mat_symm")
