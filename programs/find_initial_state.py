import sys
from pathlib import Path
# Find jVMC package
sys.path.append(sys.path[0]+"/..")

import os

import jax
from jax.config import config
config.update("jax_enable_x64", True)

#print(" rank %d"%(rank), jax.devices())

import jax.random as random
import flax
import flax.nn as nn
import jax.numpy as jnp

import numpy as np

import time
import json

import jVMC
import jVMC.operator as op
from jVMC.util import measure, ground_state_search, OutputManager
import jVMC.mpi_wrapper as mpi
import jVMC.activation_functions as act_funs
import jVMC.global_defs as global_defs

from functools import partial

# with open(sys.argv[1],'r') as f:
#     inp = json.load(f)

# wdir=inp["general"]["working_directory"]
wdir = "/p/scratch/qudyngpu/few_body_ops/test"
if mpi.rank == 0:
    try:
        os.makedirs(wdir)
    except OSError:
        print ("Creation of the directory %s failed" % wdir)
    else:
        print ("Successfully created the directory %s " % wdir)

global_defs.set_pmap_devices(jax.devices()[mpi.rank % jax.device_count()])
print(" -> Rank %d working with device %s" % (mpi.rank, global_defs.devices()), flush=True)

# L = inp["system"]["L"]
L = 2

# Initialize output manager
# outp = OutputManager(wdir+inp["general"]["data_output"], append=inp["general"]["append_data"])


# Set up hamiltonian for ground state search
hamiltonianGS = op.Operator()
hamiltonianGS.add(
    (
        op.PzDiag(0),
        op.PzDiag(1)
    )
)

sampler = jVMC.sampler.ExactSampler(L, lDim=4)


print(sampler.get_basis().shape)
print(sampler.get_basis())
print()

configurations, matrix_element = hamiltonianGS.get_s_primes(sampler.get_basis())

print(configurations)
print()
print(matrix_element)

# tdvpEquation = jVMC.tdvp.TDVP(sampler, snrTol=inp["time_evol"]["snr_tolerance"],
#                                        svdTol=inp["time_evol"]["svd_tolerance"],
#                                        rhsPrefactor=1.,
#                                        diagonalShift=inp["gs_search"]["init_regularizer"], makeReal='real')

# t=inp["time_evol"]["t_init"]

# fromCheckpoint = False
# if t<0:
#     outp.set_group("time_evolution")

#     t, weights = outp.get_network_checkpoint(t)

#     psi.set_parameters(weights)

#     fromCheckpoint = True

# else:
#     # Perform ground state search to get initial state
#     outp.print("** Ground state search")
#     outp.set_group("ground_state_search")

#     if "numSamplesGS" in inp["sampler"]:
#         sampler.set_number_of_samples(inp["sampler"]["numSamplesGS"])
#     ground_state_search(psi, hamiltonianGS, tdvpEquation, sampler,
#                         numSteps=inp["gs_search"]["num_steps"], varianceTol=inp["gs_search"]["convergence_variance"]*L**2,
#                         stepSize=1e-2, observables=observables, outp=outp)

#     sampler.set_number_of_samples(inp["sampler"]["numSamples"])

# # Time evolution
# outp.print("** Time evolution")
# outp.set_group("time_evolution")

# reim = 'imag'
# if "tdvp_make_real" in inp["time_evol"]:
#     reim = inp["time_evol"]["tdvp_make_real"]

# observables["energy"] = hamiltonian
# tdvpEquation = jVMC.tdvp.TDVP(sampler, snrTol=inp["time_evol"]["snr_tolerance"],
#                                        svdTol=inp["time_evol"]["svd_tolerance"],
#                                        rhsPrefactor=1.j, diagonalShift=0., makeReal=reim)

# def norm_fun(v, df=lambda x:x):
#     return jnp.real(jnp.conj(jnp.transpose(v)).dot(df(v)))

# stepper = jVMC.stepper.AdaptiveHeun(timeStep=inp["time_evol"]["time_step"], tol=inp["time_evol"]["stepper_tolerance"])

# tmax=inp["time_evol"]["t_final"]

# if not fromCheckpoint:
#     outp.start_timing("measure observables")
#     obs = measure(observables, psi, sampler)
#     outp.stop_timing("measure observables")

#     outp.write_observables(t, **obs)

# while t<tmax:
#     tic = time.perf_counter()
#     outp.print( ">  t = %f\n" % (t) )

#     # TDVP step
#     dp, dt = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, numSamples=inp["sampler"]["numSamples"], outp=outp, normFunction=partial(norm_fun, df=tdvpEquation.S_dot))
#     psi.set_parameters(dp)
#     t += dt
#     outp.print( "   Time step size: dt = %f" % (dt) )
#     tdvpErr, tdvpRes = tdvpEquation.get_residuals()
#     outp.print( "   Residuals: tdvp_err = %.2e, solver_res = %.2e" % (tdvpErr, tdvpRes) )

#     # Measure observables
#     outp.start_timing("measure observables")
#     obs = measure(observables, psi, sampler)
#     outp.stop_timing("measure observables")

#     # Write observables
#     outp.write_observables(t, **obs)
#     # Write metadata
#     outp.write_metadata(t, tdvp_error=tdvpErr,
#                            tdvp_residual=tdvpRes,
#                            SNR=tdvpEquation.get_snr(),
#                            spectrum=tdvpEquation.get_spectrum())
#     # Write network parameters
#     outp.write_network_checkpoint(t, psi.get_parameters())

#     outp.print("    Energy = %f +/- %f" % (obs["energy"]["mean"], obs["energy"]["MC_error"]))

#     outp.print_timings(indent="   ")

#     toc = time.perf_counter()
#     outp.print( "   == Total time for this step: %fs\n" % (toc-tic) )
