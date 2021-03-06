from mpi4py import MPI
import jax
import jax.numpy as jnp
import numpy as np

import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")

import jVMC.global_defs as global_defs

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
commSize = comm.Get_size()

globNumSamples=0
myNumSamples=0

from functools import partial
import time
communicationTime = 0.

#@partial(jax.pmap, axis_name='i')
#def _sum_up_pmapd(data):
#    s = jnp.sum(data, axis=0)
#    return jax.lax.psum(s, 'i')
#
#@partial(jax.pmap, axis_name='i', in_axes=(0,None))
#def _sum_sq_pmapd(data,mean):
#    s = jnp.linalg.norm(data - mean, axis=0)**2
#    return jax.lax.psum(s, 'i')
#
#@partial(jax.pmap, axis_name='i', in_axes=(0,None,0))
#def _sum_sq_withp_pmapd(data,mean,p):
#    s = jnp.conj(data-mean).dot(p*(data-mean))
#    return jax.lax.psum(s, 'i')

def _cov_helper_with_p(data, p):
    return jnp.expand_dims(
                jnp.matmul( jnp.conj(jnp.transpose(data)), jnp.multiply(p[:,None],data) ),
                axis=0
            )

def _cov_helper_without_p(data):
    return jnp.expand_dims(
                jnp.matmul( jnp.conj(jnp.transpose(data)), data ),
                axis=0
            )

_sum_up_pmapd = None
_sum_sq_pmapd = None 
_sum_sq_withp_pmapd = None
mean_helper = None
cov_helper_with_p = None
cov_helper_without_p = None

pmapDevices = None
jitDevice = None

import collections
def pmap_devices_updated():

    if collections.Counter(pmapDevices) == collections.Counter(global_defs.myPmapDevices):
        return False

    return True


def jit_my_stuff():

    global _sum_up_pmapd
    global _sum_sq_pmapd
    global _sum_sq_withp_pmapd
    global mean_helper
    global cov_helper_with_p
    global cov_helper_without_p
    global pmapDevices
    global jitDevice

    if global_defs.usePmap:
        if pmap_devices_updated():
            _sum_up_pmapd = global_defs.pmap_for_my_devices(lambda x: jax.lax.psum(jnp.sum(x, axis=0), 'i'), axis_name='i')
            _sum_sq_pmapd = global_defs.pmap_for_my_devices(lambda data,mean: jax.lax.psum(jnp.sum(jnp.conj(data-mean) * (data-mean), axis=0), 'i'), axis_name='i', in_axes=(0,None))
            _sum_sq_withp_pmapd = global_defs.pmap_for_my_devices(lambda data,mean,p: jax.lax.psum(jnp.conj(data-mean).dot(p*(data-mean)), 'i'), axis_name='i', in_axes=(0,None,0))
            mean_helper = global_defs.pmap_for_my_devices(lambda data,p: jnp.expand_dims(jnp.dot(p, data), axis=0), in_axes=(0,0))
            cov_helper_with_p = global_defs.pmap_for_my_devices(_cov_helper_with_p, in_axes=(0,0))
            cov_helper_without_p = global_defs.pmap_for_my_devices(_cov_helper_without_p)

            pmapDevices = global_defs.myPmapDevices

    else:
        if jitDevice != global_defs.myDevice:
            _sum_up_pmapd = global_defs.jit_for_my_device(lambda x: jnp.expand_dims(jnp.sum(x,axis=0), axis=0))
            _sum_sq_pmapd = global_defs.jit_for_my_device(lambda data,mean: jnp.expand_dims(jnp.sum(jnp.conj(data-mean) * (data-mean), axis=0), axis=0))
            _sum_sq_withp_pmapd = global_defs.jit_for_my_device(lambda data,mean,p: jnp.expand_dims(jnp.conj(data-mean).dot(p*(data-mean)), axis=0))
            mean_helper = global_defs.jit_for_my_device(lambda data,p: jnp.expand_dims(jnp.dot(p, data), axis=0))
            cov_helper_with_p = global_defs.jit_for_my_device(_cov_helper_with_p)
            cov_helper_without_p = global_defs.jit_for_my_device(_cov_helper_without_p)

            jitDevice = global_defs.myDevice


def distribute_sampling(numSamples, localDevices=None, numChainsPerDevice=1):

    global globNumSamples
    
    if localDevices is None:
    
        globNumSamples = numSamples

        mySamples = numSamples // commSize

        if rank < numSamples % commSize:
            mySamples+=1

        return mySamples

    mySamples = numSamples // commSize

    if rank < numSamples % commSize:
        mySamples+=1

    mySamples = (mySamples + localDevices - 1) // localDevices
    mySamples = (mySamples + numChainsPerDevice - 1) // numChainsPerDevice

    globNumSamples = commSize * localDevices * numChainsPerDevice * mySamples

    return mySamples


def first_sample_id():

    global globNumSamples

    mySamples = globNumSamples // commSize

    firstSampleId = rank * mySamples

    if rank < globNumSamples % commSize:
        firstSampleId += rank
    else:
        firstSampleId += globNumSamples % commSize

    return firstSampleId


def global_sum(data):

    jit_my_stuff()

    data.block_until_ready()
    t0 = time.perf_counter()

    # Compute sum locally
    localSum = np.array( _sum_up_pmapd(data)[0] )
    
    # Allocate memory for result
    res = np.empty_like(localSum, dtype=localSum.dtype)

    #t0 = time.perf_counter()

    # Global sum
    comm.Allreduce(localSum, res, op=MPI.SUM)
    
    global communicationTime 
    communicationTime += time.perf_counter() - t0

    #return jnp.array(res)
    return jax.device_put(res,global_defs.myDevice)


def global_mean(data, p=None):

    jit_my_stuff()

    if p is not None:
        return global_sum(mean_helper(data, p))

    global globNumSamples

    return global_sum(data) / globNumSamples


def global_variance(data, p=None):

    jit_my_stuff()

    data.block_until_ready()
    t0 = time.perf_counter()

    mean = global_mean(data, p)

    # Compute sum locally
    localSum = None
    if p is not None:
        localSum = np.array( _sum_sq_withp_pmapd(data,mean,p)[0] )
    else:
        res = _sum_sq_pmapd(data,mean)[0]
        res.block_until_ready()
        localSum = np.array( res )
    
    # Allocate memory for result
    res = np.empty_like(localSum, dtype=localSum.dtype)
    
    #t0 = time.perf_counter()

    # Global sum
    global globNumSamples
    comm.Allreduce(localSum, res, op=MPI.SUM)
   
    global communicationTime 
    communicationTime += time.perf_counter() - t0

    if p is not None:
        #return jnp.array(res)
        return jax.device_put(res,global_defs.myDevice)
    else:
        #return jnp.array(res) / globNumSamples
        return jax.device_put(res / globNumSamples, global_defs.myDevice)

def global_covariance(data, p=None):

    jit_my_stuff()

    if p is not None:

        return global_sum(cov_helper_with_p(data, p))

    return global_mean(cov_helper_without_p(data))


def bcast_unknown_size(data, root=0):

    dim = None
    buf = None
    if rank == root:
        dim = len(data)
        comm.bcast(dim, root=root)
        buf = np.array(data)
    else:
        dim = comm.bcast(None, root=root)
        buf = np.empty(dim, dtype=np.float64)

    comm.Bcast([buf,dim,MPI.DOUBLE], root=root)

    return buf
    

def get_communication_time():
    global communicationTime
    t = communicationTime
    communicationTime = 0.
    return t

if __name__ == "__main__":
    data=jnp.array(np.arange(720*4).reshape((720,4)))
    myNumSamples = distribute_sampling(720)
    
    myData=data[rank*myNumSamples:(rank+1)*myNumSamples]

    print(global_mean(myData)-jnp.mean(data,axis=0))
