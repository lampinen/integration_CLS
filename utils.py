import numpy as np
import scipy.optimize as opt

def fixuv(uv,s,vv):
    w = np.arange(len(vv[0,:]))
    for i in range(uv.shape[0]):
        sm = np.dot(w,vv[i,:])
        if sm < 0:
            uv[:,i]=-1*uv[:,i];vv[i,:]=-1*vv[i,:]
        if s[i] < .01:
            vv[i,:] = 0.0*vv[i,:] #s[i]*vv[i,:]
            uv[:,i] = 0.0*uv[:,i]
    return uv,vv

def fixuvd2(uv,s,vv):
    w = np.arange(len(vv[0,:]))
    for i in range(uv.shape[0]):
        sm = np.dot(w,vv[i,:])
        if sm < 0:
            uv[:,i]=-1*uv[:,i];vv[i,:]=-1*vv[i,:]
        if s[i] < .01:
            vv[i,:] = 0.0*vv[i,:] #s[i]*vv[i,:]
            uv[:,i] = 0.0*uv[:,i]
    vv[2,:] = -vv[2,:]; uv[:,2] = -uv[:,2]
    return uv,vv

def replaysample(nrows,ncols,pvec):
    rvec = np.random.uniform(size=nrows)
    svec = np.floor(pvec-rvec)+1
    return(np.tile(np.expand_dims(svec,axis=1),(1,ncols)))

def replaypartial(Source,pdrop,keep):
    RandMat = np.random.uniform(0,1,size=Source.shape)-pdrop
    Out = np.multiply(Source,(np.floor(RandMat)+1))
    Out[keep,:] = Source[keep,:]
    return(Out)

def gdo(u,sd,v,i):
    return(np.matmul(np.transpose(np.array(u[:,i],ndmin=2)),np.matmul(np.array(sd[i,i],ndmin=2),np.array(v[i,:],ndmin=2))))

#         j = 0
#         while np.abs(uv[j,i])<1e-2:
#             j = j+1
#         if uv[j,i]<0:
