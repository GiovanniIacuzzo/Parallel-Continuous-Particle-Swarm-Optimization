"""Frozen-attractor spectral diagnostics, not stochastic convergence guarantees."""
from pathlib import Path
import numpy as np
import pandas as pd
from review_study import ROOT


def main():
    rows=[]
    chi=2/(4.1-2+np.sqrt(4.1**2-4*4.1))
    r=(np.arange(32)+.5)/32
    phi=(2.05*(r[:,None]+r[None,:])).ravel()
    for mu in [.4,.65,.9]:
        alpha=(1-mu)/2
        roots=np.sqrt((alpha**2-phi).astype(complex))
        for dt in [.1,.2,.35,.38125,.5,.7,1.,2.,3.6,10.,30.]:
            continuous=np.exp((-alpha+roots)*dt)
            # Unit discrete PSO: [u';v'] = [[1-phi,mu],[-phi,mu]] [u;v].
            matrix=np.zeros((len(phi),2,2));matrix[:,0,0]=1-phi
            matrix[:,0,1]=mu;matrix[:,1,0]=-phi;matrix[:,1,1]=mu
            rho=np.abs(np.linalg.eigvals(matrix)).max(axis=1)
            constricted=matrix.copy();constricted[:,0,0]=1-chi*phi
            constricted[:,0,1]=chi;constricted[:,1,0]=-chi*phi;constricted[:,1,1]=chi
            rows.append(dict(mu=mu,dt=dt,alpha=alpha,underdamped_fraction=float(np.mean(phi>alpha**2)),
                envelope_contraction=float(np.exp(-alpha*dt)),
                analytic_spectral_radius_median=float(np.median(abs(continuous))),
                phase_median=float(np.median(np.sqrt(np.maximum(0.,phi-alpha**2))*dt)),
                discrete_spectral_radius_median=float(np.median(rho)),
                discrete_frozen_stable_fraction=float(np.mean(rho<1)),
                clerc_kennedy_chi=float(chi),
                constriction_stable_fraction=float(np.mean(np.abs(np.linalg.eigvals(constricted)).max(axis=1)<1))))
    out=ROOT/'results/review_v1/stability';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'frozen_spectral_diagnostics.csv',index=False)
    print(out)

if __name__=='__main__':main()
