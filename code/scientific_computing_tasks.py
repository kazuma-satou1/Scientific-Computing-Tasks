import numpy as np
from scipy import linalg, optimize, interpolate, integrate
import matplotlib.pyplot as plt
from timeit import timeit

# BIT 2118 Scientific Computing - task examples (Tasks 2-20)
# Run: python scientific_computing_tasks.py

def task02():
    c=np.array([18.5,18,17.8,17.5,17.9,19,21.5,23.8,25.6,27.1,28.3,29,29.4,29.1,28.5,27.6,26.2,24.8,23.4,22.1,21,20.2,19.6,19])
    f=1.8*c+32
    return {'mean':c.mean(),'min':c.min(),'max':c.max(),'F_first6':f[:6]}
def task03():
    rng=np.random.default_rng(42); x=rng.uniform(5,250,1_000_000)
    loop=timeit(lambda:[(v/1000)**2 for v in x],number=1)
    vec=timeit(lambda:(x/1000)**2,number=3)/3
    return {'first5':((x/1000)**2)[:5],'loop_s':loop,'vector_s':vec,'speedup':loop/vec}
def task04():
    x=0.1+0.2; t=np.sqrt(2); a=1.414
    return {'sum':x,'exact_equal':x==0.3,'absolute_error':abs(t-a),'relative_error':abs(t-a)/t}
def task05():
    p=np.array([[0.,0.],[1,0],[1,1],[0,1]])
    th=np.deg2rad(45); R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
    S=np.diag([2.,1.5]); return (p@S.T)@R.T
def task06():
    A=np.array([[3.,2.],[1.,2.]]); b=np.array([7.,5.]); return np.linalg.solve(A,b)
def task07():
    A=np.array([[4.,1.],[2.,3.]]); b=np.array([9.,13.]); P,L,U=linalg.lu(A); return linalg.lu_solve((P,L,U),b)
def task08():
    A=np.array([[4.,1.],[2.,3.]]); b=np.array([9.,13.]); x=np.zeros(2)
    for _ in range(30): x += .5*(b-A@x)/np.diag(A)
    return x
# power iteration gives dominant eigenpair
def task09():
    A=np.array([[2.,1.],[1.,2.]]); v=np.ones(2)/np.sqrt(2)
    for _ in range(30): v=A@v; v=v/np.linalg.norm(v)
    return {'eigenvalue':float(v@A@v),'eigenvector':v}
def task10():
    f=lambda x:x**2-2; lo,hi=1.,2.
    for _ in range(60):
        mid=(lo+hi)/2
        if f(lo)*f(mid)<=0: hi=mid
        else: lo=mid
    bisection=(lo+hi)/2
    fixed=1.0
    for _ in range(10): fixed=(fixed+2/fixed)/2
    return {'bisection':bisection,'fixed_point':fixed}
def task11():
    f=lambda x:x**2-2; return {'newton':optimize.newton(f,1.5,fprime=lambda x:2*x),'secant':optimize.newton(f,1.0)}
def task12():
    roots=np.roots([1.,-6.,11.,-6.]); return roots
# Table interpolation and fitted polynomial estimate at 2.5
def task13():
    f=interpolate.interp1d([0,1,2,3],[0,1,4,9],kind='linear'); return float(f(2.5))
def task14():
    x=np.array([0.,1,2,3]); y=np.array([1.,2.9,5.2,6.8]); return np.polyfit(x,y,1)
def task15():
    x=np.array([0.,1,2,3,4]); y=x**2; return np.gradient(y,x)
def task16():
    return integrate.simpson(np.array([0.,1,4,9,16]),x=np.arange(5.))
def task17():
    dt=.1; t=np.arange(0,1+dt,dt); y=np.zeros_like(t)
    for i in range(len(t)-1): y[i+1]=y[i]+dt*( -2*y[i])
    return {'t':t,'y':y}
def task18():
    sol=integrate.solve_ivp(lambda t,y:-2*y,(0,1),[1.],t_eval=[0,.25,.5,.75,1]); return sol.y[0]
def task19():
    r=optimize.minimize(lambda x:(x[0]-3)**2+2*(x[1]+1)**2,[0.,0.],method='BFGS'); return r.x,r.fun
# estimate integral of x^2 on [0,1] by Monte Carlo
def task20(n=100_000,seed=7):
    rng=np.random.default_rng(seed); x=rng.random(n); return float(np.mean(x*x))

if __name__=='__main__':
    for i,fn in enumerate([task02,task03,task04,task05,task06,task07,task08,task09,task10,task11,task12,task13,task14,task15,task16,task17,task18,task19,task20],start=2):
        print(f'Task {i}: {fn()}')

