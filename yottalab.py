"""
This is a procedural interface to the yttalab library

roberto.bucher@supsi.ch

The following commands are provided:

Design and plot commands
  ctrb        - controllability matrix
  acker       - pole placement using Ackermann method
  c2d         - contimous to discrete time conversion
  d2c         - discrete to continous time conversion
  care        - Solve Riccati equation for contimous time systems
  dare        - Solve Riccati equation for discrete time systems
  dlqr        - discrete linear quadratic regulator
  minreal     - minimal state space representation
  dcgain      - return the steady state value of the step response
  
  dsimul      - simulate discrete time systems
  dstep       - step response (plot) of discrete time systems
  dimpulse    - imoulse response (plot) of discrete time systems
  bb_step     - step response (plot) of continous time systems
  
  full_obs    - full order observer
  red_obs     - reduced order observer
  comp_form   - state feedback controller+observer in compact form
  comp_form_i - state feedback controller+observer+integ in compact form
  sysctr      - system+controller+observer+feedback
  set_aw      - introduce anti-windup into controller
  dcgain      - return the steady state value of the step response
  
"""
from matplotlib.pylab import *
from control.matlab import *
from numpy import hstack,vstack,pi
from scipy import zeros,ones,eye,mat,shape,size,size, \
    arange,real,poly,array,diag
from scipy.linalg import det,inv,expm,eig,eigvals,logm
import numpy as np
import scipy as sp
from slycot import sb02od, tb03ad
from scipy.signal import BadCoefficients
import warnings
warnings.filterwarnings('ignore',category=BadCoefficients)

def ctrb(A,B):       
    """Controllabilty matrix

    Call:
    ctr=ctrb(A,B)

    Parameters
    ----------
    A, B : State and input matrix of the system

    Returns
    -------
    ctrb : matrix
    Controllability matrix

    """
    a=mat(A)
    b=mat(B)
    n=shape(a)[0]
    ctrb = b
    for i in arange(1,n):
        ctrb=hstack((ctrb,a**i*b))
    return ctrb

def acker(A,B,poles):
    """Pole placemenmt using Ackermann method

    Call:
    k=acker(A,B,poles)

    Parameters
    ----------
    A, B : State and input matrix of the system
    poles: desired poles

    Returns
    -------
    k: matrix
    State feedback gains

    """
    a=mat(A)
    b=mat(B)
    p=real(poly(poles))
    ct=ctrb(A,B)
    if det(ct)==0:
        k=0
        print "Pole placement invalid"
    else:
        n=size(p)
        pmat=p[n-1]*a**0
        for i in arange(1,n):
            pmat=pmat+p[n-i-1]*a**i
        k=inv(ct)*pmat
        k=k[-1][:]
    return k

def c2d(sys,Ts,method='zoh'):
    """Continous to discrete conversion with ZOH method

    Call:
    sysd=c2d(sys,Ts,method='zoh')

    Parameters
    ----------
    sys :   System in statespace or Tf form 
    Ts:     Sampling Time
    method: 'zoh', 'bi' or 'matched'

    Returns
    -------
    sysd: ss or Tf system
    Discrete system

    """
    flag = 0
    if type(sys).__name__=='TransferFunction':
        sys=tf2ss(sys)
        flag=1

    a=sys.A
    b=sys.B
    c=sys.C
    d=sys.D
    n=shape(a)[0]
    nb=shape(b)[1]
    nc=shape(c)[0]

    if method=='zoh':
        ztmp=zeros((nb,n+nb))
        tmp=hstack((a,b))
        tmp=vstack((tmp,ztmp))
        tmp=expm(tmp*Ts)
        A=tmp[0:n,0:n]
        B=tmp[0:n,n:n+nb]
        C=c
        D=d
    elif method=='bi':
        a=mat(a)
        b=mat(b)
        c=mat(c)
        d=mat(d)
        IT=mat(2/Ts*eye(n,n))
        A=(IT+a)*inv(IT-a)
        iab=inv(IT-a)*b
        tk=2/sqrt(Ts)
        B=tk*iab
        C=tk*(c*inv(IT-a))
        D=d+c*iab
    elif method=='matched':
        if nb!=1 and nc!=1:
            print "System is not SISO"
            return
        p=exp(sys.poles*Ts)
        z=exp(sys.zeros*Ts)
        infinite_zeros = len(sys.poles) - len(sys.zeros) - 1
        for i in range(0,infinite_zeros):
            z=hstack((z,-1))
        [A,B,C,D]=zpk2ss(z,p,1)
        sysd=ss(A,B,C,D,Ts)
        cg = dcgain(sys)
        dg = dcgain(sysd)
        [A,B,C,D]=zpk2ss(z,p,cg/dg)
    else:
        print "Method not supported"
        return
    
    #sysd=ss(A,B,C,D,Ts)
    sysd=ss(A,B,C,D)
    if flag==1:
        sysd=ss2tf(sysd)
    return sysd

def d2c(sys,method='zoh'):
    """Continous to discrete conversion with ZOH method

    Call:
    sysd=c2d(sys,method='log')

    Parameters
    ----------
    sys :   System in statespace or Tf form 
    method: 'zoh' or 'bi'

    Returns
    -------
    sysc: continous system ss or tf
    

    """
    flag = 0
    if type(sys).__name__=='TransferFunction':
        sys=tf2ss(sys)
        flag=1

    a=sys.A
    b=sys.B
    c=sys.C
    d=sys.D
    Ts=sys.Tsamp
    n=shape(a)[0]
    nb=shape(b)[1]
    nc=shape(c)[0]
    tol=1e-12
    
    if method=='zoh':
        if n==1:
            if b[0,0]==1:
                A=0
                B=b/sys.Tsamp
                C=c
                D=d
        else:
            tmp1=hstack((a,b))
            tmp2=hstack((zeros((nb,n)),eye(nb)))
            tmp=vstack((tmp1,tmp2))
            s=logm(tmp)
            s=s/Ts
            if norm(imag(s),inf) > sqrt(sp.finfo(float).eps):
                print "Warning: accuracy may be poor"
            s=real(s)
            A=s[0:n,0:n]
            B=s[0:n,n:n+nb]
            C=c
            D=d
    elif method=='bi':
        a=mat(a)
        b=mat(b)
        c=mat(c)
        d=mat(d)
        poles=eigvals(a)
        if any(abs(poles-1)<200*sp.finfo(float).eps):
            print "d2c: some poles very close to one. May get bad results."
        
        I=mat(eye(n,n))
        tk = 2 / sqrt (Ts)
        A = (2/Ts)*(a-I)*inv(a+I)
        iab = inv(I+a)*b
        B = tk*iab
        C = tk*(c*inv(I+a))
        D = d- (c*iab)
    else:
        print "Method not supported"
        return
    
    sysc=ss(A,B,C,D)
    if flag==1:
        sysc=ss2tf(sysc)
    return sysc

def care(A,B,Q,R):
    """Solve Riccati equation for discrete time systems

    Usage
    =====
    [K, S, E] = care(A, B, Q, R)

    Inputs
    ------
    A, B: 2-d arrays with dynamics and input matrices
    sys: linear I/O system 
    Q, R: 2-d array with state and input weight matrices

    Outputs
    -------
    X: solution of the Riccati eq.
    """

    # Check dimensions for consistency
    nstates = B.shape[0];
    ninputs = B.shape[1];
    if (A.shape[0] != nstates or A.shape[1] != nstates):
        raise ControlDimension("inconsistent system dimensions")

    elif (Q.shape[0] != nstates or Q.shape[1] != nstates or
          R.shape[0] != ninputs or R.shape[1] != ninputs) :
        raise ControlDimension("incorrect weighting matrix dimensions")

    X,rcond,w,S,T = \
        sb02od(nstates, ninputs, A, B, Q, R, 'C');

    return X


def dare(A,B,Q,R):
    """Solve Riccati equation for discrete time systems

    Usage
    =====
    [K, S, E] = care(A, B, Q, R)

    Inputs
    ------
    A, B: 2-d arrays with dynamics and input matrices
    sys: linear I/O system 
    Q, R: 2-d array with state and input weight matrices

    Outputs
    -------
    X: solution of the Riccati eq.
    """

    # Check dimensions for consistency
    nstates = B.shape[0];
    ninputs = B.shape[1];
    if (A.shape[0] != nstates or A.shape[1] != nstates):
        raise ControlDimension("inconsistent system dimensions")

    elif (Q.shape[0] != nstates or Q.shape[1] != nstates or
          R.shape[0] != ninputs or R.shape[1] != ninputs) :
        raise ControlDimension("incorrect weighting matrix dimensions")
    print nstates,ninputs,A,B,Q,R
    X,rcond,w,S,T = \
        sb02od(nstates, ninputs, A, B, Q, R, 'D');

    return X


def dlqr(*args, **keywords):
    """Linear quadratic regulator design for discrete systems

    Usage
    =====
    [K, S, E] = dlqr(A, B, Q, R, [N])
    [K, S, E] = dlqr(sys, Q, R, [N])

    The dlqr() function computes the optimal state feedback controller
    that minimizes the quadratic cost

        J = \sum_0^\infty x' Q x + u' R u + 2 x' N u

    Inputs
    ------
    A, B: 2-d arrays with dynamics and input matrices
    sys: linear I/O system 
    Q, R: 2-d array with state and input weight matrices
    N: optional 2-d array with cross weight matrix

    Outputs
    -------
    K: 2-d array with state feedback gains
    S: 2-d array with solution to Riccati equation
    E: 1-d array with eigenvalues of the closed loop system
    """

    # 
    # Process the arguments and figure out what inputs we received
    #
    
    # Get the system description
    if (len(args) < 3):
        raise ControlArgument("not enough input arguments")

    elif (ctrlutil.issys(args[0])):
        # We were passed a system as the first argument; extract A and B
        A = array(args[0].A, ndmin=2, dtype=float);
        B = array(args[0].B, ndmin=2, dtype=float);
        index = 1;
        if args[0].Tsamp==0.0:
            print "dlqr works only for discrete systems!"
            return
    else:
        # Arguments should be A and B matrices
        A = array(args[0], ndmin=2, dtype=float);
        B = array(args[1], ndmin=2, dtype=float);
        index = 2;

    # Get the weighting matrices (converting to matrices, if needed)
    Q = array(args[index], ndmin=2, dtype=float);
    R = array(args[index+1], ndmin=2, dtype=float);
    if (len(args) > index + 2): 
        N = array(args[index+2], ndmin=2, dtype=float);
        Nflag = 1;
    else:
        N = zeros((Q.shape[0], R.shape[1]));
        Nflag = 0;

    # Check dimensions for consistency
    nstates = B.shape[0];
    ninputs = B.shape[1];
    if (A.shape[0] != nstates or A.shape[1] != nstates):
        raise ControlDimension("inconsistent system dimensions")

    elif (Q.shape[0] != nstates or Q.shape[1] != nstates or
          R.shape[0] != ninputs or R.shape[1] != ninputs or
          N.shape[0] != nstates or N.shape[1] != ninputs):
        raise ControlDimension("incorrect weighting matrix dimensions")

    if Nflag==1:
        Ao=A-B*inv(R)*N.T
        Qo=Q-N*inv(R)*N.T
    else:
        Ao=A
        Qo=Q
    
    #Solve the riccati equation
    X = dare(Ao,B,Qo,R)

    # Now compute the return value
    Phi=mat(A)
    H=mat(B)
    K=inv(H.T*X*H+R)*(H.T*X*Phi+N.T)
    L=eig(Phi-H*K)
    return K,X,L

def minreal(sys):
    """Minimal representation for state space systems

    Usage
    =====
    [sysmin]=minreal[sys]

    Inputs
    ------

    sys: system in ss or tf form

    Outputs
    -------
    sysfin: system in state space form
    """
    a=mat(sys.A)
    b=mat(sys.B)
    c=mat(sys.C)
    d=mat(sys.D)
    nx=shape(a)[0]
    ni=shape(b)[1]
    no=shape(c)[0]
    if no<ni:
        c=vstack((c,zeros((ni-no,nx))))
        d=vstack((d,zeros((ni-no,ni))))
    if ni<no:
        b=hstack((b,zeros((nx,no-ni))))
        d=hstack((d,zeros((no,no-ni))))
    out=tb03ad(nx,ni,no,a,b,c,d,'R')

    nr=out[3]
    A=out[0][:nr,:nr]
    B=out[1][:nr,:ni]
    C=out[2][:no,:nr]
    sysf=ss(A,B,C,sys.D,sys.Tsamp)
    return sysf

def dcgain(sys):
    """Return the steady state value of the step response os sys

    Usage
    =====
    dcgain=dcgain(sys)

    Inputs
    ------

    sys: system

    Outputs
    -------
    dcgain : steady state value
    """

    a=mat(sys.A)
    b=mat(sys.B)
    c=mat(sys.C)
    d=mat(sys.D)
    nx=shape(a)[0]
    if sys.Tsamp!=0.0:
        a=a-eye(nx,nx)
    r=rank(a)
    if r<nx:
        gm=[]
    else:
        gm=-c*inv(a)*b+d
    return array(gm)

def dsimul(sys,u,x0=None):
    """Simulate the discrete system sys
    Only for discrete systems!!!

    Call:
    y=dsimul(sys,u)

    Parameters
    ----------
    sys : Discrete System in State Space form
    u   : input vector
    Returns
    -------
    y: ndarray
    Simulation results

    """
    a=mat(sys.A)
    b=mat(sys.B)
    c=mat(sys.C)
    d=mat(sys.D)
    nx=shape(a)[0]
    ns=shape(u)[1]
    if(x0 == None):
        xk=zeros((nx,1))
    else:
        x0 = np.matrix(np.array(x0).squeeze()).T
        if(not x0.shape == (nx,1)):
            print x0.shape
            raise AttributeError()
        xk=x0
    for i in arange(0,ns):
        uk=u[:,i]
        xk_1=a*xk+b*uk
        yk=c*xk+d*uk
        xk=xk_1
        if i==0:
            y=yk
        else:
            y=hstack((y,yk))
    y=array(y).T
    return y

def dstep(sys,Tf=10.0):
    """Plot the step response of the discrete system sys
    Only for discrete systems!!!

    Call:
    y=dstep(sys, [,Tf=final time]))

    Parameters
    ----------
    sys : Discrete System in State Space form
    Tf  : Final simulation time
 
    Returns
    -------
    Nothing

    """
    Ts=sys.Tsamp
    if Ts==0.0:
        "Only discrete systems allowed!"
        return

    ns=int(Tf/Ts+1)
    u=ones((1,ns))
    y=dsimul(sys,u)
    T=arange(0,Tf+Ts/2,Ts)
    plot(T,y)
    grid()
    show()

def dimpulse(sys,Tf=10.0):
    """Plot the impulse response of the discrete system sys
    Only for discrete systems!!!

    Call:
    y=dimpulse(sys,[,Tf=final time]))

    Parameters
    ----------
    sys : Discrete System in State Space form
    Tf  : Final simulation time
 
    Returns
    -------
    Nothing

    """
    Ts=sys.Tsamp
    if Ts==0.0:
        "Only discrete systems allowed!"
        return

    ns=int(Tf/Ts+1)
    u=zeros((1,ns))
    u[0,0]=1/Ts
    y=dsimul(sys,u)
    T=arange(0,Tf+Ts/2,Ts)
    plot(T,y)
    grid()
    show()

# Step response (plot)
def bb_step(sys,X0=None,Tf=None,Ts=0.001):
    """Plot the step response of the continous system sys

    Call:
    y=bb_step(sys [,Tf=final time] [,Ts=time step])

    Parameters
    ----------
    sys : Continous System in State Space form
    X0: Initial state vector (not used yet)
    Ts  : sympling time
    Tf  : Final simulation time
 
    Returns
    -------
    Nothing

    """
    if Tf==None:
        vals = eigvals(sys.A)
        r = min(abs(real(vals)))
        if r < 1e-10:
            r = 0.1
        Tf = 7.0 / r
    sysd=c2d(sys,Ts)
    dstep(sysd,Tf=Tf)

def full_obs(sys,poles):
    """Full order observer of the system sys

    Call:
    obs=full_obs(sys,poles)

    Parameters
    ----------
    sys : System in State Space form
    poles: desired observer poles

    Returns
    -------
    obs: ss
    Observer

    """
    if type(sys).__name__=='TransferFunction':
        "System must be in state space form"
        return
    a=mat(sys.A)
    b=mat(sys.B)
    c=mat(sys.C)
    d=mat(sys.D)
    poles=mat(poles)
    L=place(a.T,c.T,poles)
    L=mat(L).T
    Ao=a-L*c
    Bo=hstack((b-L*d,L))
    n=shape(Ao)
    m=shape(Bo)
    Co=eye(n[0],n[1])
    Do=zeros((n[0],m[1]))
    obs=ss(Ao,Bo,Co,Do,sys.Tsamp)
    return obs

def red_obs(sys,T,poles):
    """Reduced order observer of the system sys

    Call:
    obs=red_obs(sys,T,poles)

    Parameters
    ----------
    sys : System in State Space form
    T: Complement matrix
    poles: desired observer poles

    Returns
    -------
    obs: ss
    Reduced order Observer

    """
    if type(sys).__name__=='TransferFunction':
        "System must be in state space form"
        return
    a=mat(sys.A)
    b=mat(sys.B)
    c=mat(sys.C)
    d=mat(sys.D)
    T=mat(T)
    P=mat(vstack((c,T)))
    poles=mat(poles)
    invP=inv(P)
    AA=P*a*invP
    ny=shape(c)[0]
    nx=shape(a)[0]
    nu=shape(b)[1]

    A11=AA[0:ny,0:ny]
    A12=AA[0:ny,ny:nx]
    A21=AA[ny:nx,0:ny]
    A22=AA[ny:nx,ny:nx]

    L1=place(A22.T,A12.T,poles)
    L1=mat(L1).T

    nn=nx-ny

    tmp1=mat(hstack((-L1,eye(nn,nn))))
    tmp2=mat(vstack((zeros((ny,nn)),eye(nn,nn))))
    Ar=tmp1*P*a*invP*tmp2
 
    tmp3=vstack((eye(ny,ny),L1))
    tmp3=mat(hstack((P*b,P*a*invP*tmp3)))
    tmp4=hstack((eye(nu,nu),zeros((nu,ny))))
    tmp5=hstack((-d,eye(ny,ny)))
    tmp4=mat(vstack((tmp4,tmp5)))

    Br=tmp1*tmp3*tmp4

    Cr=invP*tmp2

    tmp5=hstack((zeros((ny,nu)),eye(ny,ny)))
    tmp6=hstack((zeros((nn,nu)),L1))
    tmp5=mat(vstack((tmp5,tmp6)))
    Dr=invP*tmp5*tmp4
    
    obs=ss(Ar,Br,Cr,Dr,sys.Tsamp)
    return obs

def comp_form(sys,obs,K):
    """Compact form Conroller+Observer

    Call:
    contr=comp_form(sys,obs,K)

    Parameters
    ----------
    sys : System in State Space form
    obs : Observer in State Space form
    K: State feedback gains

    Returns
    -------
    contr: ss
    Controller

    """
    nx=shape(sys.A)[0]
    ny=shape(sys.C)[0]
    nu=shape(sys.B)[1]
    no=shape(obs.A)[0]

    Bu=mat(obs.B[:,0:nu])
    By=mat(obs.B[:,nu:])
    Du=mat(obs.D[:,0:nu])
    Dy=mat(obs.D[:,nu:])

    X=inv(eye(nu,nu)+K*Du)

    Ac = mat(obs.A)-Bu*X*K*mat(obs.C);
    Bc = hstack((Bu*X,By-Bu*X*K*Dy))
    Cc = -X*K*mat(obs.C);
    Dc = hstack((X,-X*K*Dy))
    contr = ss(Ac,Bc,Cc,Dc,sys.Tsamp)
    return contr

def comp_form_i(sys,obs,K,Ts,Cy=[[1]]):
    """Compact form Conroller+Observer+Integral part
    Only for discrete systems!!!

    Call:
    contr=comp_form_i(sys,obs,K,Ts[,Cy])

    Parameters
    ----------
    sys : System in State Space form
    obs : Observer in State Space form
    K: State feedback gains
    Ts: Sampling time
    Cy: feedback matric to choose the output for integral part

    Returns
    -------
    contr: ss
    Controller

    """
    if sys.Tsamp==0.0:
        print "contr_form_i works only with discrete systems!"
        return

    ny=shape(sys.C)[0]
    nu=shape(sys.B)[1]
    nx=shape(sys.A)[0]
    no=shape(obs.A)[0]
    ni=shape(Cy)[0]

    B_obsu = mat(obs.B[:,0:nu])
    B_obsy = mat(obs.B[:,nu:nu+ny])
    D_obsu = mat(obs.D[:,0:nu])
    D_obsy = mat(obs.D[:,nu:nu+ny])

    k=mat(K)
    nk=shape(k)[1]
    Ke=k[:,nk-ni:]
    K=k[:,0:nk-ni]
    X = inv(eye(nu,nu)+K*D_obsu);

    a=mat(obs.A)
    c=mat(obs.C)
    Cy=mat(Cy)

    tmp1=hstack((a-B_obsu*X*K*c,-B_obsu*X*Ke))

    tmp2=hstack((zeros((ni,no)),eye(ni,ni)))
    A_ctr=vstack((tmp1,tmp2))

    tmp1=hstack((zeros((no,ni)),-B_obsu*X*K*D_obsy+B_obsy))
    tmp2=hstack((eye(ni,ni)*Ts,-Cy*Ts))
    B_ctr=vstack((tmp1,tmp2))

    C_ctr=hstack((-X*K*c,-X*Ke))
    D_ctr=hstack((zeros((nu,ni)),-X*K*D_obsy))

    contr=ss(A_ctr,B_ctr,C_ctr,D_ctr,sys.Tsamp)
    return contr
    
def sysctr(sys,contr):
    """Build the discrete system controller+plant+output feedback

    Call:
    syscontr=sysctr(sys,contr)

    Parameters
    ----------
    sys : Continous System in State Space form
    contr: Controller (with observer if required)
 
    Returns
    -------
    sysc: ss system
    The system with reference as input and outputs of plants 
    as output

    """
    if contr.Tsamp!=sys.Tsamp:
        print "Systems with different sampling time!!!"
        return
    sysf=series(contr,sys)

    nu=shape(sysf.B)[1]
    b1=mat(sysf.B[:,0])
    b2=mat(sysf.B[:,1:nu])
    d1=mat(sysf.D[:,0])
    d2=mat(sysf.D[:,1:nu])

    n2=shape(d2)[0]

    Id=mat(eye(n2,n2))
    X=inv(Id-d2)

    Af=mat(sysf.A)+b2*X*mat(sysf.C)
    Bf=b1+b2*X*d1
    Cf=X*mat(sysf.C)
    Df=X*d1

    sysc=ss(Af,Bf,Cf,Df,sys.Tsamp)
    return sysc

def set_aw(sys,poles):
    """Divide in controller in input and feedback part
       for anti-windup

    Usage
    =====
    [sys_in,sys_fbk]=set_aw(sys,poles)

    Inputs
    ------

    sys: controller
    poles : poles for the anti-windup filter

    Outputs
    -------
    sys_in, sys_fbk: controller in input and feedback part
    """
    den_old=poly(eigvals(sys.A))
    den = poly(poles)
    tmp= tf(den_old,den,sys.Tsamp)
    tmpss=tf2ss(tmp)
    sys_in=minreal(series(sys,tmp))
    sys_in.Tsamp=sys.Tsamp
    sys_fbk=tf2ss(1-tmp)
    sys_fbk.Tsamp=sys.Tsamp
    return sys_in, sys_fbk

