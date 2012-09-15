import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import networkx as nx

import itertools

from lqr_tools import LQR_QP, dtfh_lqr, simulate_lti_fb_dt, AQR, final_value_LQR, lqr_dim

class Diff_RRT():
    def __init__(self,linA,linB,dyn_f,cost_0,cost_1,cost_2,n,m,max_time_horizon):
        (self.n,self.m) = (n,m)   #n+1 for affine term
        self.max_time_horizon = max_time_horizon
        
        self.linA = linA
        self.linB = linB
        self.dyn_f = dyn_f

        self.cost_zeroth_ord = cost_0
        self.cost_first_ord = cost_1
        self.cost_second_ord = cost_2
        
        def action_state_valid(x,u):
            return True

        self.action_state_valid = action_state_valid
        self.max_nodes_per_extension = None

        self.max_steer_cost = np.inf

        self.u0 = np.zeros(self.m)

    def get_ABc(self,x,u):
        """
            x[k+1] = A(x[k] - x0) + B(u[k] - u0) +   f(x0,u)
            x[k+1] = A*x[k] + B*u[k] +   -A*x0 -B*u0 + f(x0,u)
        """
        A = self.linA(x,u)
        B = self.linB(x,u)
        c = self.dyn_f(x,u).reshape(self.n,1) - np.dot(A,x.reshape(self.n,1)) - np.dot(B,u.reshape(self.m,1))

        A = np.matrix(A)
        B = np.matrix(B)
        c = np.matrix(c)

        assert A.shape == (self.n,self.n)
        assert B.shape == (self.n,self.m)
        assert c.shape == (self.n,1)
        return A,B,c
    
    def get_QRqrd(self,x,u):
        n,m = self.n, self.m
        d = self.cost_zeroth_ord(x,u)
        _qr = self.cost_first_ord(x,u)

        q = _qr[0,0:n].reshape(n,1)
        r = _qr[0,n:n+m].reshape(m,1)
        _QR = self.cost_second_ord(x,u)
        #ignoring cross terms between x and u #FIXME
        Q = _QR[0:n,0:n]
        R = _QR[n:n+m,n:n+m]
        
        assert Q.shape == (n,n)
        assert R.shape == (m,m)
                    
        return Q,R,q,r,d
        
    def run_forward(self,x0,us):

        n = self.n+1  #+1 for time dimension 
        m = self.m
        #us = np.reshape(us,newshape=(-1,m))
        assert len(x0) == n
        T = us.shape[0]
        xs = np.zeros(shape=(T+1,n))
        xs[0] = np.squeeze(x0)

        for i in range(1,T+1):
            xs[i,0:self.n] = self.dyn_f(xs[i-1,0:self.n].T,us[i-1].T).reshape((1,self.n))
            xs[i,self.n] = xs[i-1,self.n] + 1               #increase time
        return xs[1:]

    def run_forward_fb(self,x0,gain_schedule):
        raise NotImplementedError 
        A = self.A
        B = self.B
        n = self.n+1  #+1 for time dimension 
        m = self.m
        assert len(x0) == n
        T = gain_schedule.shape[0]
        assert gain_schedule.shape[2] == n-1 #no time in the gain 
        assert gain_schedule.shape[1] == m

        xs = np.zeros(shape=(T+1,n))
        us = np.zeros(shape=(T,m))

        xs[0] = np.squeeze(x0)

        for i in range(1,T+1):
            us[i-1] = -1* np.dot(gain_schedule[i-1],xs[i-1,0:self.n])
            xs[i,0:self.n] = np.dot(A,xs[i-1,0:self.n].T) + np.dot(B,us[i-1].T)
            xs[i,self.n] = xs[i-1,self.n] + 1
        return xs[1:],us


    def collision_free(self,from_node,action):
        """
        check that taking action from from_node produces a collision free trajectory
        if not, return a partial trajectory for the state (x_path) and control (u_path)
        u_path is a list of actions -- it partitions the actions.
        """

        x0 = from_node['state']
        x_path = []
        u_path = []
        all_the_way = True
        if len(action) > 0:
            if self.action_state_valid(x0,action[0]):
                x_path_np = self.run_forward(x0,action)

                for i in range(len(x_path_np)):                   
                    if not self.action_state_valid(x_path_np[i],action[i]):
                        all_the_way = False
                        break
                    x_path.append(x_path_np[i])
                    u_path.append(action[[i]])
            else:
                all_the_way = False

        u_path_all = np.zeros(shape=(len(u_path),self.m))

        #don't return any intermediate points
        if False:
            for i in range(len(u_path)):
                u_path_all[i] = u_path[i][0]
            if len(x_path)>0:
                return [x_path[-1]], [u_path_all], all_the_way
            else:
                return [], [], all_the_way

        #downsample
        if self.max_nodes_per_extension is not None and len(x_path) > self.max_nodes_per_extension:
            x_path_ds = []
            u_path_ds = []
            #getting indices correct here was a pain.
            inds = np.array(np.round(np.linspace(0,len(x_path),self.max_nodes_per_extension)),dtype=np.int)
            for (i,j) in zip(inds[0:-1],inds[1:]):
                x_path_ds.append(x_path[j-1])
                l = j-i
                action_part = np.zeros(shape=(l,self.m))
                for k in range(l):
                    action_part[k] = u_path[i+k][0]
                u_path_ds.append(action_part)
            
            assert len(u_path) == sum( [len(action_part) for action_part in u_path] )
            x_path = x_path_ds
            u_path = u_path_ds
        #u_path = np.array(u_path)
        
        return x_path, u_path, all_the_way

    def same_state(self,a,b):
        return a[self.n] == b[self.n] and np.allclose(a,b,atol=1e-4) #time has to be identical and phase-space state has to be approximate

    def cost(self,x_from,action):
        #this does not include the cost of being in at the state arrived by taking the last action
		#that is, with x_(k+1) = f(x_k,u_k) this sums cost(x_k,u_k) for k in range(len(action))
        assert len(x_from) == self.n+1
        if len(action) == 0:
            #null action
            return 0    #is captured in dynamics below, but if the action is actually null, then the next assertion fails
        assert action.shape[1] == self.m

        x_path = self.run_forward(x_from,action)
        cost = 0

        for i in range(action.shape[0]):
            #x_path does not include x_from
            x = x_path[[i-1],0:self.n].T if i>0 else x_from[0:self.n] #don't include time
            u = action[[i],:].T
            cost += np.squeeze( self.cost_zeroth_ord(x,u) ) 
        return cost

    def node_cache_ctg(self,node):
        A,B,c = self.get_ABc(node['state'][0:self.n],self.u0)
        Q,R,q,r,d = self.get_QRqrd(node['state'][0:self.n],self.u0)

        max_time_horizon = self.max_time_horizon

        #print 'calculate ctg for', node
        x = node['state']        
        #reverse system
        Ar = A.I
        Br = -A.I * B
        cr = -A.I * c

        kmax = max_time_horizon - x[self.n] +1
        assert kmax > 0

        #ctg[0] is cost-to-go zero steps -- very sharp quadratic
        #so ctg[k] with k = max_time_horizon - from_node['state'][self.n] is time to go


        Fs, Ps = final_value_LQR(Ar,Br,Q,R,x[0:self.n],kmax,c=cr,q=q,r=r,d=d)
        #storing in reverse order is easier to think about.
        #node['gain'][i] is what you should do with i steps left to go.
        node['ctg'] = Ps[::-1] 
        node['gain'] = Fs[::-1]

        node['dynamics'] = (A,B,c)
        node['reverse_dynamics'] = (Ar,Br,cr)

    def steer(self,x_from_node,x_toward,cost_limit=True):
        A,B,c = self.get_ABc(x_from_node['state'][0:self.n],self.u0)
        Q,R,q,r,d = self.get_QRqrd(x_from_node['state'][0:self.n],self.u0)

        x_from = x_from_node['state']
        assert len(x_from) == self.n+1
        T = x_toward[self.n] - x_from[self.n] #how much time to do the steering

        assert T-int(T) == 0 #discrete time

        T=int(T)
        
        if T<=0:
            return (x_from,np.zeros(shape=(0,self.m)))   #stay here

        desired = np.matrix(x_toward[0:self.n].reshape(self.n,1))

        Qf = np.eye(self.n) * 1e8
        qf = -np.dot(Qf,desired)

        Qhf = np.zeros(shape=(self.n+1,self.n+1))
        Qhf[0:self.n,0:self.n] = Qf
        Qhf[0:self.n,[self.n]] = qf
        Qhf[[self.n],0:self.n] = qf.T
        Qhf[self.n,self.n] = np.dot(desired.T,np.dot(Qf,desired))

        (Ah,Bh,Qh,Rh,pk) = AQR(     A=A,
                                    B=B,
                                    c=c,
                                    Q=Q,
                                    R=R,
                                    q=q,
                                    r=r,
                                    d=d,
                                    ctdt='dt')

        Fs, Ps = dtfh_lqr(A=Ah,B=Bh,Q=Qh,R=R,N=T,Q_terminal=Qhf)
        #print Fs
        xs = np.zeros(shape=(T+1,self.n+1))
        us = np.zeros(shape=(T,self.m))
        xs[0] = x_from

        cost = 0
        for i in range(T):
            us[i] = -1 * (np.dot(Fs[i,:,0:self.n],xs[i,0:self.n]) + Fs[i,:,self.n]) + pk #FIXME
            #xs[i+1,0:self.n] = np.dot(A,xs[i,0:self.n].T) + np.dot(B,us[i].T) + c	#pretend the system is linear
            #xs[i+1,self.n] = xs[i,self.n] + 1

            xs[i+1] = self.run_forward(xs[i], us[[i]] ) 

            cost += self.cost(xs[i].T,us[i].reshape(1,self.m))
            if cost_limit and cost > self.max_steer_cost:
                break

        if i < T-1:
            us = us[:i]
            x_final = xs[i+1]
        else:
            x_final = xs[T]

        return (x_final, us)

    def steer_cache(self,x_from_node,x_toward,cost_limit=True):
        A,B,c = self.get_ABc(x_from_node['state'][0:self.n],self.u0)


        x_from = x_from_node['state']
        assert len(x_from) == self.n+1
        T = x_toward[self.n] - x_from[self.n] #how much time to do the steering

        assert T-int(T) == 0 #discrete time

        T=int(T)
        
        if T <= 0:
            return (x_from,np.zeros(shape=(0,self.m)))   #stay here

        if T < 10:
            #this technique isn't too accurate for short times due to slack in the final-value constraint
            #so do something else.
            return self.steer(x_from_node,x_toward,cost_limit)

        desired = np.matrix(x_toward[0:self.n].reshape(self.n,1))            

        if 'gain' not in x_from_node:
            self.node_cache_ctg(x_from_node)

        Fs = x_from_node['gain']

        if T >len(Fs):
            print "requested uncached steer!!!"
            return self.steer(x_from_node,x_toward) #fixme should cache more

        #reverse system
        Ar = A.I
        Br = -A.I * B
        cr = A.I * c

        xs_r = np.zeros(shape=(T+1,self.n+1))   #r for reverse
        us = np.zeros(shape=(T,self.m))

        #we're driving backwards. start at x_toward.
        xs_r[0] = x_toward

        for i in range(T):
            j = T - i-1 #gain matrices Fs[j] is what you should do with j steps remaining
            us[i] = -1 * (np.dot(Fs[j,:,0:self.n],xs_r[i,0:self.n]) + Fs[j,:,self.n])
            xs_r[i+1,0:self.n] = np.dot(Ar,xs_r[i,0:self.n].T) + np.dot(Br,us[i].T) + np.array(cr).T[0] #sloppy dimensions
            xs_r[i+1,self.n] = xs_r[i,self.n] - 1 #reverse time
    
        us_r = us
        us = us_r[::-1]
        xs = np.zeros(shape=(T+1,self.n+1))
        cost = 0
        for i in range(T):
            xs[i+1] = self.run_forward(xs[i], us[[i]] ) 

            assert xs[i+1,self.n] == xs[i,self.n] + 1
            cost += self.cost(xs[i].T,us[i].reshape(1,self.m))
            if cost_limit and cost > self.max_steer_cost:
                break

        return (xs[i+1], us[:i])


    def steer_QP(self,x_from_node,x_toward):
        raise NotImplemented
        A,B,c = self.get_ABc(x_from_node['state'][0:self.n],self.u0)
        Q,R,q,r,d = self.get_QRqrd(node['state'][0:self.n],self.u0)


        x_from = x_from_node['state']
        assert len(x_from) == self.n+1
        T = x_toward[self.n] - x_from[self.n] #how much time to do the steering

        assert T-int(T) == 0 #discrete time

        T=int(T)
        
        if T<=0:
            return (x_from,np.zeros(shape=(0,self.m)))   #stay here

        try:
            qpsol, qpmats ,xs,us = LQR_QP(A,B,Q,R,(T+1),
                                x0=x_from[0:self.n],
                                xT=x_toward[0:self.n])
        except ValueError as e:
            #quadratic program is probably infeasible. This can happen if the time horizon is too short and the system's reachability doesn't include the final-value constraint
            return (x_from,np.zeros(shape=(0,self.m)))   #stay here (could do something smarter)
            
        (QP_P,QP_q,QP_A,QP_B) = qpmats

        xs = xs.T
        us = us.T
        x_actual = np.concatenate((     xs[-1],
                                        [x_from[self.n]+us.shape[0]]
                                ))
        return (x_actual, us)

    def distance_direct(self,from_node,to_point):
        #print from_node['state'], to_point
        #to_point is an array and from_point is a node
        assert len(to_point)==self.n+1
        x_actual,action = self.steer(from_node,to_point)
        if self.same_state(x_actual,to_point): #if actually drove there:
            return self.cost(from_node['state'],action)
        else:
            return np.inf

    def distance_direct_qp(self,from_node,to_point):
        #print from_node['state'], to_point
        #to_point is an array and from_point is a node
        assert len(to_point)==self.n
        x_actual,action = self.steer_QP(from_node,to_point)
        if self.same_state(x_actual,to_point): #if actually drove there:
            return self.cost(from_node['state'],action)
        else:
            return np.inf

    def distance(self,from_node,to_point):
        #to_point is an array and from_point is a node

        A,B,c = self.get_ABc(from_node['state'][0:self.n],self.u0)
        Q,R,q,r,d = self.get_QRqrd(from_node['state'][0:self.n],self.u0)


        x_from = from_node['state']
        x_toward = to_point
        assert len(x_toward)==self.n+1

        T = x_toward[self.n] - x_from[self.n] #how much time to do the steering

        assert T-int(T) == 0 #discrete time

        T=int(T)
        
        if T<0:
            return np.inf
        elif T==0:
            return 0 if self.same_state(x_from,x_toward) else np.inf

        desired = np.matrix(x_toward[0:self.n].reshape(self.n,1))            

        #we want the final bowl to be centered at desired:
        #(x-x_d)^T * Qf * (x-x_d)
        #xT*Qf*x -x_dT * Qf * x - xT *Qf *x_d * x_dT * Qf * x_d
        Qf = np.eye(self.n) * 1e8
        qf = -np.dot(Qf,desired)

        Qhf = np.zeros(shape=(self.n+1,self.n+1))
        Qhf[0:self.n,0:self.n] = Qf
        Qhf[0:self.n,[self.n]] = qf
        Qhf[[self.n],0:self.n] = qf.T
        Qhf[self.n,self.n] = np.dot(desired.T,np.dot(Qf,desired))

        (Ah,Bh,Qh,Rh,pk) = AQR(     A=A,
                                    B=B,
                                    Q=Q,
                                    R=R,
                                    c=c,
                                    q=q,
                                    r=r,
                                    d=d,
                                    ctdt='dt')

        T = T+1
        Fs, Ps = dtfh_lqr(A=Ah,B=Bh,Q=Qh,R=R,N=T,Q_terminal=Qhf)

        x_from_homo = np.zeros(self.n+1)
        x_from_homo[0:self.n] = x_from[0:self.n]
        x_from_homo[self.n] = 1
        #assert False
        return np.dot(x_from_homo.T,np.dot(Ps[0],x_from_homo))


    
    def distance_cache(self,from_node,to_point):
        #to_point is an array and from_point is a node

        max_time_horizon = self.max_time_horizon

        x_from = from_node['state']
        x_toward = to_point
        assert len(x_toward)==self.n+1

        T = x_toward[self.n] - x_from[self.n] #how much time to do the steering

        assert T-int(T) == 0 #discrete time

        T=int(T)
        
        if T<0:
            return np.inf
        elif T==0:
            return 0 if self.same_state(x_from,x_toward) else np.inf
        
        if T < 5:
            #this technique isn't too accurate for short times due to slack in the final-value constraint
            #so do something else.
            return self.distance(from_node,to_point)
        
        if 'ctg' not in from_node:
            self.node_cache_ctg(from_node)

        if T >= len(from_node['ctg']):
            print 'requested uncached distance!!!'
            distance(from_node,to_point)

        ctg = from_node['ctg'][T]

        x_to_homo = np.zeros(self.n+1)
        x_to_homo[0:self.n] = x_toward[0:self.n]
        x_to_homo[self.n] = 1
         
        return np.dot(x_to_homo,np.dot(ctg,x_to_homo.T))

    def check_cache_distances(self,rrt,to_point):
        for node in rrt.tree.nodes():
            from_node = rrt.tree.node[node]
            d1 = self.distance(from_node,to_point)
            d2 = self.distance_cache(from_node,to_point)
            d3 = self.distance_direct(from_node,to_point)
            d4 = self.distance_direct_qp(from_node,to_point)
            d5 = self.distance_direct_steer_cache(from_node,to_point)
            T = to_point[self.n] - from_node['state'][self.n] 
            print d1,d2,d3,d4,d5,T
    #        if not (a == np.inf and b == np.inf):
    #            print a,b,abs(a-b)/(abs(a)+abs(b))

