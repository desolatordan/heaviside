import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab
import matplotlib.colors
import yaml,time,cProfile
from scipy import interpolate


'''
Maxwell's Equations (Differential Form)

divergence(D(t)) = \rho_v(t)
divergence(B(T)) = 0
curl(E(t)) = -dB(t)/dt
curl(H(t)) = J(t)+dD(t)/dt

Constitutive Relations

D(t) = [\epsilon(t)]*E(t)
B(t) = [\mu(t)]*H(t)

Parameters
* convolution
[] tensor (can alter direction of vector, not just magnitude)

Assumptions: Maxwell's Equation

Assume rigid medium, no movement: Ignore Lorentz Force Law.
Assume no divergence of E-field: \rho_v(t) = 0
Assume no loss or induced current: J(t) = 0
divergence(D(t)) = 0
divergence(B(T)) = 0
curl(E(t)) = -dB(t)/dt
curl(H(t)) = dD(t)/dt

Assumptions: Constitutive Relations

Assume isotropic materials: Collapse tensor of \mu and \epsilon to a scalar.
Assume non-dispersive materials: \mu and \epsilon is non-time-varying
Assume linear materials: \mu and \epsilon is only applied to first-order E-field and H-field
D(t) = \epsilon E(t)
B(t) = \mu H(t)

Combine:

divergence(\epsilon E(t)) = 0
divergence(\mu H(t)) = 0
curl(E(t)) = - \mu dH(t)/dt => expand the curl equations!
curl(H(t)) = \epsilon dE(t)/dt => expand the curl equations!

Expand the curl equations:

dEz/dy - dEy/dz = \mu_xx * dHx/dt + \mu_xy * dHy/dt + \mu_xz * dHz/dt
dEx/dz - dEz/dx = \mu_yx * dHx/dt + \mu_yy * dHy/dt + \mu_yz * dHz/dt
dEy/dx - dEx/dy = \mu_zx * dHx/dt + \mu_zy * dHy/dt + \mu_zz * dHz/dt

dHz/dy - dHy/dz = \epsilon_xx * dEx/dt + \epsilon_xy * dEy/dt + \epsilon_xz * dEz/dt
dHx/dz - dHz/dx = \epsilon_yx * dEx/dt + \epsilon_yy * dEy/dt + \epsilon_yz * dEz/dt
dHy/dx - dHx/dy = \epsilon_zx * dEx/dt + \epsilon_zy * dEy/dt + \epsilon_zz * dEz/dt

Assume isotropic instead:

dEz/dy - dEy/dz = - \mu * dHx/dt
dEx/dz - dEz/dx = - \mu * dHy/dt
dEy/dx - dEx/dy = - \mu * dHz/dt

dHz/dy - dHy/dz = \epsilon * dEx/dt
dHx/dz - dHz/dx = \epsilon * dEy/dt
dHy/dx - dHx/dy = \epsilon * dEz/dt

Yee-grid
Offset E and H in space.
 - Divergence free by design (removes divergence equations)
 - Satisfies boundary conditions automatically.
 - Easy approximation of curl()

Finite-Difference Approximation
dF(1.5)/dt = (F(2)-F(1))/dt
To be stable, cannot mix with terms from other points in space/time other than F(1.5).  Must interpolate if necessary,
  which could get messy.  To avoid this, we define E and H at convenient points in space, only where needed.

Convert derivative to finite-difference:

dEz/dy - dEy/dz = - \mu * dHx/dt
(Ez(i,j+1,k)-Ez(i,j,k))/dy - (Ey(i,j,k+1)-Ey(i,j,k))/dz = -\mu * (Hx(i,j,k)(t+.5)-Hx(i,j,k)(t-.5))/dt

Find update equation: Solve for next timestep.

Hx(i,j,k)(t+.5) = Hx(i,j,k)(t-.5) - dt*(Ez(i,j+1,k)-Ez(i,j,k))/dy/\mu + dt*(Ey(i,j,k+1)-Ey(i,j,k))/dz/\mu
1D: Hx(i,j,k)(t+.5) = Hx(i,j,k)(t-.5) + dt/\mu*(Ey(i,j,k+1)-Ey(i,j,k))/dz

Normalize:
E and H are a few magnitudes different.
c = 1/sqrt(\epsilon_0*\mu_0)

Let's come up with H~, where H = H~ * sqrt(\mu_0/\epsilon_0)*H

Hx(i,j,k)(t+.5) = Hx(i,j,k)(t-.5) - c*dt*(Ez(i,j+1,k)-Ez(i,j,k))/dy/\mu_r + c*dt*(Ey(i,j,k+1)-Ey(i,j,k))/dz/\mu_r

1D: Hx(i,j,k)(t+.5) = Hx(i,j,k)(t-.5) + c*dt/\mu_r*(Ey(i,j,k+1)-Ey(i,j,k))/dz

self.Hx[z] = self.Hx[z] + self.mH[z] * (self.Ey[z + 1] - self.Ey[z]) / self.dz

 => H(t+dt) = H(t-dt)-(dt/\mu)*curl(E(t+dt))
 => E(t+dt) = E(t-dt)+(dt/\epsilon)*curl(H(t+dt))

divergence(\epsilon E(t)) = 0
divergence(\mu H(t)) = 0
curl(E(t)) = - \mu dH(t)/dt => H(t+dt) = H(t-dt)-(dt/\mu)*curl(E(t))
curl(H(t)) = \epsilon dE(t)/dt => E(t+dt) = E(t)+(dt/\epsilon)*curl(H(t+dt))

Expand curl equations into partial differential equations (turn curl into spatial derivatives).

Due to difference in magnitude of E and H fields, normalize H.
'''


class Processor:
    def preprocess(self, sim):
        pass
    def process(self, sim):
        pass
    def postprocess(self, sim):
        pass


#TODO: Plot3D class?
class Plot3D(Processor):
    def preprocess(self,sim):
        plt.ion()

        self.O = np.zeros(sim.n)
        for key, source in sim.sources.items():
            self.O[source.cfg.pos] = 1.0

        cmap = plt.get_cmap('bwr')
        self.my_cmap = cmap(np.arange(cmap.N))
        self.my_cmap[:, -1] = np.append(np.linspace(0, 1, cmap.N/2),np.linspace(1, 0, cmap.N/2))
        self.my_cmap = matplotlib.colors.ListedColormap(self.my_cmap)

        self.E = np.zeros(sim.n)
        self.E = np.linalg.norm(sim.E,axis=3)

        self.fig1 = plt.figure()
        self.ax1 = self.fig1.add_subplot(1,1,1,projection='3d')
        self.Odata = self.ax1.voxels(self.O,facecolors='#00FF0040')
        self.Edata = self.ax1.voxels(np.ones(sim.n), facecolors='k')

        self.ax1.set(xlabel='x', ylabel='y', zlabel='z')

        plt.tight_layout()

    def process(self,sim,drawNow=False):
        if not sim.step % 10 or drawNow:
            print(sim.step)

            self.E = np.linalg.norm(sim.E, axis=3)
            for key,face in self.Edata.items():
                face.set_facecolor(self.my_cmap(self.E[key]))
                #face.set_alpha(self.E[key])

            plt.figure(self.fig1.number)
            plt.sca(self.ax1)

            plt.gca().relim()
            plt.draw()
            plt.pause(.001)

    def postprocess(self,sim):
        plt.ioff()
        plt.show()


class Plot2D(Processor):
    #TODO: Only handles x/y plane currently
    def preprocess(self,sim):
        plt.ion()

        self.O = np.zeros(sim.n[:2]+[4])
        for key, source in sim.sources.items():
            if np.shape(source.cfg.pos) == (3,):
                self.O[source.cfg.pos[0:1]] = [0.0, 1.0, 0.0, 1.0]
            else:
                self.O[source.cfg.pos[:,:,0,2]] = [0.0,1.0,0.0,1.0]
        self.O = np.swapaxes(self.O,0,1)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.Edata = plt.imshow(np.swapaxes(sim.E[:, :, 0, 2],0,1), cmap='bwr', origin='lower')
        self.Odata = plt.imshow(self.O, origin='lower')
        self.annoText = plt.text(0.1, 0.1, '', horizontalalignment='center', verticalalignment='center',
                            transform=self.ax.transAxes, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5})
        plt.tight_layout()

    def process(self,sim,drawNow=False):
        if not sim.step % 10 or drawNow:
            plt.figure(self.fig.number)
            plt.sca(self.ax)
            self.Edata.set_data(np.swapaxes(sim.E[:, :, 0, 2],0,1))
            maxdata = max(abs(np.max(sim.E[:, :, 0, 2])),abs(np.min(sim.E[:, :, 0, 2])))
            self.Edata.set_clim(vmax=maxdata, vmin=-maxdata)
            self.annoText.set_text(sim.step)
            plt.gca().relim()
            plt.draw()
            #plt.savefig('{}/{}.png'.format('gif',sim.step))

    def postprocess(self,sim):
        plt.ioff()
        plt.show()

class Plot1D(Processor):
    #TODO: Only handles z axis currently
    def __init__(self,plot_eh=True,plot_f=False):
        self.plot_eh = plot_eh
        self.plot_f = plot_f
    def preprocess(self,sim):
        plt.ion()
        self.lasttime = time.time()

        # Plot er with E/H fields on top
        if self.plot_eh:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            plt.imshow([sim.er[0,0,:]], cmap='binary', extent=[0, sim.d * sim.n[2], -2, 2], aspect='auto',
                       vmax=1.5 * np.max(sim.er))

            # Plot sources
            for k, source in sim.sources.items():
                if source.cfg.efield:
                    if source.cfg.type == 'source':
                        plt.plot([range(sim.n[2])[source.cfg.pos[2]] * sim.d, range(sim.n[2])[source.cfg.pos[2]] * sim.d], [-2, 2],
                                 'c:')
                    elif source.cfg.type == 'sink':
                        plt.plot([range(sim.n[2])[source.cfg.pos[2]] * sim.d, range(sim.n[2])[source.cfg.pos[2]] * sim.d], [-2, 2],
                                 'm:')
                    else:
                        plt.plot([range(sim.n[2])[source.cfg.pos[2]] * sim.d, range(sim.n[2])[source.cfg.pos[2]] * sim.d], [-1, 1],
                                 'r--')
                else:
                    if source.cfg.type == 'source':
                        plt.plot(
                            [(range(sim.n[2])[source.cfg.pos[2]] + 1) * sim.d, (range(sim.n[2])[source.cfg.pos[2]] + 1) * sim.d],
                            [-2, 2], 'm:')
                    elif source.cfg.type == 'sink':
                        plt.plot(
                            [(range(sim.n[2])[source.cfg.pos[2]] + 1) * sim.d, (range(sim.n[2])[source.cfg.pos[2]] + 1) * sim.d],
                            [-2, 2], 'm:')
                    else:
                        plt.plot([range(sim.n[2])[source.cfg.pos[2]] * sim.d, range(sim.n[2])[source.cfg.pos[2]] * sim.d], [-1, 1],
                                 'r--')

            # E,H fields
            self.lineH, = plt.plot(np.linspace(0, sim.n[2] * sim.d, sim.n[2]), sim.H[0,0,:,0], 'm')
            self.lineE, = plt.plot(np.linspace(0, sim.n[2] * sim.d, sim.n[2]), sim.E[0,0,:,1], 'b')

            self.annoText = plt.text(0.1, 0.1, '', horizontalalignment='center', verticalalignment='center',
                                transform=self.ax.transAxes, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5})
            plt.xlim([-sim.d / 10, sim.n[2] * sim.d])
            plt.tight_layout()

        # Plot FFT
        if self.plot_f:
            self.fig2 = plt.figure()
            plt.subplot(3, 1, 1)
            plt.title('FFT')
            plt.grid(True)
            self.lineR, = plt.plot(sim.freq, np.zeros(sim.nfreq, dtype=complex), 'r', label='refl')
            self.lineT, = plt.plot(sim.freq, np.zeros(sim.nfreq, dtype=complex), 'b', label='tran')
            self.lineS, = plt.plot(sim.freq, np.zeros(sim.nfreq, dtype=complex), 'g', label='source')
            plt.legend()
            plt.subplot(3, 1, 2)
            plt.title('FFT Relative to Source')
            plt.grid(True)
            self.lineRS, = plt.plot(sim.freq, np.zeros(sim.nfreq, dtype=complex), 'r', label='refl/source')
            self.lineTS, = plt.plot(sim.freq, np.zeros(sim.nfreq, dtype=complex), 'b', label='tran/source')
            self.lineTOT, = plt.plot(sim.freq, np.zeros(sim.nfreq, dtype=complex),'g', label='sum')
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.title('Time Domain')
            plt.grid(True)
            for k, source in sim.sources.items():
                if source.__class__.__name__ == 'Recorder':
                    plt.plot(source.history, label=k)
            plt.legend()
            plt.tight_layout()

    def process(self, sim, drawNow=False, step_skip = 10, ):

        if not sim.step % step_skip or drawNow:
            newtime = time.time()
            timedelta = newtime - self.lasttime
            self.lasttime = newtime

            if self.plot_eh:
                plt.figure(self.fig.number)
                plt.sca(self.ax)
                self.lineH.set_ydata(sim.H[0,0,:,0])
                self.lineE.set_ydata(sim.E[0,0,:,1])
                self.annoText.set_text('{}; {:.2f}/s'.format(sim.step,step_skip/timedelta))
                plt.gca().relim()
                plt.gca().autoscale_view(True, False, True)
                #plt.savefig('{}/{}.png'.format('gif',sim.step))

            if self.plot_f:
                plt.figure(self.fig2.number)
                refl = np.square(np.abs(sim.sources['refl'].fft))
                tran = np.square(np.abs(sim.sources['tran'].fft))
                source = np.square(np.abs(sim.sources['source'].fft))
                plt.subplot(3, 1, 1)
                self.lineR.set_ydata(refl)
                self.lineT.set_ydata(tran)
                self.lineS.set_ydata(source)
                plt.gca().relim()
                plt.gca().autoscale_view(True, False, True)
                plt.subplot(3, 1, 2)
                self.lineRS.set_ydata(np.divide(refl,source))
                self.lineTS.set_ydata(np.divide(tran,source))
                self.lineTOT.set_ydata(np.divide(refl,source)+np.divide(tran,source))
                plt.gca().relim()
                plt.gca().autoscale_view(True, False, True)
                plt.subplot(3, 1, 3)
                plt.cla()
                for k, source in sim.sources.items():
                    if k in ['source', 'refl', 'tran']:
                        plt.plot(source.history, label=k)
                plt.legend()
                #plt.savefig('{}/{}.png'.format('gif',sim.step))

            plt.show()

    def postprocess(self,sim):
        plt.ioff()
        plt.show()


class Job:
    '''Specifies, configures, and runs a simulation.'''
    # unload() >> load() >> config[] >> stage() >> run()
    def __init__(self,files=None):
        self.solved = False
        self.unloaded = False

        self.config = {}
        self.solver = SimFDTD()

        self.unload()
        self.load(files)

    def load(self, files = None):
        '''Loads a simulation from a set of files, in order.'''
        if files:
            if isinstance(files,str):
                files = [files]
            for file in files:
                with open(file) as f:
                    newsetup = yaml.load(f)
                    self.config = {**self.config, **newsetup} # Combine dicts

        self.unloaded = False

    def unload(self):
        '''Remove all loaded job and simulation configuration.'''
        with open('default.json') as fp:
            self.config = yaml.load(fp)
        self.unloaded = True

    def initialize(self):
        if isinstance(self.config['config']['processor'],str):
            self.config['config']['processor'] = eval(self.config['config']['processor'])

        self.solver.setup(**self.config['config'])

        for source in self.config['config']['sources']:
            if isinstance(self.config['config']['sources'][source],str):
                self.config['config']['sources'][source] = eval(self.config['config']['sources'][source])

    def solve(self):
        '''Run simulator.'''


        self.solver.run()

        self.solved = True

    def unsolve(self):
        '''Removes solution.'''
        self.solved = False

class SimFDTD:
    def setup(self,steps,er,ur,error,dt=None,d=None,fmin=None,fmax=None,nfreq=None,processor=Processor(),sources=None,n=None,breakpoints=None):
        self.u0 = 1.2566370614e-6  # permeability
        self.e0 = 8.854187817e-12  # permittivity
        self.c0 = 1 / np.sqrt(self.u0 * self.e0)  # speed of light
        self.eta0 = np.sqrt(self.u0 / self.e0) #impedance
        self.n0 = np.sqrt(self.u0 * self.e0)# refractive index
        self.processor = processor
        self.sources = sources

        if breakpoints:
            breakpoints.sort(reverse=True)
            self.breakpoints = breakpoints
        else:
            self.breakpoints = []
        self.breakpoint = None

        self.step = 0
        self.steps = steps
        #TODO: Implement end of sim when error is small enough
        self.error = error
        self.fmax = fmax
        self.fmin = fmin
        self.nfreq = nfreq
        self.freq = np.linspace(self.fmin, self.fmax, self.nfreq)

        if d:
            self.d = d
        else:
            # TODO: Get rid of nmax dependency, this isn't valid
            self.d = self.c0 / fmax / self.nmax / 20

        self.n = n

        x = n[0]*self.d
        y = n[1]*self.d
        z = n[2]*self.d
        self.dim = sum(1 for count in self.n if count > 1)

        # Can be used in equation based materials.
        x,y,z = np.meshgrid(    np.arange(0, n[0]*self.d, d),
                                np.arange(0, n[1]*self.d, d),
                                np.arange(0, n[2]*self.d, d),
                                sparse=False)

        if isinstance(er, str):
            er = eval(er)
        if not hasattr(er, "__iter__"):
            self.er = np.ones(n) * er
        else:
            # TODO: Fix for other dimensions
            f = interpolate.interp1d(np.linspace(0, self.n[2], len(er)), er, kind='nearest')
            self.er = np.ones(n)
            self.er[0,0,:] = f(range(self.n[2]))

        if isinstance(ur, str):
            ur = eval(ur)
        if not hasattr(ur, "__iter__"):
            self.ur = np.ones(n) * ur
        else:
            #TODO: Implement this for other axes
            f = interpolate.interp1d(np.linspace(0, self.n[2], len(ur)), ur, kind='nearest')
            self.ur[0,0,:] = f(range(self.n[2]))

        self.nmax = np.sqrt(np.max(ur) * np.max(er))

        if dt:
            self.dt = dt
        else:
            # Courant Stability Condition
            #TODO: Use real eta at boundaries, not just assume 1
            eta_bc = 1  # refractive index at boundaries
            self.dt = self.d / self.c0 * eta_bc / 2

        self.z0 = np.sqrt(self.u0/self.e0)

        # Compute Update Coefficients
        self.mE = self.c0 * self.dt / self.er
        self.mH = - self.c0 * self.dt / self.ur

        # Initialize Fields to Zero
        self.E = np.zeros(self.n+[3])
        self.H = np.zeros(self.n+[3])

        # Setup Fourier Transforms
        self.kernel = [np.exp(-1j*2*np.pi*self.dt*x) for x in self.freq]

    def __init__(self):
        self.sources = {}

    def console(self):
        #ensure we have a breakpoint
        if not self.breakpoint:
            if self.breakpoints:
                self.breakpoint = self.breakpoints.pop()
            else:
                return

        #check for obsolete breakpoints
        while self.breakpoint < self.step:
            if self.breakpoints:
                self.breakpoint = self.breakpoints.pop()
            else:
                return

        #handle breakpoint
        if self.breakpoint:
            if self.breakpoint == self.step:
                self.processor.process(self, drawNow=True)
                choice = input("{}: ".format(self.step))
                if choice:
                    if choice[0] == 'c': #continue
                        self.breakpoint = None
                        return
                    elif choice[0] == 'j': #jump
                        self.breakpoint = int(choice.split(' ')[1])
                        return
                    elif choice[0] == 'a': #add
                        self.breakpoints.append(int(choice.split(' ')[1]))
                        self.breakpoints.sort(reverse=True)
                        print('breakpoints = {}'.format(self.breakpoints))
                        return
                    elif choice[0] == 's': #step
                        self.breakpoint += 1
                        return
                    else:
                        self.console()
                else:
                    self.breakpoint += 1 #step by default

    def run(self,plot=True):
        print("Go!")
        self.processor.preprocess(self)
        print("Pre-processing complete.")
        # Loop
        finished = False

        for self.step in range(1, self.steps + 1):
            self.console()

            # Update H from E
            for key,source in self.sources.items():
                source.injectBeforeH()

            for i,j,k in np.ndindex(self.H.shape[:-1]):
                self.H[i,j,k][0] =    self.H[i,j,k][0] \
                                    + self.mH[i,j,k] * (self.E[i,(j+1)%self.n[1],k][2] - self.E[i,j,k][2]) / self.d\
                                    - self.mH[i,j,k] * (self.E[i,j,(k+1)%self.n[2]][1] - self.E[i,j,k][1]) / self.d
                self.H[i,j,k][1] =    self.H[i,j,k][1]\
                                    + self.mH[i,j,k] * (self.E[i,j,(k+1)%self.n[2]][0] - self.E[i,j,k][0]) / self.d\
                                    - self.mH[i,j,k] * (self.E[(i+1)%self.n[0],j,k][2] - self.E[i,j,k][2]) / self.d
                self.H[i,j,k][2] =    self.H[i,j,k][2]\
                                    + self.mH[i,j,k] * (self.E[(i+1)%self.n[0],j,k][1] - self.E[i,j,k][1]) / self.d\
                                    - self.mH[i,j,k] * (self.E[i,(j+1)%self.n[1],k][0] - self.E[i,j,k][0]) / self.d

            for k,source in self.sources.items():
                source.injectAfterH()

            # Update E from H
            for k,source in self.sources.items():
                source.injectBeforeE()

            for i,j,k in np.ndindex(self.E.shape[:-1]):
                self.E[i,j,k][0] =    self.E[i,j,k][0]\
                                    + self.mE[i,j,k] * (self.H[i,j,k][2] - self.H[i,j-1,k][2]) / self.d\
                                    - self.mE[i,j,k] * (self.H[i,j,k][1] - self.H[i,j,k-1][1]) / self.d
                self.E[i,j,k][1] =    self.E[i,j,k][1]\
                                    + self.mE[i,j,k] * (self.H[i,j,k][0] - self.H[i,j,k-1][0]) / self.d\
                                    - self.mE[i,j,k] * (self.H[i,j,k][2] - self.H[i-1,j,k][2]) / self.d
                self.E[i,j,k][2] =    self.E[i,j,k][2]\
                                    + self.mE[i,j,k] * (self.H[i,j,k][1] - self.H[i-1,j,k][1]) / self.d\
                                    - self.mE[i,j,k] * (self.H[i,j,k][0] - self.H[i,j-1,k][0]) / self.d

            for k,source in self.sources.items():
                source.injectAfterE()

            for k,source in self.sources.items():
                source.log()

            self.processor.process(self)

            if finished:
                break


        # Post-processing
        print("Run complete.")
        self.processor.postprocess(self)
        print("Post-processing complete.")

class SourceSink():
    def injectBeforeH(self):
        pass
    def injectAfterH(self):
        pass
    def injectBeforeE(self):
        pass
    def injectAfterE(self):
        pass
    def log(self):
        pass

class GaussianSource(SourceSink):
    class Config():
        def __init__(self, tau, pos=(0,0,0), vec=0, mag=1, efield=True, hard=False):
            self.type = 'source'
            self.tau = tau
            self.pos = pos
            self.vec = vec
            self.mag = mag
            self.efield = efield
            self.hard = hard
    def __init__(self,sim,*args,**kwargs):
        self.cfg = self.Config(*args,**kwargs)
        self.sim = sim
        self.fft = np.zeros(self.sim.nfreq, dtype=complex)
        self.history = []
    def injectAfterE(self):
        if self.cfg.efield:
            self.sim.E[self.cfg.pos][self.cfg.vec] += self.cfg.mag * np.exp(-((self.sim.step * self.sim.dt - 6 * self.cfg.tau) / self.cfg.tau) ** 2)
        else:
            self.sim.H[self.cfg.pos][self.cfg.vec] += self.cfg.mag * np.exp(-((self.sim.step * self.sim.dt - 6 * self.cfg.tau) / self.cfg.tau) ** 2)
    def log(self):
        if self.cfg.efield:
            self.history.append(self.sim.E[self.cfg.pos][self.cfg.vec])
            for nf in range(self.sim.nfreq):
                self.fft[nf] = self.fft[nf] + (self.sim.kernel[nf] ** (self.sim.step + 1)) * self.sim.E[self.cfg.pos][self.cfg.vec]
        else:
            self.history.append(self.sim.H[self.cfg.pos][self.cfg.vec])
            for nf in range(self.sim.nfreq):
                self.fft[nf] = self.fft[nf] + (self.sim.kernel[nf] ** (self.sim.step + 1)) * self.sim.H[self.cfg.pos][self.cfg.vec]

#TODO move gsource to utility module or parent class
#TODO move slice_shifted to utility module, or parent class
class TotalFieldScatteredSource(SourceSink):
    # Shout-out to Huygen
    # https://kb.lumerical.com/en/ref_sim_obj_tfsf_sources_examples.html
    # https://www.e-fermat.org/files/articles/Potter-ART-2017-Vol19-Jan._Feb.-001%20A%20Review%20of%20the%20Total%20Field..........pdf
    class Config():
        def slice_shifted(self, s, offset):
            return slice(s.start + offset, s.stop + offset, s.step)
        def __init__(self,tau, pos=(0,0,0), E=(0,1,0), H=(-1,0,0), mag=1, efield=True, hard=False):
            self.type = 'source'
            self.tau = tau
            self.pos = pos
            self.mag = mag
            self.efield = efield
            self.hard = hard
            self.E = E
            self.H = H
            self.pos2 = []

            for axis_slice,offset in zip(pos,np.clip(np.cross(E,H),-1,0)):
                if isinstance(axis_slice,slice):
                    self.pos2.append(self.slice_shifted(axis_slice,offset))
                else:
                    self.pos2.append(axis_slice+offset)
            self.pos2 = tuple(self.pos2)
            #self.pos2 = tuple(pos + np.clip(np.cross(E,H),-1,0))
    def __init__(self,sim,*args,**kwargs):
        self.cfg = self.Config(*args,**kwargs)
        self.sim = sim
        self.nsrc = np.sqrt(self.sim.ur[self.cfg.pos] * self.sim.er[self.cfg.pos]) #n
        self.etasrc = np.sqrt(self.sim.ur[self.cfg.pos] / self.sim.er[self.cfg.pos]) #n
        self.history = []
        self.last = 0
        self.fft = np.zeros(self.sim.nfreq, dtype=complex)
        self.lastE = [0, 0, 0]
        self.lastH = [0, 0, 0]
    def gsource(self,t,tau,t0=None,mag=1):
        if not t0:
            t0 = tau*6
        return mag*np.exp(-((t-t0)/tau)**2)
    def injectAfterE(self):
        for component,axis in zip(self.cfg.E,range(3)):
            self.lastE[axis] = component*self.sim.mE[self.cfg.pos] * np.sqrt(self.sim.er[self.cfg.pos] / self.sim.ur[self.cfg.pos]) * self.gsource(t=self.sim.step * self.sim.dt + self.nsrc*self.sim.d / 2 / self.sim.c0 + self.sim.dt / 2, tau=self.cfg.tau) / self.sim.d
            self.sim.E[self.cfg.pos+(axis,)] +=self.lastE[axis]
    def injectAfterH(self):
        for component,axis in zip(self.cfg.H,range(3)):
            self.lastH[axis] = component*self.cfg.mag * self.sim.mH[self.cfg.pos] * self.gsource( t=self.sim.step * self.sim.dt, tau=self.cfg.tau) / self.sim.d
            self.sim.H[self.cfg.pos2+(axis,)] += self.lastH[axis]
    def log(self):
        if self.cfg.efield:
            self.history.append(np.linalg.norm(2*self.lastE))
            for nf in range(self.sim.nfreq):
                self.fft[nf] = self.fft[nf] + (self.sim.kernel[nf] ** (self.sim.step + 1)) * 2*np.linalg.norm(self.lastE)

class SineSource(SourceSink):
    class Config():
        def __init__(self,freq,k,mag=1,efield=True,hard=False):
            self.freq = freq
            self.type = 'source'
            self.hard = hard
            self.k = k
            self.mag = mag
            self.efield = efield
    def __init__(self,sim,*args,**kwargs):
        self.cfg = self.Config(*args,**kwargs)
        self.sim = sim
        self.history = []
        self.fft = np.zeros(self.sim.nfreq, dtype=complex)
    def injectAfterE(self):
        self.lastE = self.cfg.mag*np.sin(2*np.pi*self.cfg.freq*self.sim.step*self.sim.dt)
        self.sim.E[0,0,self.cfg.k][1]   += self.sim.mE[0,0,self.cfg.k] * self.lastE / self.sim.d
    def log(self):
        if self.cfg.efield:
            self.history.append(self.lastE)
            for nf in range(self.sim.nfreq):
                self.fft[nf] = self.fft[nf] + (self.sim.kernel[nf] ** (self.sim.step + 1)) * self.lastE
        else:
            self.history.append(self.lastH)
            for nf in range(self.sim.nfreq):
                self.fft[nf] = self.fft[nf] + (self.sim.kernel[nf] ** (self.sim.step + 1)) * self.lastH

#TODO This implementation is likely not correct.
class PerfectAbsorbingBoundary(SourceSink):
    class Config():
        def __init__(self, pos=(0,0,0), efield=True):
            self.type = 'sink'
            self.pos = pos
            self.efield = efield
    def __init__(self, sim, *args, **kwargs):
        self.cfg = self.Config(*args, **kwargs)
        self.sim = sim
        self.length = 1
        for axis_slice,axis_n in zip(self.cfg.pos, self.sim.n):
            if isinstance(axis_slice, slice):
                self.length *= len(range(*axis_slice.indices(axis_n)))
        self.h = np.zeros([3]+[self.length]+[3])
        self.e = np.zeros([3]+[self.length]+[3])
        self.history = []
        self.fft = np.zeros(self.sim.nfreq, dtype=complex)
    def injectBeforeH(self):
        if not self.cfg.efield:
            self.h[0,...,0] = self.sim.H[self.cfg.pos+(0,)] \
                         + self.sim.mH[self.cfg.pos] * (self.e[2,...,2] - self.sim.E[self.cfg.pos+(2,)]) / self.sim.d \
                         - self.sim.mH[self.cfg.pos] * (self.e[2,...,1] - self.sim.E[self.cfg.pos+(1,)]) / self.sim.d
            self.h[0,...,1] = self.sim.H[self.cfg.pos+(1,)] \
                         + self.sim.mH[self.cfg.pos] * (self.e[2,...,0] - self.sim.E[self.cfg.pos+(0,)]) / self.sim.d \
                         - self.sim.mH[self.cfg.pos] * (self.e[2,...,2] - self.sim.E[self.cfg.pos+(2,)]) / self.sim.d
            self.h[0,...,2] = self.sim.H[self.cfg.pos+(2,)] \
                         + self.sim.mH[self.cfg.pos] * (self.e[2,...,1] - self.sim.E[self.cfg.pos+(1,)]) / self.sim.d \
                         - self.sim.mH[self.cfg.pos] * (self.e[2,...,0] - self.sim.E[self.cfg.pos+(0,)]) / self.sim.d
    def injectAfterH(self):
        if self.cfg.efield:
            self.h[2,...,0] = self.h[1,...,0]
            self.h[1,...,0] = self.h[0,...,0]
            self.h[0,...,0] = self.sim.H[self.cfg.pos+(0,)]
            self.h[2,...,1] = self.h[1,...,1]
            self.h[1,...,1] = self.h[0,...,1]
            self.h[0,...,1] = self.sim.H[self.cfg.pos+(1,)]
            self.h[2,...,2] = self.h[1,...,2]
            self.h[1,...,2] = self.h[0,...,2]
            self.h[0,...,2] = self.sim.H[self.cfg.pos+(2,)]
        if not self.cfg.efield:
            self.sim.H[self.cfg.pos] = self.h[0]
    def injectBeforeE(self):
        if self.cfg.efield:
            self.e[0,...,0] = self.sim.E[self.cfg.pos+(0,)] \
                         + self.sim.mE[self.cfg.pos] * (self.sim.H[self.cfg.pos+(2,)] - self.h[2,...,2]) / self.sim.d \
                         - self.sim.mE[self.cfg.pos] * (self.sim.H[self.cfg.pos+(1,)] - self.h[2,...,1]) / self.sim.d
            self.e[0,...,1] = self.sim.E[self.cfg.pos+(1,)] \
                         + self.sim.mE[self.cfg.pos] * (self.sim.H[self.cfg.pos+(0,)] - self.h[2,...,0]) / self.sim.d \
                         - self.sim.mE[self.cfg.pos] * (self.sim.H[self.cfg.pos+(2,)] - self.h[2,...,2]) / self.sim.d
            self.e[0,...,2] = self.sim.E[self.cfg.pos+(2,)] \
                         + self.sim.mE[self.cfg.pos] * (self.sim.H[self.cfg.pos+(1,)] - self.h[2,...,1]) / self.sim.d \
                         - self.sim.mE[self.cfg.pos] * (self.sim.H[self.cfg.pos+(0,)] - self.h[2,...,0]) / self.sim.d
    def injectAfterE(self):
        if self.cfg.efield:
            self.sim.E[self.cfg.pos] = self.e[0]
        if not self.cfg.efield:
            self.e[2,...,0] = self.e[1,...,0]
            self.e[1,...,0] = self.e[0,...,0]
            self.e[0,...,0] = self.sim.E[self.cfg.pos+(0,)]
            self.e[2,...,1] = self.e[1,...,1]
            self.e[1,...,1] = self.e[0,...,1]
            self.e[0,...,1] = self.sim.E[self.cfg.pos+(1,)]
            self.e[2,...,2] = self.e[1,...,2]
            self.e[1,...,2] = self.e[0,...,2]
            self.e[0,...,2] = self.sim.E[self.cfg.pos+(2,)]
    def log(self):
        if self.cfg.efield:
            self.history.append(np.linalg.norm(self.e[0]))
            for nf in range(self.sim.nfreq):
                self.fft[nf] = self.fft[nf] + (self.sim.kernel[nf] ** (self.sim.step + 1)) * np.linalg.norm(self.e[0])
        else:
            self.history.append(np.linalg.norm(self.h[0]))
            for nf in range(self.sim.nfreq):
                self.fft[nf] = self.fft[nf] + (self.sim.kernel[nf] ** (self.sim.step + 1)) * np.linalg.norm(self.h[0])

class DirichletBoundaryCondition(SourceSink):
    '''
    https://en.wikipedia.org/wiki/Dirichlet_boundary_condition
    '''
    class Config():
        def __init__(self, sim, pos=(0,0,0,0), efield=True):
            self.type = 'sink'
            # TODO: Should allow equation-based definition.
            if isinstance(pos, str):
                nx, ny, nz, nd = np.meshgrid(range(sim.n[0]),
                                         range(sim.n[1]),
                                         range(sim.n[2]),
                                         range(3),
                                         sparse=False,
                                         indexing='ij')
                self.pos = eval(pos)
            else:
                self.pos = pos
            self.efield = efield
    def __init__(self, sim, *args, **kwargs):
        # TODO: Update all Config() to pass sim
        self.cfg = self.Config(sim, *args, **kwargs)
        self.sim = sim
    def injectAfterE(self):
        self.sim.E[self.cfg.pos] = 0
    def injectAfterH(self):
        self.sim.H[self.cfg.pos] = 0

# TODO need to implement radiation condition
class RadiationCondition(SourceSink):
    class Config():
        def __init__(self, pos=(0,0,0), vec=0, efield=True):
            self.type = 'sink'
            self.pos = pos
            self.vec = vec
            self.efield = efield
    def __init__(self, sim, *args, **kwargs):
        self.cfg = self.Config(*args, **kwargs)
        self.sim = sim

# TODO need to implement mur ABC, in work
class MurAbsorbingBoundaryCondition(SourceSink):
    class Config():
        def __init__(self, pos=(0,0,0), vec=0, efield=True):
            self.type = 'sink'
            self.pos = pos
            self.i = pos[0]
            self.j = pos[1]
            self.k = pos[2]
            self.vec = vec
            self.efield = efield
    def __init__(self, sim, *args, **kwargs):
        self.cfg = self.Config(*args, **kwargs)
        self.sim = sim
        self.e1 = 0
        self.e2 = 0
        self.e3 = 0
    def injectBeforeE(self):
        self.e1 = self.sim.E[self.cfg.pos][self.cfg.vec] #old
        self.e2 = self.sim.E[self.cfg.i,self.cfg.j,self.cfg.k+1][self.cfg.vec] #old neighbor
    def injectAfterE(self):
        self.sim.E[self.cfg.pos][self.cfg.vec] = self.e1 #old self
        self.e3 = self.sim.E[self.cfg.i,self.cfg.j,self.cfg.k+1][self.cfg.vec] #new neighbor
        '''
        self.E[i,j,k][1] =    self.E[i,j,k][1]\
                            - self.mE[i,j,k] * (self.H[i,j,k][2] - self.H[i-1,j,k][2]) / self.d
                            + self.mE[i,j,k] * (self.H[i,j,k][0] - self.H[i,j,k-1][0]) / self.d\
        '''
        '''
        self.sim.E[self.cfg.pos][self.cfg.vec] = self.e2\
                                                 + ((self.sim.c0*self.sim.dt-self.sim.d)/(self.sim.c0*self.sim.dt+self.sim.d))\
                                                 *(self.e3-self.e1)
        '''

#TODO may need updates after 2/3d conversion?
class Recorder(SourceSink):
    class Config():
        def __init__(self, pos=(0,0,0), vec=0, efield=True):
            self.type = 'silent'
            self.pos = pos
            self.vec = vec
            self.efield = efield
    def __init__(self, sim, *args, **kwargs):
        self.cfg = self.Config(*args, **kwargs)
        self.sim = sim
        self.history = []
        self.fft = np.zeros(self.sim.nfreq, dtype=complex)
    def log(self):
        if self.cfg.efield:
            self.history.append(self.sim.E[self.cfg.pos][self.cfg.vec])
            for nf in range(self.sim.nfreq):
                self.fft[nf] = self.fft[nf] + (self.sim.kernel[nf] ** (self.sim.step + 1)) * self.sim.E[self.cfg.pos][self.cfg.vec]
        else:
            self.history.append(self.sim.H[self.cfg.pos][self.cfg.vec])
            for nf in range(self.sim.nfreq):
                self.fft[nf] = self.fft[nf] + (self.sim.kernel[nf] ** (self.sim.step + 1)) * self.sim.H[self.cfg.pos][self.cfg.vec]

if __name__ == "__main__":
    profile = False
    options = [
        'EXIT',
        'default1d.yaml',
        'default2d.yaml',
        'default3d.yaml',
        'example_1d_dielectric_slab.yaml',
        'example_1d_quarterwavelength_transformer.yaml'
    ]

    # Print menu of config files
    width = 80
    print(int((width-6)/2) * "-", "MENU", int((width-6)/2) * "-")
    for index,option in zip(range(len(options)),options):
        print('{}. {}'.format(index,option))
    print(width * "-")

    # Get user input for config file number
    choice = int(input("Enter your choice: "))
    if 0 < choice < len(options):
        job = Job(files=options[int(choice)])
        job.initialize()
        if profile:
            cProfile.run('job.solve()', filename="heaviside.pstat")
        else:
            job.solve()

    quit()
