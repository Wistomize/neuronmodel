from neuron import h, gui
from neuron.units import ms, mV, µm
import neuron, matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.font_manager import FontProperties
import numpy as np
from mpl_toolkits.mplot3d import art3d
from scipy.interpolate import interp1d

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.weight'] = "bold"
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelpad'] = 5
matplotlib.rcParams['xtick.major.pad'] = 3
matplotlib.rcParams['ytick.major.pad'] = 3


AIS_LENGTH = 100
class BallAndStick:
    def __init__(self, gid):
        self._gid = gid
        self._setup_morphology()
        self._setup_biophysics()

    def _setup_morphology(self):
        self.soma = h.Section(name="soma", cell=self)
        self.dend = h.Section(name="dend", cell=self)
        self.axon = h.Section(name="axon", cell=self)
        self.AIS = h.Section(name="AIS", cell=self)
        self.soma.connect(self.dend)
        self.axon.connect(self.AIS)
        self.AIS.connect(self.soma)
        self.all = self.soma.wholetree()
        self.soma.L = self.soma.diam = 20 * µm
        self.dend.L = 200 * µm
        self.dend.diam = 1 * µm
        self.axon.L = 100 * µm
        self.axon.diam = 1 * µm
        self.AIS.diam = 1 * µm
        self.AIS.L = AIS_LENGTH * µm
        self.AIS.nseg = int(AIS_LENGTH/0.19)

    def _setup_biophysics(self):
        for sec in self.all:
            sec.Ra = 300  # Axial resistance in Ohm * cm
            sec.cm = 0.5  # Membrane capacitance in micro Farads / cm^2
            #sec.Rm = 10000

        self.soma.insert("hh")
        for seg in self.soma:
            seg.hh.gnabar = 0.012  # Sodium conductance in S/cm2
            seg.hh.gkbar = 0.0036  # Potassium conductance in S/cm2
            seg.hh.gl = 0.0003  # Leak conductance in S/cm2
            seg.hh.el = -65 * mV  # Reversal potential

        self.axon.insert("hh")
        for seg in self.axon:
            seg.hh.gnabar = 0.12  # Sodium conductance in S/cm2
            seg.hh.gkbar = 0.015  # Potassium conductance in S/cm2
            seg.hh.gl = 0.0003  # Leak conductance in S/cm2
            seg.hh.el = -65 * mV  # Reversal potential

        self.AIS.insert("hh")
        for seg in self.AIS:
            seg.hh.gnabar = 0.5  # Sodium conductance in S/cm2
            seg.hh.gkbar = 0.1  # Potassium conductance in S/cm2
            seg.hh.gl = 0.0003  # Leak conductance in S/cm2
            seg.hh.el = -65 * mV  # Reversal potential

        # Insert passive current in the dendrite                       # <-- NEW
        self.dend.insert("pas")  # <-- NEW
        for seg in self.dend:  # <-- NEW
            seg.pas.g = 0.001  # Passive conductance in S/cm2        # <-- NEW
            seg.pas.e = -65 * mV  # Leak reversal potential             # <-- NEW

        

    def __repr__(self):
        return "BallAndStick[{}]".format(self._gid)


my_cell = BallAndStick(0)

stim = h.IClamp(my_cell.dend(1))

stim.delay = 5
stim.dur = 10
stim.amp = 0.1
soma_v = h.Vector().record(my_cell.soma(0.5)._ref_v)
dend_v = h.Vector().record(my_cell.dend(0.5)._ref_v)
t = h.Vector().record(h._ref_t)

h.finitialize(-65 * mV)
h.continuerun(25 * ms)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig2, su = plt.subplots()
#ax.plot(t, soma_v, label = 'soma')
#ax.plot(t, dend_v, label = 'dend')

AIS_DIVISION = 101
AIS_dist = [x*AIS_LENGTH/(AIS_DIVISION-1) for x in range(AIS_DIVISION)] # define distance on the AIS (1/1000)
AIS_aptime = np.array([x*AIS_LENGTH/(AIS_DIVISION-1) for x in range(AIS_DIVISION)])
AIS_maxvtime = np.array([x*AIS_LENGTH/(AIS_DIVISION-1) for x in range(AIS_DIVISION)])
AIS_vlist = np.zeros((AIS_DIVISION, 1001))
for i in range(len(AIS_dist)):
    AIS_v = h.Vector().record(my_cell.AIS(AIS_dist[i]/AIS_LENGTH)._ref_v) # potential on a certain location
    h.finitialize(-65 * mV)
    h.continuerun(25 * ms)
    iter = 1
    while (AIS_v[iter] - AIS_v[iter-1] < 1):
        iter += 1
    AIS_aptime[i] = iter*25/1001
    
    for j in range(len(AIS_v)):
        AIS_vlist[i][j] = AIS_v[j] # AIS_vlist[i][j] = potential on the "i" location in the "j"th time 
    AIS_maxvtime[i] = np.argmax(AIS_v)*25/1001
        

AIS_d = AIS_dist
t, AIS_dist = np.meshgrid(t, AIS_dist)
wire = ax.plot_wireframe(t, AIS_dist, AIS_vlist, rstride=13, cstride=0)
nx, ny, _  = np.shape(wire._segments3d)
wire_x = np.array(wire._segments3d)[:, :, 0].ravel()
wire_y = np.array(wire._segments3d)[:, :, 1].ravel()
wire_z = np.array(wire._segments3d)[:, :, 2].ravel()
wire.remove()

# create data for a LineCollection
wire_x1 = np.vstack([wire_x, np.roll(wire_x, 1)])
wire_y1 = np.vstack([wire_y, np.roll(wire_y, 1)])
wire_z1 = np.vstack([wire_z, np.roll(wire_z, 1)])
to_delete = np.arange(0, nx*ny, ny)
wire_x1 = np.delete(wire_x1, to_delete, axis=1)
wire_y1 = np.delete(wire_y1, to_delete, axis=1)
wire_z1 = np.delete(wire_z1, to_delete, axis=1)
scalars = np.delete(wire_z, to_delete)

segs = [list(zip(xl, yl, zl)) for xl, yl, zl in \
                 zip(wire_x1.T, wire_y1.T, wire_z1.T)]

# Plots the wireframe by a line3DCollection
my_wire = art3d.Line3DCollection(segs, cmap="winter")
my_wire.set_array(scalars)
ax.add_collection(my_wire)



midaptime = []
midap = []
tag = AIS_aptime[0]
tagloc = 0
for i in range(1, len(AIS_aptime)):
    if (AIS_aptime[i] != tag):
        midaptime.append((i+tagloc)/2)
        midap.append(tag)
        tag = AIS_aptime[i]
        tagloc = i
midaptime.append((i+tagloc)/2)
midap.append(tag)

# make it smoother
cubic_interpolation_model = interp1d(midaptime, midap, kind = "cubic")
midaptime_=np.linspace(min(midaptime), max(midaptime), 500)
midap_=cubic_interpolation_model(midaptime_)
#su.scatter(AIS_d, AIS_aptime)
su.scatter(midaptime, midap, color = 'red', label='AP initiation point')
su.plot(midaptime_, midap_, label='Fitted curve')

legendfont = {'size':14, 'weight':'bold', 'family':'sans-serif'}
labelfont = {'size':16, 'weight':'bold', 'family':'sans-serif'}
axisparams = {'labelsize':12, 'pad':2}

ax.set_xlabel('time (ms)', fontdict = labelfont)
ax.set_ylabel('distance (µm)', fontdict = labelfont)
ax.set_zlabel('Membrane  \npotential (mV)', fontdict = labelfont, labelpad = 10)
ax.xaxis.set_tick_params(**axisparams)
ax.yaxis.set_tick_params(**axisparams)
'''
axisfont = FontProperties()
axisfont.set_family('sans-serif')
axisfont.set_size(12)
axisfont.set_weight('bold')
axisfont.set_size(14)
ax.set_xticklabels([label.get_text() for label in ax.get_xticklabels()], fontproperties=axisfont)
'''

axisparams = {'labelsize':14, 'pad':2}
su.set_xlabel('distance (µm)', fontdict = labelfont)
su.set_ylabel('time (ms)', fontdict = labelfont)
su.xaxis.set_tick_params(**axisparams)
su.yaxis.set_tick_params(**axisparams)
su.legend(prop = legendfont)

fig.savefig('cell.tiff', dpi=600)
fig2.savefig('aptime.tiff', dpi=600)