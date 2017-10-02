from __future__ import division, absolute_import
"""For fitting and displaying SDSS plate profiles from dial indicator measurements
"""
import copy
import numpy
import numpy.linalg
from collections import OrderedDict
import itertools

import scipy.interpolate
import scipy.spatial
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import matplotlib.tri as mtri

ErrorTolerance = [-0.2, 0.2] #range in mm in which the profile error (measured - focal plane) is acceptable
MMPerInch = 25.4


DuPontFocalRadius = 8800 # mm


class DuPontMeasurement(object):
    measRadii = numpy.asarray([0.755, 3.75, 5.75, 7.75, 10.75]) * MMPerInch # Nick's measurements
    dirThetaMap = OrderedDict((
        (1, 6. * numpy.pi / 4.), # tab
        (2, 7. * numpy.pi / 4.), # tab + 45 degrees ccw
        (3, 0.), # tab + 90 degrees ccw
        (4, 1. * numpy.pi / 4.),
        (5, 2. * numpy.pi / 4.),
        (6, 3. * numpy.pi / 4.),
        (7, 4. * numpy.pi / 4.),
        (8, 5. * numpy.pi / 4.),
    ))
    def __init__(self, direction, measList):
        # direction is 1-8, measlist are values in mm meaured from small
        # radius to large
        self.measList = measList
        assert direction in self.dirThetaMap.keys()
        self.direction = direction

    @property
    def theta(self):
        return self.dirThetaMap[self.direction]

class DuPontProfile(object):
    def __init__(self):
        pass


def plateSurfPlot(x,y,z):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_trisurf(x, y, z, cmap=cm.coolwarm, vmin=ErrorTolerance[0], vmax=ErrorTolerance[1])
    ax.set_zlabel("focal plane error (mm)")
    ax.text(300, 0, 0, 'TAB', size=20, zorder=1, color='k')

def doNewInterp(measList):
    # http://stackoverflow.com/questions/22653956/using-scipy-spatial-delaunay-in-place-of-matplotlib-tri-triangulations-built-in
    # get x,y positions for r, thetas
    measList.sort(key=lambda x: x.theta)
    # sort by theta (0-2pi)
    rawRadii = measList[0].measRadii # all radii should be the same
    rawThetas = numpy.array([cc.theta for cc in measList] + [2*numpy.pi])
    rawMeas = numpy.array([cc.measList for cc in measList] + [measList[0].measList])
    # raw meas is 2D array ra
    thetaInterp = numpy.linspace(rawThetas[0], rawThetas[-1], 40)
    radiiInterp = numpy.linspace(rawRadii[0], rawRadii[-1], 20)
    radInterpList = []
    for radialMeas in rawMeas:
        spl = scipy.interpolate.spline(rawRadii, radialMeas, radiiInterp)
        radInterpList.append(spl)
    radInterpList = numpy.asarray(radInterpList).T
    fullInterp = []
    for radInterp  in radInterpList:
        spl = scipy.interpolate.spline(rawThetas, radInterp, thetaInterp)
        fullInterp.append(spl)
    fullInterp = numpy.array(fullInterp).T
    xInterp = []
    yInterp = []
    measInterp = []
    model = []
    areaUnits = [] # for determining percent of plate out of spec
    errorUnits = [] # for determining percent of plate out of spec
    for theta, interpMeas in itertools.izip(thetaInterp, fullInterp):
        # theta - 90 to make tab at -y on plot, N == tab direction == 0 theta.
        # theta increases in the normal way counter clockwise from x axis
        xInterp.append(numpy.cos(theta)*radiiInterp)
        yInterp.append(numpy.sin(theta)*radiiInterp)
        measInterp.append(interpMeas)
        model.append(numpy.sqrt(DuPontFocalRadius**2-radiiInterp**2))
        # for determining percent of plate out of spec
        if theta < 2*numpy.pi:
            # don't count last 2pi, it was added to ensure the spline 0==2pi
            for rad1, rad2, measurement in itertools.izip(radiiInterp[:-1], radiiInterp[1:], interpMeas[:-1]):
                errorUnits.append(numpy.sqrt(DuPontFocalRadius**2-rad1**2) - measurement)
                areaUnits.append(numpy.pi*(rad2**2-rad1**2)/39.0) # 40 for interpolated theta
    xInterp = numpy.array(xInterp).flatten()
    yInterp = numpy.array(yInterp).flatten()
    measInterp = numpy.array(measInterp).flatten()
    model = numpy.array(model).flatten()
    err = model - measInterp
    err = err - numpy.mean(err) # maybe use median?
    plateSurfPlot(xInterp, yInterp, err)
    errorUnits = numpy.asarray(errorUnits)
    errorUnits = errorUnits - numpy.mean(errorUnits)
    areaUnits = numpy.asarray(areaUnits)
    #determine out of specness
    errorUnits = numpy.abs(errorUnits)
    outOfSpecInds = numpy.argwhere(errorUnits>0.2)
    areaOutOfSpec = numpy.sum(areaUnits[outOfSpecInds])
    totalArea = numpy.sum(areaUnits)
    percentInSpec = 100 - areaOutOfSpec/totalArea * 100
    print("percent of plate in spec: %.2f"%percentInSpec)




