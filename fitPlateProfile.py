from __future__ import division, absolute_import
"""For fitting and displaying SDSS plate profiles from dial indicator measurements
"""
import copy
import numpy
import numpy.linalg
from collections import OrderedDict

import scipy.interpolate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata

MMPerInch = 25.4
#FocalRad = 9000 # MM, du Pont telescope focal plane radius of curvature.
MeasRadii = numpy.asarray([0.8, 3.8, 5.8, 7.8, 10.8]) * MMPerInch
MeasRadii = numpy.asarray([0.755, 3.75, 5.75, 7.75, 10.75]) * MMPerInch # Nick's measurements
MeasRadiiCurtis = numpy.asarray([25.4, 76.2, 152.4, 228.6, 279.4])
DirThetaMapCard = OrderedDict((
    ("N", 0.),
    ("NE", 7. * numpy.pi / 4.),
    ("E", 3. * numpy.pi / 2.),
    ("SE", 5. * numpy.pi / 4.),
    ("S", numpy.pi),
    ("SW", 3. * numpy.pi / 4.),
    ("W", numpy.pi / 2.),
    ("NW", numpy.pi / 4.),
))

DirThetaMapHour = OrderedDict((
    ("twelve", 0 * 2 * numpy.pi / 12.),
    ("one", 1 * 2 * numpy.pi / 12.),
    ("two", 2 * 2 * numpy.pi / 12.),
    ("three", 3 * 2 * numpy.pi / 12.),
    ("four", 4 * 2 * numpy.pi / 12.),
    ("five", 5 * 2 * numpy.pi / 12.),
    ("six", 6 * 2 * numpy.pi / 12.),
    ("seven", 7 * 2 * numpy.pi / 12.),
    ("eight", 8 * 2 * numpy.pi / 12.),
    ("nine", 9 * 2 * numpy.pi / 12.),
    ("ten", 10 * 2 * numpy.pi / 12.),
    ("eleven", 11 * 2 * numpy.pi / 12.),

))

def plotRadialSurface(R, T, Z, zlim=None, plate=None, newFig=True, ax=None):
    #http://stackoverflow.com/questions/3526514/problem-with-2d-interpolation-in-scipy-non-rectangular-grid
    #http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
    if newFig:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        if zlim:
            ax.set_zlim(zlim)
        plt.xlabel("+West")
        plt.ylabel("+North")
        titleTxt = "focal plane residual (mm)"
        if plate is not None:
            titleTxt += "\nplate %i"%plate
        plt.title(titleTxt)
    surf = ax.plot_surface(R*numpy.cos(T), R*numpy.sin(T), Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    return ax
    #plt.show()

class RadialSurfaceProfile(object):
    """Generic class describing a radial profile in cylindrical coords
    """
    def __init__(self, radii, thetas, zs, plate=None):
        """ radii: 1D increasing array of sampled radii
            thetas: 1D array of thetas in [0, 2pi]: note one *should* include both 0 and 2pi values for
                polar continuity...0=2pi.
            zs: 2D array of z values. 0th index corresponds to theta, 1st index corresponds to radius.
                so shape is [len(thetas), len(radii)]
        """
        self.plate = plate
        radii = numpy.squeeze(radii)
        thetas = numpy.squeeze(thetas)
        zs = zs.squeeze()
        assert radii.ndim == 1
        assert thetas.ndim == 1
        assert zs.shape == (len(thetas), len(radii))
        self.radii, self.thetas, self.zs = radii, thetas, zs
        self.minRad = numpy.min(self.radii)
        self.maxRad = numpy.max(self.radii)
        # thetas are inteded to be [0, 2pi], not enforcing maybe it isn't always wanted?
        self.minTheta = numpy.min(self.thetas)
        self.maxTheta = numpy.max(self.thetas)

    def zRT(self, radius, theta):
        # return z position from r, theta
        # get intpolated position for any radius, theta
        # radius theta may be 1D arrays or floats
        R, T, Z = self.radialSurface()
        return griddata(R.flatten(), T.flatten(), Z.flatten(), radius, theta)

    def zXY(self, xPos, yPos):
        # return z position from x, y position
        r = numpy.sqrt(xPos**2 + yPos**2)
        theta = numpy.arctan2(yPos, xPos) + numpy.pi # [to keep in range 0, 2pi]
        return self.zRT(r, theta)

    def radialSurface(self):
        R, T = numpy.meshgrid(self.radii, self.thetas)
        return R, T, self.zs

    def interpRadialSurface(self, nRadii, nTheta):
        radiiInterp = numpy.linspace(self.minRad, self.maxRad, nRadii)
        thetaInterp = numpy.linspace(self.minTheta, self.maxTheta, nTheta)
        Rinterp, Tinterp = numpy.meshgrid(radiiInterp, thetaInterp)
        Zinterp = self.zRT(radiiInterp, thetaInterp)
        # return Rinterp, Tinterp, Zinterp
        return radiiInterp, thetaInterp, Zinterp

    def plotSampledSurface(self):
        R, T, Z = self.radialSurface()
        plotRadialSurface(R, T, Z, plate=self.plate)

    def plotInterpSurface(self, nRadii, nTheta):
        R, T, Z = self.interpRadialSurface(nRadii, nTheta)
        plotRadialSurface(R, T, Z, plate=self.plate)

class DuPontFocalProfile(RadialSurfaceProfile):
    focalRadius = 8800 # mm
    # zOffset is z distance from spherical origin to chord/surface describing
    # the plate location in the focal plane
    zOffsetToPlate = numpy.sqrt(focalRadius**2 - numpy.max(MeasRadii)**2)
    def __init__(self):
        # default to cardinal direction thetas (matching plate meas directions)
        # and radii matching plate profilometry radius measurements
        thetas = numpy.asarray(DirThetaMapHour.values() + [2*numpy.pi]) # add 2pi for boundary conditions
        zs = self.zRT(MeasRadii, thetas)
        # repeat zs for each theta
        zs = numpy.tile(zs, (len(thetas), 1))
        RadialSurfaceProfile.__init__(self, MeasRadii, thetas, zs)

    def zRT(self, radius, theta):
        # return z position from r, theta
        return numpy.sqrt(self.focalRadius**2-radius**2) - self.zOffsetToPlate

class Measurement(object):
    def __init__(self, direction, measList, dirThetaMap, toMM = False):
        # meas least must be in increasing radius
        # measurements in inches (converted to mm)
        assert len(measList) == len(MeasRadii)
        assert direction in dirThetaMap.keys(), direction + "  " + str(dirThetaMap.keys())
        self.direction = direction
        self.theta = dirThetaMap[direction]
        self.measList = numpy.asarray([measList])
        if toMM is True:
            self.measList = self.measList * MMPerInch


class CardinalMeasurement(Measurement):
    def __init__(self, direction, measList):
        Measurement.__init__(self, direction, measList, DirThetaMapCard, toMM = True)

class HourMeasurement(Measurement):
    def __init__(self, direction, measList):
        Measurement.__init__(self, direction, measList, DirThetaMapHour)

class PlateProfile(RadialSurfaceProfile):
    def __init__(self, measurementList, plate=None):
        self.plate = plate
        self.measurementList = measurementList
        # add an additional (identical) measurement as 0
        # at 2pi to enforce continuity
        for cml in self.measurementList:
            if cml.theta == 0:
                cml0 = copy.deepcopy(cml)
                break
        cml0.theta = 2 * numpy.pi
        measurementList.append(cml0)
        # sort in order of increasing theta (probably is already...)
        measurementList = sorted(measurementList, key=lambda cml: cml.theta)
        self.measurementList = measurementList
        thetas = numpy.asarray([cml.theta for cml in self.measurementList])
        zs = numpy.asarray([cml.measList for cml in self.measurementList]).squeeze()
        RadialSurfaceProfile.__init__(self, MeasRadii, thetas, zs)

class FocalPlaneFitter(object):
    def __init__(self, targetProfile, measuredProfile, plate=None):
        """ targetProfile a RadialSurfaceProfile, desired profile
            measuredProfile a RadialSurfaceProfile, measured profile
        """
        self.plate = plate
        self.targetProfile = targetProfile
        self.measuredProfile = measuredProfile

    def fitSurf(self):
        # determine the least squares fit and residuals
        # from the measured surface to target model
        rMeas = self.measuredProfile.radii
        tMeas = self.measuredProfile.thetas
        zMeas = self.measuredProfile.zs
        zModel = self.targetProfile.zRT(rMeas, tMeas)

    def plotSurfErr(self):
        rMeas = self.measuredProfile.radii
        tMeas = self.measuredProfile.thetas
        zMeas = self.measuredProfile.zs
        # interpolate r, t and z for a nicer plot?
        rMeas, tMeas, zMeas = self.measuredProfile.interpRadialSurface(10,36)
        zModel = self.targetProfile.zRT(rMeas, tMeas)
        surfZerr = zModel - zMeas
        # surfZerr = zModel
        surfZerr = surfZerr - numpy.mean(surfZerr)
        rGrid, tGrid = numpy.meshgrid(rMeas, tMeas)
        zLim = numpy.max(numpy.abs(surfZerr))*1.1
        zLim = (-1*zLim, zLim)
        # zLim = (0.02, .08)
        plotRadialSurface(rGrid, tGrid, surfZerr, zlim = zLim, plate=self.plate)

    def plot2Surf(self):
        rMeas = self.measuredProfile.radii
        tMeas = self.measuredProfile.thetas
        zMeas = self.measuredProfile.zs
        # interpolate r, t and z for a nicer plot?
        rMeas, tMeas, zMeas = self.measuredProfile.interpRadialSurface(10,36)
        zModel = self.targetProfile.zRT(rMeas, tMeas)
        rGrid, tGrid = numpy.meshgrid(rMeas, tMeas)
        maxZ = numpy.max(zModel)
        minZ = numpy.min(zModel)
        zLim = (minZ - .1 * minZ, maxZ + .1 * maxZ)
        # zLim = (0.02, .08)
        ax = plotRadialSurface(rGrid, tGrid, zModel, zlim = zLim, plate=self.plate)
        plotRadialSurface(rGrid, tGrid, zMeas, plate=self.plate, newFig=False, ax=ax)





### first profilometry plate 8770
cardMeasList1 = [
    CardinalMeasurement("N", [.130, .132, .124, .105, .048]),
    CardinalMeasurement("NE", [.127, .128, .120, .101, .048]),
    CardinalMeasurement("E", [.120, .1235, .115, .097, .0455]),
    CardinalMeasurement("SE", [.126, .123, .114, .095, .043]),
    CardinalMeasurement("S", [.125, .120, .111, .094, .043]),
    CardinalMeasurement("SW", [.127, .124, .114, .094, .042]),
    CardinalMeasurement("W", [.124, .123, .116, .098, .047]),
    CardinalMeasurement("NW", [.126, .127, .120, .102, .048]),
]

## 2nd prof of 8770
cardMeasList2 = [
    CardinalMeasurement("N", [.201, .178, .153, .121, .051]),
    CardinalMeasurement("NE", [.198, .174, .149, .117, .051]),
    CardinalMeasurement("E", [.200, .174, .149, .1165, .050]),
    CardinalMeasurement("SE", [.199, .174, .149, .116, .050]),
    CardinalMeasurement("S", [.198, .172, .148, .117, .050]),
    CardinalMeasurement("SW", [.198, .172, .148, .115, .049]),
    CardinalMeasurement("W", [.124, .197, .172, .148, .116]),
    CardinalMeasurement("NW", [.199, .174, .150, .118, .050]),
]

# plate 8771
cardMeasList3 = [
    CardinalMeasurement("N", [.217, .185, .156, .120, .050]),
    CardinalMeasurement("NE", [.215, .181, .152, .118, .050]),
    CardinalMeasurement("E", [.22, .1825, .1535, .1185, .050]),
    CardinalMeasurement("SE", [.217, .185, .157, .122, .051]),
    CardinalMeasurement("S", [.217, .186, .157, .122, .052]),
    CardinalMeasurement("SW", [.216, .186, .158, .122, .051]),
    CardinalMeasurement("W", [.215, .183, .155, .120, .051]),
    CardinalMeasurement("NW", [.216, .182, .154, .119, .050]),
]

# plate 8645
cardMeasList4 = [
    CardinalMeasurement("N", [.216, .186, .158, .123, .051]),
    CardinalMeasurement("NE", [.215, .184, .155, .120, .051]),
    CardinalMeasurement("E", [.21, .183, .1545, .12, .052]),
    CardinalMeasurement("SE", [.216, .184, .155, .120, .050]),
    CardinalMeasurement("S", [.215, .183, .154, .12, .051]),
    CardinalMeasurement("SW", [.213, .181, .152, .118, .049]),
    CardinalMeasurement("W", [.215, .182, .153, .118, .05]),
    CardinalMeasurement("NW", [.216, .185, .156, .121, .051]),
]

#no pin
cardMeasList5 = [
    CardinalMeasurement("N", [.164, .154, .138, .112, .049]),
    CardinalMeasurement("NE", [.160, .150, .134, .109, .049]),
    CardinalMeasurement("E", [.16, .1475, .132, .107, .0475]),
    CardinalMeasurement("SE", [.158, .147, .131, .106, .047]),
    CardinalMeasurement("S", [.158, .146, .131, .107, .048]),
    CardinalMeasurement("SW", [.158, .147, .132, .107, .047]),
    CardinalMeasurement("W", [.156, .146, .131, .107, .048]),
    CardinalMeasurement("NW", [.157, .147, .132, .108, .049]),
]

#no pin 8772
cardMeasList5 = [
    CardinalMeasurement("N", [.164, .154, .138, .112, .049]),
    CardinalMeasurement("NE", [.160, .150, .134, .109, .049]),
    CardinalMeasurement("E", [.16, .1475, .132, .107, .0475]),
    CardinalMeasurement("SE", [.158, .147, .131, .106, .047]),
    CardinalMeasurement("S", [.158, .146, .131, .107, .048]),
    CardinalMeasurement("SW", [.158, .147, .132, .107, .047]),
    CardinalMeasurement("W", [.156, .146, .131, .107, .048]),
    CardinalMeasurement("NW", [.157, .147, .132, .108, .049]),
]

#8772
cardMeasList6 = [
    CardinalMeasurement("N", [.241, .204, .170, .129, .053]),
    CardinalMeasurement("NE", [.239, .201, .167, .127, .053]),
    CardinalMeasurement("E", [.24, .201, .1665, .1265, .052]),
    CardinalMeasurement("SE", [.240, .200, .166, .126, .052]),
    CardinalMeasurement("S", [.24, .2, .166, .127, .053]),
    CardinalMeasurement("SW", [.239, .199, .165, .125, .051]),
    CardinalMeasurement("W", [.239, .2, .166, .126, .052]),
    CardinalMeasurement("NW", [.240, .201, .167, .127, .052]),
]

#8645 prebend pin in
cardMeasList7 = [
    # CardinalMeasurement("N", [.211, .186, .160, .122, .052]),
    CardinalMeasurement("N", [.216, .188, .160, .125, .052]),
    CardinalMeasurement("NE", [.209, .183, .157, .123, .052]),
    CardinalMeasurement("E", [.21, .182, .1555, .121, .052]),
    CardinalMeasurement("SE", [.210, .182, .155, .120, .051]),
    CardinalMeasurement("S", [.209, .18, .153, .12, .051]),
    CardinalMeasurement("SW", [.207, .179, .152, .119, .050]),
    CardinalMeasurement("W", [.208, .18, .153, .119, .051]),
    CardinalMeasurement("NW", [.210, .182, .156, .122, .051]),
]

#8645 prebend more weight pin in
cardMeasList8 = [
    CardinalMeasurement("N", [.285, .236, .194, .146, .058]),
    CardinalMeasurement("NE", [.284, .234, .193, .145, .058]),
    CardinalMeasurement("E", [.28, .2335, .1925, .1445, .0585]),
    CardinalMeasurement("SE", [.284, .233, .191, .144, .058]),
    CardinalMeasurement("S", [.284, .232, .191, .144, .058]),
    CardinalMeasurement("SW", [.283, .231, .190, .142, .057]),
    CardinalMeasurement("W", [.282, .231, .19, .143, .057]),
    CardinalMeasurement("NW", [.284, .233, .192, .144, .058]),
]

#releasing pin NW only
mm1 = [.265, .222, .184, .140, .057]
mm2 = [.255, .215, .180, .138, .056]
mm3 = [.239, .205, .173, .134, .055]
mm4 = [.233, .201, .170, .132, .055]
mm5 = [.228, .198, .168, .131, .055]
mm6 = [.220, .193, .165, .129, .054]
mm7 = [.211, .187, .161, .127, .054]

#8645 observed
mmN = numpy.asarray([.222, .193, .165, .129, .054])
mmE = numpy.asarray([.22, .193, .165, .1275, .053])
mmS = numpy.asarray([.22, .191, .162, .126, .053])
mmW = numpy.asarray([.222, .193, .165, .129, .054])
mmNE = numpy.mean(numpy.asarray([mmN, mmE]), axis=0)
mmSE = numpy.mean(numpy.asarray([mmS, mmE]), axis=0)
mmNW = numpy.mean(numpy.asarray([mmN, mmW]), axis=0)
mmSW = numpy.mean(numpy.asarray([mmS, mmW]), axis=0)
cardMeasList9 = [
    CardinalMeasurement("N", mmN),
    CardinalMeasurement("NE", mmNE),
    CardinalMeasurement("E", mmE),
    CardinalMeasurement("SE", mmSE),
    CardinalMeasurement("S", mmS),
    CardinalMeasurement("SW", mmSW),
    CardinalMeasurement("W", mmW),
    CardinalMeasurement("NW", mmNW),
]


#8770 Dec 2
cardMeasList10 = [
    CardinalMeasurement("N", [.222, .193, .165, .128, .053]),
    CardinalMeasurement("NE", [.219, .190, .161, .125, .053]),
    CardinalMeasurement("E", [.22, .190, .162, .125, .053]),
    CardinalMeasurement("SE", [.220, .190, .162, .125, .053]),
    CardinalMeasurement("S", [.22, .188, .16, .124, .053]),
    CardinalMeasurement("SW", [.221, .191, .162, .125, .053]),
    CardinalMeasurement("W", [.22, .19, .161, .125, .053]),
    CardinalMeasurement("NW", [.221, .192, .163, .127, .054]),
]

# Curtis Measurement at UW April 6 2016

hourMeasList1 = [
    HourMeasurement("twelve", numpy.asarray([-4.24, -4.11, -2.57, -2.54, -1.42])*-1.),
    HourMeasurement("one", numpy.asarray([-4.29, -4.14, -3.58, -2.54, -1.41])*-1.),
    HourMeasurement("two", numpy.asarray([-4.35, -4.19, -3.61, -2.55, -1.41])*-1.),
    HourMeasurement("three", numpy.asarray([-4.38, -4.22, -3.65, -2.57, -1.42])*-1.),
    HourMeasurement("four", numpy.asarray([-4.33, -4.18, -3.63, -2.98, -1.44])*-1.),
    HourMeasurement("five", numpy.asarray([-4.25, -4.1, -3.55, -2.53, -1.41])*-1.),
    HourMeasurement("six", numpy.asarray([-4.25, -4.1, -3.56, -2.53, -1.41])*-1.),
    HourMeasurement("seven", numpy.asarray([-4.3, -4.17, -3.62, -2.57, -1.43])*-1.),
    HourMeasurement("eight", numpy.asarray([-4.37, -4.24, -3.69, -2.61, -1.45])*-1.),
    HourMeasurement("nine", numpy.asarray([-4.39, -4.25, -3.69, -2.6, -1.44])*-1.),
    HourMeasurement("ten", numpy.asarray([-4.34, -4.21, -3.65, -2.57, -1.42])*-1.),
    HourMeasurement("eleven", numpy.asarray([-4.26, -4.13, -3.59, -2.54, -1.41])*-1.),
]

#july run
cardMeasList11 = [
    CardinalMeasurement("N", [.2840, .2465, .2125, .1705, .0765]),
    CardinalMeasurement("NE", [.2840, .2465, .2125, .1705, .0765]),
    CardinalMeasurement("E", [.2840, .2465, .2125, .1705, .0765]),
    CardinalMeasurement("SE", [.2840, .2465, .2125, .1705, .0765]),
    CardinalMeasurement("S", [.2840, .2465, .2125, .1705, .0765]),
    CardinalMeasurement("SW", [.2840, .2465, .2125, .1705, .0765]),
    CardinalMeasurement("W", [.2840, .2465, .2125, .1705, .0765]),
    CardinalMeasurement("NW", [.2840, .2465, .2125, .1705, .0765]),
]
#july run
cardMeasList12 = [
    CardinalMeasurement("N", [.3025, .2560, .2170, .1710, .0755]),
    CardinalMeasurement("NE", [.3025, .2560, .2170, .1710, .0755]),
    CardinalMeasurement("E", [.3025, .2560, .2170, .1710, .0755]),
    CardinalMeasurement("SE", [.3025, .2560, .2170, .1710, .0755]),
    CardinalMeasurement("S", [.3025, .2560, .2170, .1710, .0755]),
    CardinalMeasurement("SW", [.3025, .2560, .2170, .1710, .0755]),
    CardinalMeasurement("W", [.3025, .2560, .2170, .1710, .0755]),
    CardinalMeasurement("NW", [.3025, .2560, .2170, .1710, .0755]),
]

cardMeasList12 = [
    CardinalMeasurement("N", [.2950, .2520, .2140, .168, .0725]),
    CardinalMeasurement("NE", [.2950, .2520, .2140, .168, .0725]),
    CardinalMeasurement("E", [.2950, .2520, .2140, .168, .0725]),
    CardinalMeasurement("SE", [.2950, .2520, .2140, .168, .0725]),
    CardinalMeasurement("S", [.2950, .2520, .2140, .168, .0725]),
    CardinalMeasurement("SW", [.2950, .2520, .2140, .168, .0725]),
    CardinalMeasurement("W", [.2950, .2520, .2140, .168, .0725]),
    CardinalMeasurement("NW", [.2950, .2520, .2140, .168, .0725]),
]

cardMeasList12 = [
    CardinalMeasurement("N", [.290, .2485, .2120, .1675, .0725]),
    CardinalMeasurement("NE", [.290, .2485, .2120, .1675, .0725]),
    CardinalMeasurement("E", [.290, .2485, .2120, .1675, .0725]),
    CardinalMeasurement("SE", [.290, .2485, .2120, .1675, .0725]),
    CardinalMeasurement("S", [.290, .2485, .2120, .1675, .0725]),
    CardinalMeasurement("SW", [.290, .2485, .2120, .1675, .0725]),
    CardinalMeasurement("W", [.290, .2485, .2120, .1675, .0725]),
    CardinalMeasurement("NW", [.290, .2485, .2120, .1675, .0725]),
]


cardMeasList12 = [
    CardinalMeasurement("N", [.2830, .2460, .2120, .1690, .0750]),
    CardinalMeasurement("NE", [.2830, .2460, .2120, .1690, .0750]),
    CardinalMeasurement("E", [.2830, .2460, .2120, .1690, .0750]),
    CardinalMeasurement("SE", [.2830, .2460, .2120, .1690, .0750]),
    CardinalMeasurement("S", [.2830, .2460, .2120, .1690, .0750]),
    CardinalMeasurement("SW", [.2830, .2460, .2120, .1690, .0750]),
    CardinalMeasurement("W", [.2830, .2460, .2120, .1690, .0750]),
    CardinalMeasurement("NW", [.2830, .2460, .2120, .1690, .0750]),
]

cardMeasList12 = [
    CardinalMeasurement("N", [.280, .2445, .2110, .1685, .0750]),
    CardinalMeasurement("NE", [.280, .2445, .2110, .1685, .0750]),
    CardinalMeasurement("E", [.280, .2445, .2110, .1685, .0750]),
    CardinalMeasurement("SE", [.280, .2445, .2110, .1685, .0750]),
    CardinalMeasurement("S", [.280, .2445, .2110, .1685, .0750]),
    CardinalMeasurement("SW", [.280, .2445, .2110, .1685, .0750]),
    CardinalMeasurement("W", [.280, .2445, .2110, .1685, .0750]),
    CardinalMeasurement("NW", [.280, .2445, .2110, .1685, .0750]),
]

cardMeasList12 = [
    CardinalMeasurement("N", [.2700, .2380, .2075, .1665, .0745]),
    CardinalMeasurement("NE", [.2700, .2380, .2075, .1665, .0745]),
    CardinalMeasurement("E", [.2700, .2380, .2075, .1665, .0745]),
    CardinalMeasurement("SE", [.2700, .2380, .2075, .1665, .0745]),
    CardinalMeasurement("S", [.2700, .2380, .2075, .1665, .0745]),
    CardinalMeasurement("SW", [.2700, .2380, .2075, .1665, .0745]),
    CardinalMeasurement("W", [.2700, .2380, .2075, .1665, .0745]),
    CardinalMeasurement("NW", [.2700, .2380, .2075, .1665, .0745]),
]


cardMeasList12 = [
    CardinalMeasurement("N", [.2600, .2320, .2035, .1640, .0740]),
    CardinalMeasurement("NE", [.2600, .2320, .2035, .1640, .0740]),
    CardinalMeasurement("E", [.2600, .2320, .2035, .1640, .0740]),
    CardinalMeasurement("SE", [.2600, .2320, .2035, .1640, .0740]),
    CardinalMeasurement("S", [.2600, .2320, .2035, .1640, .0740]),
    CardinalMeasurement("SW", [.2600, .2320, .2035, .1640, .0740]),
    CardinalMeasurement("W", [.2600, .2320, .2035, .1640, .0740]),
    CardinalMeasurement("NW", [.2600, .2320, .2035, .1640, .0740]),
]

cardMeasList12 = [
    CardinalMeasurement("N", [.270, .2380, .2065, .1655, .0735]),
    CardinalMeasurement("NE", [.270, .2380, .2065, .1655, .0735]),
    CardinalMeasurement("E", [.270, .2380, .2065, .1655, .0735]),
    CardinalMeasurement("SE", [.270, .2380, .2065, .1655, .0735]),
    CardinalMeasurement("S", [.270, .2380, .2065, .1655, .0735]),
    CardinalMeasurement("SW", [.270, .2380, .2065, .1655, .0735]),
    CardinalMeasurement("W", [.270, .2380, .2065, .1655, .0735]),
    CardinalMeasurement("NW", [.270, .2380, .2065, .1655, .0735]),
]

# oct eng, first plate, not torqued

cardMeasList13 = [
    CardinalMeasurement("N", [.2750, .2210, .1785, .1335, .0525]),
    CardinalMeasurement("NE", [.2750, .2210, .1785, .1335, .0525]),
    CardinalMeasurement("E", [.2750, .2210, .1785, .1335, .0525]),
    CardinalMeasurement("SE", [.2750, .2210, .1785, .1335, .0525]),
    CardinalMeasurement("S", [.2750, .2210, .1785, .1335, .0525]),
    CardinalMeasurement("SW", [.2750, .2210, .1785, .1335, .0525]),
    CardinalMeasurement("W", [.2750, .2210, .1785, .1335, .0525]),
    CardinalMeasurement("NW", [.2750, .2210, .1785, .1335, .0525]),
]

# same plate, torqued

cardMeasList14 = [
    CardinalMeasurement("N", [.265, .217, .179, .135, .054]),
    CardinalMeasurement("NE", [.265, .217, .179, .135, .054]),
    CardinalMeasurement("E", [.265, .217, .179, .135, .054]),
    CardinalMeasurement("SE", [.265, .217, .179, .135, .054]),
    CardinalMeasurement("S", [.265, .217, .179, .135, .054]),
    CardinalMeasurement("SW", [.265, .217, .179, .135, .054]),
    CardinalMeasurement("W", [.265, .217, .179, .135, .054]),
    CardinalMeasurement("NW", [.265, .217, .179, .135, .054]),
]

# back off middle
cardMeasList15 = [
    CardinalMeasurement("N", [.260, .214, .177, .133 ,.054]),
    CardinalMeasurement("NE", [.260, .214, .177, .133 ,.054]),
    CardinalMeasurement("E", [.260, .214, .177, .133 ,.054]),
    CardinalMeasurement("SE", [.260, .214, .177, .133 ,.054]),
    CardinalMeasurement("S", [.260, .214, .177, .133 ,.054]),
    CardinalMeasurement("SW", [.260, .214, .177, .133 ,.054]),
    CardinalMeasurement("W", [.260, .214, .177, .133 ,.054]),
    CardinalMeasurement("NW", [.260, .214, .177, .133 ,.054]),
]

# replugged oct plate

cardMeasList15 = [
    CardinalMeasurement("N", [.27, .2220, .1825, .1370, .0550]),
    CardinalMeasurement("NE", [.27, .2220, .1825, .1370, .0550]),
    CardinalMeasurement("E", [.27, .2220, .1825, .1370, .0550]),
    CardinalMeasurement("SE", [.27, .2220, .1825, .1370, .0550]),
    CardinalMeasurement("S", [.27, .2220, .1825, .1370, .0550]),
    CardinalMeasurement("SW", [.27, .2220, .1825, .1370, .0550]),
    CardinalMeasurement("W", [.27, .2220, .1825, .1370, .0550]),
    CardinalMeasurement("NW", [.27, .2220, .1825, .1370, .0550]),
]


# new / black bending rings

# cardMeasList15 = [
#     CardinalMeasurement("N", [.1545, .1365, .1155, .0875, .0340]),
#     CardinalMeasurement("NE", [.1545, .1365, .1155, .0875, .0340]),
#     CardinalMeasurement("E", [.1545, .1365, .1155, .0875, .0340]),
#     CardinalMeasurement("SE", [.1545, .1365, .1155, .0875, .0340]),
#     CardinalMeasurement("S", [.1545, .1365, .1155, .0875, .0340]),
#     CardinalMeasurement("SW", [.1545, .1365, .1155, .0875, .0340]),
#     CardinalMeasurement("W", [.1545, .1365, .1155, .0875, .0340]),
#     CardinalMeasurement("NW", [.1545, .1365, .1155, .0875, .0340]),
# ]

cardMeasList15 = [
    CardinalMeasurement("N", [.2755, .2415, .2105, .1710, .079]),
    CardinalMeasurement("NE", [.2755, .2415, .2105, .1710, .079]),
    CardinalMeasurement("E", [.2755, .2415, .2105, .1710, .079]),
    CardinalMeasurement("SE", [.2755, .2415, .2105, .1710, .079]),
    CardinalMeasurement("S", [.2755, .2415, .2105, .1710, .079]),
    CardinalMeasurement("SW", [.2755, .2415, .2105, .1710, .079]),
    CardinalMeasurement("W", [.2755, .2415, .2105, .1710, .079]),
    CardinalMeasurement("NW", [.2755, .2415, .2105, .1710, .079]),
]


nicksnominals = [
    CardinalMeasurement("N", [.2202, .2007, .1733, .1343, .0542]),
    CardinalMeasurement("NE", [.2202, .2007, .1733, .1343, .0542]),
    CardinalMeasurement("E", [.2202, .2007, .1733, .1343, .0542]),
    CardinalMeasurement("SE", [.2202, .2007, .1733, .1343, .0542]),
    CardinalMeasurement("S", [.2202, .2007, .1733, .1343, .0542]),
    CardinalMeasurement("SW", [.2202, .2007, .1733, .1343, .0542]),
    CardinalMeasurement("W", [.2202, .2007, .1733, .1343, .0542]),
    CardinalMeasurement("NW", [.2202, .2007, .1733, .1343, .0542]),
]

november1 = [
    CardinalMeasurement("N", [.2530, .2285, .2030, .1675, .0790]),
    CardinalMeasurement("NE", [.2530, .2285, .2030, .1675, .0790]),
    CardinalMeasurement("E", [.2530, .2285, .2030, .1675, .0790]),
    CardinalMeasurement("SE", [.2530, .2285, .2030, .1675, .0790]),
    CardinalMeasurement("S", [.2530, .2285, .2030, .1675, .0790]),
    CardinalMeasurement("SW", [.2530, .2285, .2030, .1675, .0790]),
    CardinalMeasurement("W", [.2530, .2285, .2030, .1675, .0790]),
    CardinalMeasurement("NW", [.2530, .2285, .2030, .1675, .0790]),
]

obstest1 = [
    CardinalMeasurement("N", [.231, .2070, 0.1770, .1375, .0565]),
    CardinalMeasurement("NE", [.231, .2070, 0.1770, .1375, .0565]),
    CardinalMeasurement("E", [.231, .2070, 0.1770, .1375, .0565]),
    CardinalMeasurement("SE", [.231, .2070, 0.1770, .1375, .0565]),
    CardinalMeasurement("S", [.231, .2070, 0.1770, .1375, .0565]),
    CardinalMeasurement("SW", [.231, .2070, 0.1770, .1375, .0565]),
    CardinalMeasurement("W", [.231, .2070, 0.1770, .1375, .0565]),
    CardinalMeasurement("NW", [.231, .2070, 0.1770, .1375, .0565]),
]

obstest1_weird = [
    CardinalMeasurement("N", [.2285, .2045, .1740, .1350, .0555]),
    CardinalMeasurement("NE", [.2285, .2045, .1740, .1350, .0555]),
    CardinalMeasurement("E", [.2440, .2235, .2000, .1660, .0780]),
    CardinalMeasurement("SE", [.2440, .2235, .2000, .1660, .0780]),
    CardinalMeasurement("S", [.2175, .1870, .1625, .1315, .0615]),
    CardinalMeasurement("SW", [.2175, .1870, .1625, .1315, .0615]),
    CardinalMeasurement("W", [.2260, .1905, .1610, .1235, .0500]),
    CardinalMeasurement("NW", [.2260, .1905, .1610, .1235, .0500]),
]

obstest2 = [
    CardinalMeasurement("N", [.2500, .2220, .1925, .1535, .0675]),
    CardinalMeasurement("NE", [.2500, .2220, .1925, .1535, .0675]),
    CardinalMeasurement("E", [.2500, .2220, .1925, .1535, .0675]),
    CardinalMeasurement("SE", [.2500, .2220, .1925, .1535, .0675]),
    CardinalMeasurement("S", [.2500, .2220, .1925, .1535, .0675]),
    CardinalMeasurement("SW", [.2500, .2220, .1925, .1535, .0675]),
    CardinalMeasurement("W", [.2500, .2220, .1925, .1535, .0675]),
    CardinalMeasurement("NW", [.2500, .2220, .1925, .1535, .0675]),
]

obstest3 = [
    CardinalMeasurement("N", [.2500, .2210, .1945, .1585, .0735]),
    CardinalMeasurement("NE", [.2500, .2210, .1945, .1585, .0735]),
    CardinalMeasurement("E", [.2405, .2135, .1895, .1560, .0735]),
    CardinalMeasurement("SE", [.2405, .2135, .1895, .1560, .0735]),
    CardinalMeasurement("S", [.2570, .2295, .2, .1605, .0720]),
    CardinalMeasurement("SW", [.2570, .2295, .2, .1605, .0720]),
    CardinalMeasurement("W", [.2445, .2215, .1965, .1620, .0765]),
    CardinalMeasurement("NW", [.2445, .2215, .1965, .1620, .0765]),
]

december = [
    CardinalMeasurement("N", [.2500, .2180, .1890, .1515, .0685]),
    CardinalMeasurement("NE", [.2480, .2175, .1895, .1520, .0685]),
    CardinalMeasurement("E", [.2555, .2255, .1955, .1570, .0710]),
    CardinalMeasurement("SE", [.2620, .2325, .2020, .1620, .0720]),
    CardinalMeasurement("S", [.2535, .2270, .2000, .1635, .0765]),
    CardinalMeasurement("SW", [.2475, .2205, .945, .1600, .0760]),
    CardinalMeasurement("W", [.2500, .2185, .1905, .1545, .0715]),
    CardinalMeasurement("NW", [.2535, .2200, .1905, .1530, .0690]),
]

pf = PlateProfile(december, plate=9104)
dp = DuPontFocalProfile()
fpf = FocalPlaneFitter(dp, pf, plate=9104)
fpf.plotSurfErr()
# fpf.plot2Surf()

plt.show()

