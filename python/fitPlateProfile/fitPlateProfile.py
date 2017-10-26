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
from matplotlib import gridspec
from scipy.interpolate import griddata, LinearNDInterpolator

ErrorTolerance = [-0.2, 0.2] #range in mm in which the profile error (measured - focal plane) is acceptable
MMPerInch = 25.4
ThetaInterp = 100 #number of interpolation points in theta
RadialInterp = 100 #number of interpolation points in radius


DuPontFocalRadius = 8800 # mm

def plateSurfPlot(x,y,z):
    colormap = cm.hot
    paneColor = (0.25,0.25,0.25,0.25)
    xlim = [-300,300]
    ylim = [-300, 300]
    zlim = [-0.4,0.4]
    xticks = [-200, 0, 200]
    yticks = [-200, 0, 200]
    zticks = [-0.4, -0.2, 0, 0.2, 0.4]

    fig = plt.figure(figsize=(14,4))
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, subplot_kw=dict(projection="3d"), figsize=(11,3))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 5, 5, 5])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], projection="3d")
    ax3 = plt.subplot(gs[2], projection="3d")
    ax4 = plt.subplot(gs[3], projection="3d")

    # fig = plt.figure()
    # ax = fig.gca(projection="3d")
    ax4.plot_trisurf(x, y, z, cmap=colormap, vmin=ErrorTolerance[0], vmax=ErrorTolerance[1], edgecolor='none')
    ax4.view_init(90,-90)
    # ax1.set_zlabel("focal plane error (mm)")
    ax4.text(0, -300, 0, 'TAB', size=10, weight="bold", zorder=1, color='k', verticalalignment='center', horizontalalignment='center')
    ax4.set_xlabel("plate x (mm)")
    ax4.set_ylabel("plate y (mm)", labelpad=15)
    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim)
    ax4.set_zlim(zlim)
    ax4.set_zticks([])
    ax4.set_xticks(xticks)
    ax4.set_yticks(yticks)
    ax4.w_zaxis.line.set_lw(0.)
    ax4.w_xaxis.set_pane_color(paneColor)
    ax4.w_yaxis.set_pane_color(paneColor)
    ax4.w_zaxis.set_pane_color(paneColor)

    ax3.plot_trisurf(x, y, z, cmap=colormap, vmin=ErrorTolerance[0], vmax=ErrorTolerance[1], edgecolor="none")
    ax3.view_init(0,-90)
    ax3.set_xlabel("plate x (mm)")
    # ax3.set_zlabel("focal plane error (mm)")
    ax3.set_yticks([])
    ax3.set_xticks(xticks)
    ax3.set_zticks(zticks)
    ax3.w_yaxis.line.set_lw(0.)
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.set_zlim(zlim)
    ax3.w_xaxis.set_pane_color(paneColor)
    ax3.w_yaxis.set_pane_color(paneColor)
    ax3.w_zaxis.set_pane_color(paneColor)

    ax2.plot_trisurf(x, y, z, cmap=colormap, vmin=ErrorTolerance[0], vmax=ErrorTolerance[1], edgecolor="none")
    ax2.view_init(0,0)
    #ax2.set_xlabel("plate x (mm)")
    ax2.set_ylabel("plate y (mm)")
    ax2.set_zlabel("focal plane error (mm)")
    ax2.set_xticks([])
    ax2.set_yticks(yticks)
    ax2.w_xaxis.line.set_lw(0.)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_zlim(zlim)
    ax2.set_zticks(zticks)
    ax2.w_xaxis.set_pane_color(paneColor)
    ax2.w_yaxis.set_pane_color(paneColor)
    ax2.w_zaxis.set_pane_color(paneColor)

    plt.sca(ax1)
    ax1.axis("off")
    m = cm.ScalarMappable(cmap=colormap)
    m.set_array(ErrorTolerance)
    plt.colorbar(m, ticks=[-0.2, -0.1, 0, 0.1, 0.2])

class DuPontMeasurement(object):
    measRadii = numpy.asarray([0.755, 3.75, 5.75, 7.75, 10.75]) * MMPerInch # Nick's measurements
    dirThetaMap = OrderedDict((
        (1, 6. * numpy.pi / 4.), # tab
        (2, 5. * numpy.pi / 4.), # tab + 45 degrees cw
        (3, 4. * numpy.pi / 4.), # tab + 90 degrees cw
        (4, 3. * numpy.pi / 4.),
        (5, 2. * numpy.pi / 4.),
        (6, 1. * numpy.pi / 4.),
        (7, 0.0),
        (8, 7. * numpy.pi / 4.),
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

        self.duPontMeasList = None
        self.plateID = None
        self.fscanID = None
        self.fscanMJD = None
        self.profID = None
        self.percentInSpec = None
        self.cartID = None
        self.isActivePlugging = None

    def addMeasList(self, measList):
        measList.sort(key=lambda x: x.theta)
        self.duPontMeasList = measList

    def addPlateID(self, plateID):
        self.plateID = plateID

    def addFscanID(self, fscanID):
        self.fscanID = fscanID

    def addFscanMJD(self, fscanMJD):
        self.fscanMJD = fscanMJD

    def addProfID(self, profID):
        self.profID = profID

    def addCartID(self, cartID):
        self.cartID = cartID


    def doNewInterp(self):
        # todo, break this up into understandable pieces?
        if self.duPontMeasList is None:
            raise RuntimeError("Empty Measurement List")
        if self.plateID is None:
            raise RuntimeError("Unknown plateID")
        measList = self.duPontMeasList
        # http://stackoverflow.com/questions/22653956/using-scipy-spatial-delaunay-in-place-of-matplotlib-tri-triangulations-built-in
        # get x,y positions for r, thetas
        measList.sort(key=lambda x: x.theta) # should already be sorted but whatever
        # sort by theta (0-2pi)
        rawRadii = measList[0].measRadii # all radii should be the same
        rawThetas = numpy.array([cc.theta for cc in measList] + [2*numpy.pi])
        rawMeas = numpy.array([cc.measList for cc in measList] + [measList[0].measList])
        # raw meas is 2D array ra
        thetaInterp = numpy.linspace(rawThetas[0], rawThetas[-1], ThetaInterp)
        radiiInterp = numpy.linspace(rawRadii[0], rawRadii[-1], RadialInterp)
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
        self.xInterp = numpy.array(xInterp).flatten()
        self.yInterp = numpy.array(yInterp).flatten()
        self.measInterp = numpy.array(measInterp).flatten()
        self.model = numpy.array(model).flatten()
        err = self.model - self.measInterp
        self.err = err - numpy.mean(err) # maybe use median?
        errorUnits = numpy.asarray(errorUnits)
        errorUnits = errorUnits - numpy.mean(errorUnits)
        areaUnits = numpy.asarray(areaUnits)
        #determine out of specness
        errorUnits = numpy.abs(errorUnits)
        outOfSpecInds = numpy.argwhere(errorUnits>0.2)
        areaOutOfSpec = numpy.sum(areaUnits[outOfSpecInds])
        totalArea = numpy.sum(areaUnits)
        self.percentInSpec = 100 - areaOutOfSpec/totalArea * 100
        print("percent of plate in spec - %.2f"%self.percentInSpec)
        self.ndInterp = LinearNDInterpolator(numpy.array([self.xInterp, self.yInterp]).T, self.err)

    def plot(self):
        plateSurfPlot(self.xInterp, self.yInterp, self.err)
        f = plt.gcf()
        f.suptitle("Plate %i:  %.0f%% within specifications ($\pm$ 0.2 mm)"%(self.plateID, self.percentInSpec))


    def _getErr(self, xPos, yPos):
        # return the (estimated) focal plane error in mm for a given xyPos on the plate (mm)
        dist = numpy.sqrt((xPos - self.xInterp)**2+(yPos - self.yInterp)**2)
        minDistInd = numpy.argmin(dist)
        minX = self.xInterp[minDistInd]
        minY = self.yInterp[minDistInd]
        error = self.err[minDistInd]
        # print("x err: %.2f, y err: %.2f, error: %.2f"%(minX-xPos, minY-yPos, error))
        return error

    def getErr(self, xPos, yPos):
        # use linear interpolation
        return self.ndInterp(numpy.array([xPos,yPos]))[0]

    def testInterp(self):
        xArr = numpy.linspace(-300,300,250)
        yArr = numpy.linspace(-300,300,250)
        xVals = []
        yVals = []
        errVals = []
        for x in xArr:
            for y in yArr:
                xVals.append(x)
                yVals.append(y)
                err = self.getErr(x,y)[0]
                errVals.append(err)
        xVals = numpy.asarray(xVals)
        yVals = numpy.asarray(yVals)
        errVals = numpy.asarray(errVals)
        plateSurfPlot(xVals, yVals, errVals)

    def getMeasureDict(self):
        """List of CardinalMeasurement objects
        in order of directionList
        """
        outDict = OrderedDict()
        for direction, duPontMeas in itertools.izip(DuPontMeasurement.dirThetaMap.keys(), self.duPontMeasList):
            outDict[direction] = duPontMeas.measList
        # logMsg(outDict)
        return outDict


    def addProfileToDB(self, comment=None):
        """Write the profilometry information in profilometryDict to the db.

        @param[in] cartID, for getting the active plugging for this cart.
        @param[in] List of CardinalMeasurement objects in order of directionList N, NE, ...
        @param[in] comment: a comment associated with this profilometry.

        return MJD, scanID for saving profile images alongside mapper data
        """
        #DBSession.query(PlPlugMapM).filter(PlPlugMapM.filename==kwargs['plplugmap']).one()
        from sdss.internal.database.connections import LCODatabaseAdminLocalConnection
        from sdss.internal.database.apo.platedb import ModelClasses as plateDB
        if self.plateID is None:
            raise RuntimeError("Unknown Plate ID")
        if self.duPontMeasList is None:
            raise RuntimeError("No Measurements!")
        session = plateDB.db.Session()
        try:
            plugging = session.query(plateDB.Plugging).join(plateDB.ActivePlugging).join(plateDB.Plate).filter(plateDB.Plate.plate_id==self.plateID).one()
        except Exception as e:
            print(str(e))
            raise RuntimeError("No active plugging found for plateID: %i!  Was it mapped?"%self.plateID)
        profilometryDict = self.getMeasureDict()
        profilometry = plateDB.Profilometry()
        #LCO hack just pick the first tolerance
        tolerances = session.query(plateDB.ProfilometryTolerances).all()[0]
        # tolerances = session.query(plateDB.ProfilometryTolerances).join(plateDB.Survey)\
        #                 .filter(plateDB.ProfilometryTolerances.survey == self.plate.surveys[0])\
        #                 .limit(1).one()

        # loop over measurement numbers (directions)
        for ii, measList in enumerate(profilometryDict.itervalues()):
            pm = plateDB.ProfilometryMeasurement()
            pm.number = ii+1
            pm.r1 = measList[0]
            pm.r2 = measList[1]
            pm.r3 = measList[2]
            pm.r4 = measList[3]
            pm.r5 = measList[4]
            profilometry.measurements.append(pm)

        if comment:
            profilometry.comment = comment

        # get the active plugging of this plate,
        # the mapper sets the active plugging.
        # associate the profilometry with it
        profilometry.tolerances = tolerances
        profilometry.plugging = plugging
        with session.begin():
            session.add(profilometry)
        return plugging.fscan_mjd, plugging.fscan_id


    def getProfileFromDB(self, plateID=None):
        from sdss.internal.database.connections import LCODatabaseAdminLocalConnection
        from sdss.internal.database.apo.platedb import ModelClasses as plateDB
        # for now just grab the last profile for this plate
        if plateID is None:
            if self.plateID is None:
                raise RuntimeError("Unknown Plate ID")
            else:
                plateID = self.plateID
        elif self.plateID is not None:
            raise RuntimeError("Conflicting plate IDs!")
        self.plateID = plateID

        session = plateDB.db.Session()
        profs = session.query(plateDB.Profilometry).join(plateDB.Plugging).join(plateDB.Plate).filter(plateDB.Plate.plate_id == plateID).all()
        profs.sort(key=lambda x: x.timestamp)
        lastProf = profs[-1]
        duPontMeasList = []
        # meas.number is 0-2pi (which is differend from the dir theta map directions which begin at -pi/4)
        # pluggers measure from tab and proceed clockwise.
        # values are stored in db beginning at 0 and proceeding counter clockwise because the
        # dupont measurements are sorted by theta before entering them in the db.
        # an unfortuante mess but be sure to unreavel these things correctly.
        measNum2DirMap = {
            1 : 7,
            2 : 6,
            3 : 5,
            4 : 4,
            5 : 3,
            6 : 2,
            7 : 1,
            8 : 8,
        }
        for meas in lastProf.measurements:
            rList = [meas.r1, meas.r2, meas.r3, meas.r4, meas.r5]
            rList = [float(x) for x in rList]
            measNum = int(meas.number)
            direction = measNum2DirMap[measNum]
            duPontMeasList.append(DuPontMeasurement(direction, rList))
        self.addMeasList(duPontMeasList)
        self.doNewInterp()







