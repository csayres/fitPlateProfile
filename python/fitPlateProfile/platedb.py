
from collections import OrderedDict
import itertools

from sdss.internal.database.connections import LCODatabaseAdminTunnelConnection
from sdss.internal.database.apo.platedb import ModelClasses as plateDB

from fitPlateProfile import DirThetaMapCard

def getMeasureDict(cardMeasList):
    """List of CardinalMeasurement objects
    in order of directionList
    """
    outDict = OrderedDict()
    for direction, cardMeas in itertools.izip(DirThetaMapCard.keys(), cardMeasList):
        outDict[direction] = cardMeas.measList
    # logMsg(outDict)
    return outDict

def addProfilometry(plateID, cardMeasList, comment=None):
    """Write the profilometry information in profilometryDict to the db.

    @param[in] cartID, for getting the active plugging for this cart.
    @param[in] List of CardinalMeasurement objects in order of directionList N, NE, ...
    @param[in] comment: a comment associated with this profilometry.

    return MJD, scanID for saving profile images alongside mapper data
    """
    #DBSession.query(PlPlugMapM).filter(PlPlugMapM.filename==kwargs['plplugmap']).one()
    session = plateDB.db.Session()
    try:
        plugging = session.query(plateDB.Plugging).join(plateDB.ActivePlugging).join(plateDB.Plate).filter(plateDB.Plate.plate_id==plateID).one()
    except:
        raise RuntimeError("No active plugging found for plateID: %i!  Was it mapped?"%plateID)
    profilometryDict = getMeasureDict(cardMeasList)
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