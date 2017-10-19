from fitPlateProfile.fitPlateProfile_old import CardinalMeasurement, doNewInterp, plt#, MeasRadii, DirThetaMapDuPont
import numpy

# cardinalMeasurementList = []

december9 = [
    CardinalMeasurement("N", list(numpy.asarray([.2400, .2110, .1820, .1450, .0640]))),
    CardinalMeasurement("NE", list(numpy.asarray([.2295, .2040, .1790, .1455, .0665]))),
    CardinalMeasurement("E", list(numpy.asarray([.2345, .2090, .1840, .1495, .0685]))),
    CardinalMeasurement("SE", list(numpy.asarray([.2455, .2200, .1925, .1545, .0685]))),
    CardinalMeasurement("S", list(numpy.asarray([.2425, .2185, .1920, .1550, .0700]))),
    CardinalMeasurement("SW", list(numpy.asarray([.2310, .2095, .1855, .1520, .0710]))),
    CardinalMeasurement("W", list(numpy.asarray([.2345, .2110, .1860, .1510, .0690]))),
    CardinalMeasurement("NW", list(numpy.asarray([.2450, .2175, .1885, .1495, .0650]))),
]

# cardinalDirections = DirThetaMapDuPont.keys() # cardinal directions
# radialFloats = [6.3754,  5.7657,  5.0673,  4.1528,  1.955]

radInterpList1 = numpy.asarray(doNewInterp(december9)).flatten()

from fitPlateProfile import DuPontMeasurement, doNewInterp, plt#, MeasRadii, DirThetaMapDuPont

december9 = [
    DuPontMeasurement(1, list(numpy.asarray([.2400, .2110, .1820, .1450, .0640])*25.4)),
    DuPontMeasurement(2, list(numpy.asarray([.2295, .2040, .1790, .1455, .0665])*25.4)),
    DuPontMeasurement(3, list(numpy.asarray([.2345, .2090, .1840, .1495, .0685])*25.4)),
    DuPontMeasurement(4, list(numpy.asarray([.2455, .2200, .1925, .1545, .0685])*25.4)),
    DuPontMeasurement(5, list(numpy.asarray([.2425, .2185, .1920, .1550, .0700])*25.4)),
    DuPontMeasurement(6, list(numpy.asarray([.2310, .2095, .1855, .1520, .0710])*25.4)),
    DuPontMeasurement(7, list(numpy.asarray([.2345, .2110, .1860, .1510, .0690])*25.4)),
    DuPontMeasurement(8, list(numpy.asarray([.2450, .2175, .1885, .1495, .0650])*25.4)),
]

december9 = [
    DuPontMeasurement(3, list(numpy.asarray([.2400, .2110, .1820, .1450, .0640])*25.4)),
    DuPontMeasurement(2, list(numpy.asarray([.2295, .2040, .1790, .1455, .0665])*25.4)),
    DuPontMeasurement(1, list(numpy.asarray([.2345, .2090, .1840, .1495, .0685])*25.4)),
    DuPontMeasurement(8, list(numpy.asarray([.2455, .2200, .1925, .1545, .0685])*25.4)),
    DuPontMeasurement(7, list(numpy.asarray([.2425, .2185, .1920, .1550, .0700])*25.4)),
    DuPontMeasurement(6, list(numpy.asarray([.2310, .2095, .1855, .1520, .0710])*25.4)),
    DuPontMeasurement(5, list(numpy.asarray([.2345, .2110, .1860, .1510, .0690])*25.4)),
    DuPontMeasurement(4, list(numpy.asarray([.2450, .2175, .1885, .1495, .0650])*25.4)),
]

# cardinalDirections = DirThetaMapDuPont.keys() # cardinal directions
# radialFloats = [6.3754,  5.7657,  5.0673,  4.1528,  1.955]

radInterpList2 = numpy.asarray(doNewInterp(december9)).flatten()
print(radInterpList2-radInterpList1)

plt.show(block=True)
# import pdb; pdb.set_trace()

"""
gauge 1: 0.2530
gauge 2: 0.2275
gauge 3: 0.1980
gauge 4: 0.1610
gauge 5: 0.0750
"""