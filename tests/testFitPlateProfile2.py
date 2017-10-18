from fitPlateProfile import DuPontMeasurement, DuPontProfile, plt#, MeasRadii, DirThetaMapDuPont
import numpy

# cardinalMeasurementList = []

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

# cardinalDirections = DirThetaMapDuPont.keys() # cardinal directions
# radialFloats = [6.3754,  5.7657,  5.0673,  4.1528,  1.955]

dpf = DuPontProfile()
dpf.addMeasList(december9)
dpf.addPlateID(9999)
dpf.doNewInterp()
xPos = numpy.arange(-200,200,10)
yPos = numpy.arange(-200,200,10)
for x in xPos:
    for y in yPos:
        print("%.2f, %.2f  err: %.2f"%(x, y, dpf.getErr(x,y)))
dpf.testInterp()
plt.show(block=True)
# import pdb; pdb.set_trace()