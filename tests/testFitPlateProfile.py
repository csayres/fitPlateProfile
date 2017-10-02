from fitPlateProfile import CardinalMeasurement, DirThetaMapCard, doNewInterp, plt, MeasRadii

cardinalMeasurementList = []

december9 = [
    CardinalMeasurement("N", [.2400, .2110, .1820, .1450, .0640]),
    CardinalMeasurement("NE", [.2295, .2040, .1790, .1455, .0665]),
    CardinalMeasurement("E", [.2345, .2090, .1840, .1495, .0685]),
    CardinalMeasurement("SE", [.2455, .2200, .1925, .1545, .0685]),
    CardinalMeasurement("S", [.2425, .2185, .1920, .1550, .0700]),
    CardinalMeasurement("SW", [.2310, .2095, .1855, .1520, .0710]),
    CardinalMeasurement("W", [.2345, .2110, .1860, .1510, .0690]),
    CardinalMeasurement("NW", [.2450, .2175, .1885, .1495, .0650]),
]

cardinalDirections = DirThetaMapCard.keys() # cardinal directions
radialFloats = [6.3754,  5.7657,  5.0673,  4.1528,  1.955]
for measDir in cardinalDirections:
    cardinalMeasurementList.append(CardinalMeasurement(measDir, radialFloats, toMM=False))

doNewInterp(december9, MeasRadii)
plt.show(block=True)
# import pdb; pdb.set_trace()