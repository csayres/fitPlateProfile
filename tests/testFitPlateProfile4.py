from fitPlateProfile import DuPontMeasurement, DuPontProfile, plt#, MeasRadii, DirThetaMapDuPont


dpf = DuPontProfile()
dpf.getProfileFromDB(9990)
dpf.plot()
plt.show(block=True)
# import pdb; pdb.set_trace()
