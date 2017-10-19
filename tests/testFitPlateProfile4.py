from fitPlateProfile import DuPontMeasurement, DuPontProfile, plt#, MeasRadii, DirThetaMapDuPont


dpf = DuPontProfile()
dpf.getProfileFromDB(10004)
dpf.plot()
plt.show(block=True)
# import pdb; pdb.set_trace()
