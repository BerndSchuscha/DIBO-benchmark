from pathlib import Path

import numpy as np
from kawin.precipitation import MatrixParameters, PrecipitateParameters
from kawin.precipitation.KWNEuler import PrecipitateModel
from kawin.precipitation.parameters.ElasticFactors import StrainEnergy
from kawin.thermo import BinaryThermodynamics

import pibob


def precipitator(x):  # temp in K, time in s

    def ar3(r):
        a = 18.4012307
        b = 51.3197329
        R = r * 1e9
        return a * R / (b - R)

    Ms_temp = np.zeros(np.shape(x)[0])
    mipd = np.zeros(np.shape(x)[0])
    bFrac = np.zeros(np.shape(x)[0])

    for k in range(np.shape(x)[0]):

        comp = x[k, 0]
        temp = x[k, 1]
        time = x[k, 2]

        # inputs
        xInit = comp
        T = temp

        # physical properties
        gamma = 0.053
        Dni = lambda T: 1.8e-8 * np.exp(-155000 / (8.314 * T))

        path = Path(pibob.PROJECT_ROOT, "pibob", "problems", "Ms_NiTi", "NiTi_SMA.tdb")
        # initializing thermodynamics model
        therm = BinaryThermodynamics(str(path), ["TI", "NI"], ["BCC_B2", "TI3NI4"],    interfacialCompMethod='equilibrium',drivingForceMethod='approximate')
        therm.setGuessComposition(0.56)
        therm.setDiffusivity(Dni, "BCC_B2")
        therm.setDiffusivity(Dni, "TI3NI4")
        

        Vaalpha = 0.02681144066 * 1e-27
        nalpha = 2
        matrix = MatrixParameters(solutes=["NI"])
        matrix.initComposition = xInit  # Mole fraction
        matrix.volume.setVolume(Vaalpha, "VA", nalpha)

        gamma = 0.053
        Vabeta = 0.184614835 * 1e-27
        nbeta = 14

        precipitate = PrecipitateParameters("TI3NI4")
        precipitate.gamma = gamma  # J/m2
        precipitate.volume.setVolume(Vabeta, "VA", nbeta)
        precipitate.nucleation.setNucleationType("bulk")
        precipitate.calculateAspectRatio = False

        B2e = np.asarray([175, 45, 35]) * 1e9
        eigenstrain = [-0.00417, -0.00417, -0.0257]
        rotate = [
            [-4 / np.sqrt(42), 5 / np.sqrt(42), -1 / np.sqrt(42)],
            [-2 / np.sqrt(14), -1 / np.sqrt(14), 3 / np.sqrt(14)],
            [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
        ]

        # initializing strain energy model
        se = StrainEnergy()
        se.setEllipsoidal()
        se.setEigenstrain(eigenstrain)
        se.setElasticConstants(B2e[0], B2e[1], B2e[2])
        se.setRotationMatrix(rotate)

        precipitate.strainEnergy = se
        precipitate.shapeFactor.setPrecipitateShape(precipitateShape="plate", ar=ar3)

        # solving model
        model = PrecipitateModel(
            matrix, precipitate, thermodynamics=therm, temperature=T
        )
        model.currentTime = 1e-6
        model.solve(time, verbose=True, vIt=10000)

        mComp = model.data.composition[-1]  # final matrix composition

        # fitting parameters for transformation temperature equation (these are for Ms only)
        A = 4511.2373
        B = -83.32325
        C = -0.04753
        D = 204.86781

        Ms_temp[k] = (
            A + (B * mComp) * 100 + C * D ** ((mComp * 100) - 50)
        )  # martensitic start temperature in K
        bFrac[k] = model.data.volFrac[-1]
        nDens = model.data.precipitateDensity[-1]  # number of precipitates/m^3

        if nDens > 0:
            mipd[k] = np.cbrt(
                3 / (4 * np.pi * nDens)
            )  # mean inter-particle distance in meters
        else:
            mipd[k] = (
                1000  # artificially high number to indicate no precipitates form, may need to be changed
            )
        print("xInit:", xInit)
        print("mComp final:", model.data.composition[-1])

        # if these exist in your KaWiN version:
        for name in [
            "drivingForce",
            "chemDrivingForce",
            "strainEnergy",
            "eqMatrixComposition",
        ]:
            if hasattr(model.data, name):
                arr = getattr(model.data, name)
                print(name, arr[0], "->", arr[-1])

    return Ms_temp, mipd, bFrac
