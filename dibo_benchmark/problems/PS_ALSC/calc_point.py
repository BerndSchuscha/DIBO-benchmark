import numpy as np
from kawin.thermo import BinaryThermodynamics
from kawin.precipitation import MatrixParameters, PrecipitateParameters
from kawin.precipitation.coupling import StrengthModel, DislocationParameters
from kawin.precipitation.coupling import (
    CoherencyContribution,
    ModulusContribution,
    APBContribution,
    InterfacialContribution,
)

import numpy as np
import matplotlib.pyplot as plt

from kawin.precipitation.PrecipitationParameters import TemperatureParameters

from kawin.precipitation import PrecipitateModel
from kawin.solver import explicitEulerIterator
import warnings

warnings.filterwarnings("ignore")


def calc_points(X):
    T1 = X[0]
    T2 = X[1]
    t1 = X[2]
    t2 = X[3]
    c = X[4]

    therm = BinaryThermodynamics("AlScZr.tdb", ["AL", "SC"], ["FCC_A1", "AL3SC"])
    therm.setGuessComposition(0.24)
    diff = lambda T: 1.9e-4 * np.exp(-164000 / (8.314 * T))
    therm.setDiffusivity(diff, "FCC_A1")

    matrix = MatrixParameters(["SC"])
    matrix.initComposition = c
    matrix.volume.setVolume((0.409e-9) ** 3, "VA", 4)

    precipitate = PrecipitateParameters("AL3SC")
    precipitate.gamma = 0.1
    precipitate.volume.setVolume((0.4196e-9) ** 3, "VA", 4)
    precipitate.nucleation.setNucleationType("bulk")

    dislocations = DislocationParameters(G=25.4e9, b=0.286e-9, nu=0.34)
    coherency = CoherencyContribution(eps=2 / 3 * 0.0125)
    modulus = ModulusContribution(Gp=67.9e9)
    apb = APBContribution(yAPB=0.5)
    interfacial = InterfacialContribution(gamma=0.1)

    sm = StrengthModel(
        precipitate, [coherency, modulus, apb, interfacial], dislocations
    )

    temperature = TemperatureParameters(
        [0, t1, t1 + 1 / 60, 21], [T1 + 273.15, T1 + 273.15, T2 + 273.15, T2 + 273.15]
    )

    model = PrecipitateModel(matrix, precipitate, therm, temperature)
    model.addCouplingModel(sm)
    model.currentTime = 1
    model.solve(
        (t1 + t2) * 3600,
        iterator=explicitEulerIterator,
        verbose=False,
        vIt=1000,
        minDtFrac=1e-5,
    )

    sigma = sm.totalStrength(model, returnContributions=False)[-1] / 1e6

    c_frac = model.data.volFrac[-1][0]

    return sigma, t1 + t2, c_frac
