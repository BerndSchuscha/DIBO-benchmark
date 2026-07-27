from pibob.problems.BraninCurrin.registry import make_problem as make_branin_currin
from pibob.problems.Ms_NiTi.registry import make_problem as make_Ms_NiTi
from pibob.problems.NanoParticles.registry import make_problem as make_NanoParticles
from pibob.problems.EnergyDensity.registry import make_problem as make_EnergyDensity
from pibob.problems.PS_ALSC.registry import make_problem as make_PS_ALSC
from pibob.problems.Peroskites.registry import make_problem as make_Peroskietes
from pibob.problems.Structures.registry import make_problem as make_Structures
from pibob.problems.Udimet.registry import make_problem as make_Udimet
from pibob.problems.Magnetic.registry import make_problem as make_Magnetic
from pibob.problems.Bainite.registry import make_problem as make_Bainite

PROBLEMS = {
    "branin_currin": make_branin_currin,
    "Ms_NiTi_Problem": make_Ms_NiTi,
    "NanoParticles": make_NanoParticles,
    "EnergyDensity_Problem": make_EnergyDensity,
    "PS_ALSC_Problem": make_PS_ALSC,
    "Peroskites_Problem": make_Peroskietes,
    "Udimet_Problem": make_Udimet,
    "Bainite_Problem": make_Bainite,
    "Peroskites_Problem": make_Peroskietes,
    "Structures_Problem": make_Structures,
    "Magnetic_Problem": make_Magnetic,
}
