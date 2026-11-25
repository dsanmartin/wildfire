# CSIRO grassland fires F19 and C064

Measured data used by the FDS validation suite
([firemodels/exp, CSIRO_Grassland_Fires](https://github.com/firemodels/exp/tree/master/CSIRO_Grassland_Fires)),
converted to plain CSV:

| File | Content |
|---|---|
| `F19_front.csv`, `C064_front.csv` | head-fire position `t [s], x [m]` (for `firefront.py --obs`) |
| `F19_perimeters.csv`, `C064_perimeters.csv` | measured fire perimeters, one point per row: `t [s], x [m], y [m]` |

Coordinates as in the FDS cases (`Case_F19.fds`, `Case_C064.fds`): x downwind from the ignition line, y
across it, t from the start of ignition. The FDS input files themselves are in
[firemodels/fds, Validation/CSIRO_Grassland_Fires](https://github.com/firemodels/fds/tree/master/Validation/CSIRO_Grassland_Fires/FDS_Input_Files).
