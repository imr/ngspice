#include "ngspice/ngspice.h"

#include "plotting.h"

/* Where 'constants' go when defined on initialization. */

struct plot constantplot = {
    "Constant values", Spice_Build_Date, "constants",
    "const", NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    TRUE, FALSE, 0, 0, 0
};

struct plot *plot_cur = &constantplot;
struct plot *plot_list = &constantplot;

int plotl_changed;      /* TRUE after a load */

int plot_num = 1;
