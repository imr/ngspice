#include <ngspice.h>

#include "plotting.h"

/* Where 'constants' go when defined on initialization. */

struct plot constantplot = {
    "Constant values", "Sat Aug 16 10:55:15 PDT 1986", "constants",
    "const", NULL, NULL, NULL, NULL, NULL, NULL, TRUE
} ;

struct plot *plot_cur = &constantplot;
struct plot *plot_list = &constantplot;
int plotl_changed;      /* TRUE after a load */

int plot_num = 1;
