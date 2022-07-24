#ifndef ngspice_GRID_H
#define ngspice_GRID_H

#include "typedefs.h"

/* Grid types. 

   Note: SMITHGRID is only a smith grid, SMITH transforms the data */
typedef enum {
    GRID_NONE, GRID_LIN, GRID_LOGLOG, GRID_XLOG, GRID_YLOG,
    GRID_POLAR, GRID_SMITH, GRID_SMITHGRID, GRID_DIGITAL_NONE,
    GRID_DIGITAL
} GRIDTYPE;

void gr_fixgrid(GRAPH *graph, double xdelta, double ydelta, int xtype, int ytype);
void gr_redrawgrid(GRAPH *graph);

#endif
