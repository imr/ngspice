#ifndef _GRID_H
#define _GRID_H

/* Grid types. 

   Note: SMITHGRID is only a smith grid, SMITH transforms the data */
typedef enum {
    GRID_NONE, GRID_LIN, GRID_LOGLOG, GRID_XLOG,
    GRID_YLOG, GRID_POLAR, GRID_SMITH, GRID_SMITHGRID
} GRIDTYPE;

#endif
