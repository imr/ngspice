/*************
 * Header file for grid.c
 * 1999 E. Rouat
 ************/

#ifndef GRID_H_INCLUDED
#define GRID_H_INCLUDED

typedef enum { x_axis, y_axis } Axis;

void gr_fixgrid(GRAPH *graph, double xdelta, double ydelta, int xtype, int ytype);
void gr_redrawgrid(GRAPH *graph);
void drawlingrid(GRAPH *graph, char *units, int spacing, int nsp, double dst, double lmt, 
		 double hmt, bool onedec, int mult, double mag, int digits, Axis axis);
void drawloggrid(GRAPH *graph, char *units, int hmt, int lmt, int decsp, int subs, 
		 int pp, Axis axis);

#endif
