/*************
 * Header file for plot5.c
 * 1999 E. Rouat
 * $Id$
 ************/

#ifndef PLOT5_H_INCLUDED
#define PLOT5_H_INCLUDED

int Plt5_Init(void);
int Plt5_NewViewport(GRAPH *graph);
int Plt5_Close(void);
int Plt5_Clear(void);
int Plt5_DrawLine(int x1, int y1, int x2, int y2);
int Plt5_Arc(int x0, int y0, int radius, double theta1, double theta2);
int Plt5_Text(char *text, int x, int y);
int Plt5_SetLinestyle(int linestyleid);
int Plt5_SetColor(int colorid);
int Plt5_Update(void);

#endif /* PLOT5_H_INCLUDED */
