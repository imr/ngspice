/*************
 * Header file for plot5.c
 * 1999 E. Rouat
 ************/

#ifndef PLOT5_H_INCLUDED
#define PLOT5_H_INCLUDED

int Plt5_Init(void);
int Plt5_NewViewport(GRAPH *graph);
void Plt5_Close(void);
void Plt5_Clear(void);
void Plt5_DrawLine(int x1, int y1, int x2, int y2);
void Plt5_Arc(int x0, int y0, int radius, double theta1, double theta2);
void Plt5_Text(char *text, int x, int y);
int Plt5_SetLinestyle(int linestyleid);
void Plt5_SetColor(int colorid);
void Plt5_Update(void);

#endif
