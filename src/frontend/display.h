/*************
 * Header file for display.c
 * 1999 E. Rouat
 ************/

/*  See if we have been already included  */
#ifndef ngspice_DISPLAY_H
#define ngspice_DISPLAY_H

/*  Include a bunch of other stuff to make display.h work  */
#include "ngspice/ftedev.h"
#include "ngspice/fteinput.h"
#include "ngspice/graph.h"

DISPDEVICE *FindDev(char *name);
void DevInit(void);
int NewViewport(GRAPH *pgraph);
void DevClose(void);
void DevClear(void);
void DevDrawLine(int x1, int y1, int x2, int y2, bool isgrid);
void DevDrawArc(int x0, int y0, int radius, double theta, double delta_theta, bool isgrid);
void DevDrawText(const char *text, int x, int y, int angle);
void DefineColor(int colorid, double red, double green, double blue);
void DefineLinestyle(int linestyleid, int mask);
void SetLinestyle(int linestyleid);
void SetColor(int colorid);
void DevUpdate(void);
void DevFinalize(void);
void DatatoScreen(GRAPH *graph, double x, double y, int *screenx, int *screeny);
void Input(REQUEST *request, RESPONSE *response);
void SaveText(GRAPH *graph, char *text, int x, int y);
int DevSwitch(char *devname);


#endif

