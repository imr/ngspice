/*************
 * Header file for display.c
 * 1999 E. Rouat
 ************/

/*  See if we have been already included  */
#ifndef DISPLAY_H_INCLUDED
#define DISPLAY_H_INCLUDED

/*  Include a bunch of other stuff to make display.h work  */
#include <ftedev.h>
#include <fteinput.h>
#include <graph.h>

DISPDEVICE *FindDev(char *name);
void DevInit(void);
int NewViewport(GRAPH *pgraph);
void DevClose(void);
void DevClear(void);
void DrawLine(int x1, int y1, int x2, int y2);
void Arc(int x0, int y0, int radius, double theta1, double theta2);
void Text(char *text, int x, int y);
void DefineColor(int colorid, double red, double green, double blue);
void DefineLinestyle(int linestyleid, int mask);
void SetLinestyle(int linestyleid);
void SetColor(int colorid);
void Update(void);
void DatatoScreen(GRAPH *graph, double x, double y, int *screenx, int *screeny);
void Input(REQUEST *request, RESPONSE *response);
void SaveText(GRAPH *graph, char *text, int x, int y);
int DevSwitch(char *devname);


#endif /* DISPLAY_H_INCLUDED */

