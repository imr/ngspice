/*************
 * Header file for x11.c
 * 1999 E. Rouat
 * $Id$
 ************/

#ifndef X11_H_INCLUDED
#define X11_H_INCLUDED

#ifndef X_DISPLAY_MISSING

#include <X11/Intrinsic.h>	/* required for Widget */

int X11_Init(void);
int X11_NewViewport(GRAPH *graph);
int X11_Close(void);
int X11_DrawLine(int x1, int y1, int x2, int y2);
int X11_Arc(int x0, int y0, int radius, double theta1, double theta2);
int X11_Text(char *text, int x, int y);
int X11_DefineColor(int colorid, double red, double green, double blue);
int X11_DefineLinestyle(int linestyleid, int mask);
int X11_SetLinestyle(int linestyleid);
int X11_SetColor(int colorid);
int X11_Update(void);
int X11_Clear(void);
void handlekeypressed(Widget w, caddr_t clientdata, caddr_t calldata);
void handlebuttonev(Widget w, caddr_t clientdata, caddr_t calldata);
void slopelocation(GRAPH *graph, int x0, int y0);
void zoomin(GRAPH *graph);
void hardcopy(Widget w, caddr_t client_data, caddr_t call_data);
void killwin(Widget w, caddr_t client_data, caddr_t call_data);
void redraw(Widget w, caddr_t client_data, caddr_t call_data);
void resize(Widget w, caddr_t client_data, caddr_t call_data);
int X11_Input(REQUEST *request, RESPONSE *response);

#endif /* X_DISPLAY_MISSING */

#endif /* X11_H_INCLUDED */
