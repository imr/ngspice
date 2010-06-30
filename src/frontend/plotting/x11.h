/*************
 * Header file for x11.c
 * 1999 E. Rouat
 * $Id$
 ************/

#ifndef X11_H_INCLUDED
#define X11_H_INCLUDED

#ifndef X_DISPLAY_MISSING

#include <X11/Intrinsic.h>	/* required for Widget */

disp_fn_Init_t             X11_Init;
disp_fn_NewViewport_t      X11_NewViewport;
disp_fn_Close_t            X11_Close;
disp_fn_DrawLine_t         X11_DrawLine;
disp_fn_Arc_t              X11_Arc;
disp_fn_Text_t             X11_Text;
disp_fn_DefineColor_t      X11_DefineColor;
disp_fn_DefineLinestyle_t  X11_DefineLinestyle;
disp_fn_SetLinestyle_t     X11_SetLinestyle;
disp_fn_SetColor_t         X11_SetColor;
disp_fn_Update_t           X11_Update;
disp_fn_Clear_t            X11_Clear;

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
