/*************
 * Header file for x11.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_X11_H
#define ngspice_X11_H

#ifndef X_DISPLAY_MISSING

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

int X11_Input(REQUEST *request, RESPONSE *response);

#endif

#endif
