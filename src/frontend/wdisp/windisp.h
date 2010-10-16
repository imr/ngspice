/*************
* Header file for windisp.c
* $Id$
************/

#ifndef WINDISP_H
#define WINDISP_H

disp_fn_Init_t             WIN_Init;
disp_fn_NewViewport_t      WIN_NewViewport;
disp_fn_Close_t            WIN_Close;
disp_fn_Clear_t            WIN_Clear;
disp_fn_DrawLine_t         WIN_DrawLine;
disp_fn_Arc_t              WIN_Arc;
disp_fn_Text_t             WIN_Text;
disp_fn_DefineColor_t      WIN_DefineColor;
disp_fn_DefineLinestyle_t  WIN_DefineLinestyle;
disp_fn_SetLinestyle_t     WIN_SetLinestyle;
disp_fn_SetColor_t         WIN_SetColor;
disp_fn_Update_t           WIN_Update;

//extern int WIN_DiagramReady();

#endif /* WINDISP_H */
