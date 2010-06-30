/*************
* Header file for winprint.c
* $Id$
************/

#ifndef WINPRINT_H
#define WINPRINT_H

disp_fn_Init_t             WPRINT_Init;
disp_fn_NewViewport_t      WPRINT_NewViewport;
disp_fn_Close_t            WPRINT_Close;
disp_fn_Clear_t            WPRINT_Clear;
disp_fn_DrawLine_t         WPRINT_DrawLine;
disp_fn_Arc_t              WPRINT_Arc;
disp_fn_Text_t             WPRINT_Text;
disp_fn_DefineColor_t      WPRINT_DefineColor;
disp_fn_DefineLinestyle_t  WPRINT_DefineLinestyle;
disp_fn_SetLinestyle_t     WPRINT_SetLinestyle;
disp_fn_SetColor_t         WPRINT_SetColor;
disp_fn_Update_t           WPRINT_Update;

extern int WPRINT_DiagramReady();

#endif /* WINPRINT_H */
