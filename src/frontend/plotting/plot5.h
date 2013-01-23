/*************
 * Header file for plot5.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_PLOT5_H
#define ngspice_PLOT5_H

disp_fn_Init_t             Plt5_Init;
disp_fn_NewViewport_t      Plt5_NewViewport;
disp_fn_Close_t            Plt5_Close;
disp_fn_Clear_t            Plt5_Clear;
disp_fn_DrawLine_t         Plt5_DrawLine;
disp_fn_Arc_t              Plt5_Arc;
disp_fn_Text_t             Plt5_Text;
disp_fn_SetLinestyle_t     Plt5_SetLinestyle;
disp_fn_SetColor_t         Plt5_SetColor;
disp_fn_Update_t           Plt5_Update;

#endif
