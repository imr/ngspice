/*************
 * Header file for postsc.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_POSTSC_H
#define ngspice_POSTSC_H

disp_fn_Init_t             PS_Init;
disp_fn_NewViewport_t      PS_NewViewport;
disp_fn_Close_t            PS_Close;
disp_fn_Clear_t            PS_Clear;
disp_fn_DrawLine_t         PS_DrawLine;
disp_fn_Arc_t              PS_Arc;
disp_fn_Text_t             PS_Text;
disp_fn_SetLinestyle_t     PS_SetLinestyle;
disp_fn_SetColor_t         PS_SetColor;
disp_fn_Update_t           PS_Update;
disp_fn_Finalize_t         PS_Finalize;

#endif
