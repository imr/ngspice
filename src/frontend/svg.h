/* Header file for SVG.c */

#ifndef ngspice_SVG_H
#define ngspice_SVG_H

disp_fn_Init_t             SVG_Init;
disp_fn_NewViewport_t      SVG_NewViewport;
disp_fn_Close_t            SVG_Close;
disp_fn_Clear_t            SVG_Clear;
disp_fn_DrawLine_t         SVG_DrawLine;
disp_fn_Arc_t              SVG_Arc;
disp_fn_Text_t             SVG_Text;
disp_fn_SetLinestyle_t     SVG_SetLinestyle;
disp_fn_SetColor_t         SVG_SetColor;
disp_fn_Update_t           SVG_Update;
disp_fn_Finalize_t         SVG_Finalize;

#endif
