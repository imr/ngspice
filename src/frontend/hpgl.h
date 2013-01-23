/*************
* Header file for hpgl.c
************/

#ifndef ngspice_HPGL_H
#define ngspice_HPGL_H

disp_fn_Init_t             GL_Init;
disp_fn_NewViewport_t      GL_NewViewport;
disp_fn_Close_t            GL_Close;
disp_fn_Clear_t            GL_Clear;
disp_fn_DrawLine_t         GL_DrawLine;
disp_fn_Arc_t              GL_Arc;
disp_fn_Text_t             GL_Text;
disp_fn_SetLinestyle_t     GL_SetLinestyle;
disp_fn_SetColor_t         GL_SetColor;
disp_fn_Update_t           GL_Update;


#endif
