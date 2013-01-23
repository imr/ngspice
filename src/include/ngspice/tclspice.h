/*Include file to allow spice to export certain data */
#ifndef ngspice_TCLSPICE_H
#define ngspice_TCLSPICE_H

extern int steps_completed;
extern void blt_init(void *run);
extern void blt_add(int index,double value);
extern void blt_relink(int index, void* v);
extern void blt_lockvec(int index);

/* For things to do per loop */
int Tcl_ExecutePerLoop(void);

#include "ngspice/graph.h"
#include "ngspice/ftedev.h"

/* For tk ploting */
disp_fn_Init_t             sp_Tk_Init;
disp_fn_NewViewport_t      sp_Tk_NewViewport;
disp_fn_Close_t            sp_Tk_Close;
disp_fn_Clear_t            sp_Tk_Clear;
disp_fn_DrawLine_t         sp_Tk_DrawLine;
disp_fn_Arc_t              sp_Tk_Arc;
disp_fn_Text_t             sp_Tk_Text;
disp_fn_DefineColor_t      sp_Tk_DefineColor;
disp_fn_DefineLinestyle_t  sp_Tk_DefineLinestyle;
disp_fn_SetLinestyle_t     sp_Tk_SetLinestyle;
disp_fn_SetColor_t         sp_Tk_SetColor;
disp_fn_Update_t           sp_Tk_Update;

/* The blt callback method */
#include "ngspice/dvec.h"
extern int blt_plot(struct dvec *y,struct dvec *x,int new);

#endif
