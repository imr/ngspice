/*Include file to allow spice to export certain data */
#ifndef TCLSPICE_H
#define TCLSPICE_H

extern int steps_completed;
extern void blt_init(void *run);
extern void blt_add(int index,double value);
extern void blt_relink(int index, void* v);
extern void blt_lockvec(int index);

/* For things to do per loop */
int Tcl_ExecutePerLoop();

/* For tk ploting */
extern int  sp_Tk_Init(void);
#include <graph.h>
extern int  sp_Tk_NewViewport(GRAPH *graph);
extern int  sp_Tk_Close(void);
extern int  sp_Tk_Clear(void);
extern int  sp_Tk_DrawLine(int x1, int y1, int x2, int y2);
extern int  sp_Tk_Arc(int x0, int y0, int radius, double theta1, double theta2);
extern int  sp_Tk_Text(char *text, int x, int y);
extern int  sp_Tk_DefineColor(int colorid, double red, double green, double blue);
extern int  sp_Tk_DefineLinestyle(int linestyleid, int mask);
extern int  sp_Tk_SetLinestyle(int linestyleid);
extern int  sp_Tk_SetColor(int colorid);
extern int  sp_Tk_Update(void);

/* The blt callback method */
#include <dvec.h>
extern int blt_plot(struct dvec *y,struct dvec *x,int new);

#endif
