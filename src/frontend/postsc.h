/*************
 * Header file for postsc.c
 * 1999 E. Rouat
 ************/

#ifndef POSTSC_H_INCLUDED
#define POSTSC_H_INCLUDED

int PS_Init(void);
int PS_NewViewport(GRAPH *graph);
int PS_Close(void);
int PS_Clear(void);
int PS_DrawLine(int x1, int y1, int x2, int y2);
int PS_Arc(int x0, int y0, int r, double theta1, double theta2);
int PS_Text(char *text, int x, int y);
int PS_SetLinestyle(int linestyleid);
int PS_SetColor(int colorid);
int PS_Update(void);


#endif
