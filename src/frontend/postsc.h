/*************
 * Header file for postsc.c
 * 1999 E. Rouat
 ************/

#ifndef POSTSC_H_INCLUDED
#define POSTSC_H_INCLUDED

int PS_Init(void);
int PS_NewViewport(GRAPH *graph);
void PS_Close(void);
void PS_Clear(void);
void PS_DrawLine(int x1, int y1, int x2, int y2);
void PS_Arc(int x0, int y0, int r, double theta1, double theta2);
void PS_Text(char *text, int x, int y);
int PS_SetLinestyle(int linestyleid);
void PS_SetColor(int colorid);
void PS_Update(void);


#endif
