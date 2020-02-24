/*************
* Header file for winprint.c
************/

#ifndef ngspice_WINPRINT_H
#define ngspice_WINPRINT_H

void WPRINT_PrintInit(HWND hwnd);
BOOL CALLBACK WPRINT_Abort( HDC hdc, int iError);
int WPRINT_Init(void);
int WPRINT_NewViewport( GRAPH * graph);
int WPRINT_Close(void);
int WPRINT_Clear(void);
int WPRINT_DrawLine(int x1, int y1, int x2, int y2, bool isgrid);
int WPRINT_Arc(int x0, int y0, int radius, double theta, double delta_theta);
int WPRINT_Text( char * text, int x, int y, int degrees);
int WPRINT_DefineColor(int colorid, double red, double green, double blue);
int WPRINT_DefineLinestyle(int num, int mask);
int WPRINT_SetLinestyle(int style);
int WPRINT_SetColor( int color);
int WPRINT_Update(void);
int WPRINT_DiagramReady(void);

#endif
