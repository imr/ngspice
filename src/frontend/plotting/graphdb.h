/*************
 * Header file for graphdb.c
 * 1999 E. Rouat
 ************/

#ifndef GRAPHDB_H_INCLUDED
#define GRAPHDB_H_INCLUDED

int DestroyGraph(int id);
void FreeGraphs(void);
void SetGraphContext(int graphid);
void PushGraphContext(GRAPH *graph);
void PopGraphContext(void);

#endif
