/*************
 * Header file for graphdb.c
 * 1999 E. Rouat
 ************/

#ifndef GRAPHDB_H_INCLUDED
#define GRAPHDB_H_INCLUDED

GRAPH *NewGraph(void);
GRAPH *FindGraph(int id);
GRAPH *CopyGraph(GRAPH *graph);
int DestroyGraph(int id);
void FreeGraphs(void);
void SetGraphContext(int graphid);
void PushGraphContext(GRAPH *graph);
void PopGraphContext(void);



#endif
