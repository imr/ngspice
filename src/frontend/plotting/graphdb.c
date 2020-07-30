/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

/*
  Manage graph data structure.
*/

#include "ngspice/ngspice.h"
#include "ngspice/graph.h"
#include "ngspice/ftedebug.h"
#include "ngspice/fteext.h"

#include "ngspice/ftedbgra.h"
#include "graphdb.h"
#include "../breakp2.h"
#include "../display.h"


/* invariant:  currentgraph contains the current graph */
GRAPH *currentgraph;

/*
 *  We use a linked list rather than a circular one because we
 *    expect few links per list and we don't need to add at the
 *    end of a list (we can add at the beginning).
 */

/* linked list of graphs */
typedef struct listgraph {
    /* we use GRAPH here instead of a pointer to save a tmalloc */
    GRAPH graph;
    struct listgraph *next;
} LISTGRAPH;

#define NUMGBUCKETS 16

typedef struct gbucket {
    LISTGRAPH *list;
} GBUCKET;

static GBUCKET GBucket[NUMGBUCKETS];

/* note: Zero is not a valid id.  This is used in plot() in graf.c. */
static int RunningId = 1;

/* Initialize graph structure */
static inline void setgraph(GRAPH *pgraph, int id)
{
    pgraph->graphid = id;
    pgraph->degree = 1;
    pgraph->linestyle = -1;
} /* end of function setgraph */



/* Creates a new graph. Returns NULL on error */
GRAPH *NewGraph(void)
{
    LISTGRAPH *list;
    const int BucketId = RunningId % NUMGBUCKETS;

    /* allocate memory for graph via LISTGRAPH */
    if ((list = TMALLOC(LISTGRAPH, 1)) == NULL) {
        internalerror("can't allocate a listgraph");
        return (GRAPH *) NULL;
    }

    GRAPH * const pgraph = &list->graph;
    setgraph(pgraph, RunningId);
    GBUCKET *p_bucket = GBucket + BucketId;

    /* Add to the appropriate bucket at the front of the linked list */
    if (!p_bucket->list) { /* no list yet */
        p_bucket->list = list;
    }
    else {
        /* insert at front of current list */
        list->next = p_bucket->list;
        p_bucket->list = list;
    }

    RunningId++;

    return pgraph;
} /* end of function NewGraph */



/* Given graph id, return graph */
GRAPH *FindGraph(int id)
{
    LISTGRAPH *list;

    /* Step through list of graphs until found or list ends */
    for (list = GBucket[id % NUMGBUCKETS].list;
            list && list->graph.graphid != id;
            list = list->next) {
        ;
    }

    if (list) { /* found */
        return &list->graph;
    }
    else {
        return (GRAPH *) NULL;
    }
} /* end of function FindGraph */



GRAPH *CopyGraph(GRAPH *graph)
{
    GRAPH *ret;
    struct dveclist *link = NULL, *newlink = NULL;

    if (!graph) {
        return NULL;
    }

    ret = NewGraph();

    {
        const int id = ret->graphid; /* save ID of the new graph */
        memcpy(ret, graph, sizeof(GRAPH)); /* copy graph info (inc. ID) */
        ret->graphid = id;   /* restore ID */
    }

    /* copy keyed */
    {
        struct _keyed *k;
        for (ret->keyed = NULL, k = graph->keyed; k; k = k->next) {
            SaveText(ret, k->text, k->x, k->y);
        }
    }

    /* copy dvecs or reuse if "borrowed" already */
    {
        struct dveclist *new_plotdata = (struct dveclist *) NULL;
        for (link = graph->plotdata; link; link = link->next) {
            if (link->f_own_vector) {
                struct dvec * const old_vector = link->vector;
                struct dvec * const new_vector = vec_copy(old_vector);
                /* vec_copy doesn't set v_color or v_linestyle */
                new_vector->v_color = old_vector->v_color;
                new_vector->v_linestyle = old_vector->v_linestyle;
                new_vector->v_flags |= VF_PERMANENT;
                newlink = TMALLOC(struct dveclist, 1);
                newlink->next = new_plotdata;
                newlink->f_own_vector = TRUE;
                newlink->vector = new_vector;

                /* If the link owns the vector, it also owns its scale
                 * vector, if present */
                struct dvec *old_scale = old_vector->v_scale;
                if (old_scale != (struct dvec *) NULL) {
                    new_plotdata = newlink; /* put in front */
                    struct dvec * const new_scale = vec_copy(old_scale);
                    new_scale->v_flags |= VF_PERMANENT;
                    newlink->vector->v_scale = new_scale;
                }
            }
            else {
                newlink->vector = link->vector;
                newlink->f_own_vector = FALSE;
            }
           new_plotdata = newlink; /* put in front */
        }

        ret->plotdata = new_plotdata; /* give vector list to plot */
    } /* end of block copying or reusing dvecs */

    ret->commandline = copy(graph->commandline);
    ret->plotname = copy(graph->plotname);

    {
        const char * const lbl = graph->grid.xlabel;
        if (lbl) {
            ret->grid.xlabel = copy(lbl);
        }
    }

    {
        const char * const lbl = graph->grid.ylabel;
        if (lbl) {
            ret->grid.ylabel = copy(lbl);
        }
    }

    /* Copy devdep information and size if present */
    {
        const void * const p = graph->devdep;
        if (p != NULL) {
            const size_t n = ret->n_byte_devdep = graph->n_byte_devdep;
            void * const dst = ret->devdep = tmalloc(n);
            (void) memcpy(dst, graph->devdep, n);
        }
    }

    return ret;
} /* end of function CopyGraph */



int DestroyGraph(int id)
{
    /* Locate hash bucket for this graph */
    const int index = id % NUMGBUCKETS;
    LISTGRAPH *list = GBucket[index].list;

    /* Pointer before current one. Allows fixing list when the current
     * node is deleted. Init to NULL to indicate that at head of list */
    LISTGRAPH *lastlist = (LISTGRAPH *) NULL;

    /* Step through graphs in the bucket until the one with id is found */
    while (list) {
        if (list->graph.graphid == id) {  /* found it */
            struct _keyed *k, *nextk;
            struct dbcomm *db;

            /* Fix the iplot/trace dbs list */
            for (db = dbs; db && db->db_graphid != id; db = db->db_next) {
                ;
            }

            if (db && (db->db_type == DB_IPLOT ||
                    db->db_type == DB_IPLOTALL)) {
                db->db_type = DB_DEADIPLOT;
                /* Delete this later */
                return 0;
            }

            /* Adjust bucket pointers to remove the current node */
            if (lastlist) { /* not at front */
                lastlist->next = list->next;
            }
            else { /* at front */
                GBucket[index].list = list->next;
            }

            /* Run through and de-allocate dynamically allocated keyed list */
            k = list->graph.keyed;
            while (k) {
                nextk = k->next;
                txfree(k->text);
                txfree(k);
                k = nextk;
            }

            /* Free vectors owned by this graph and free the list */
            {
                struct dveclist *d = list->graph.plotdata;
                struct dveclist *nextd;
                while (d != (struct dveclist *) NULL) {
                    nextd = d->next;
                    if (d->f_own_vector) {
                        /* list responsible for freeing this vector */
                        if (d->vector->v_scale) {
                            dvec_free(d->vector->v_scale);  
                        }                          
                        dvec_free(d->vector);
                    }
                    txfree(d);                    
                    d = nextd;                    
                }
            }

            txfree(list->graph.commandline);
            txfree(list->graph.plotname);
            txfree(list->graph.grid.xlabel);
            txfree(list->graph.grid.ylabel);

            /* If device-dependent space was allocated, free it. */
            {
                void * const p = list->graph.devdep;
                if (p) {
                    txfree(p);
                }
            }

            txfree(list);

            return 1;
        } /* end of case that graph ID was found */

        lastlist = list; /* update previous node */
        list = list->next; /* step to next node */
    } /* end of loop over graphs in the current bucket */

    /* The graph with ID id was not found */
    internalerror("tried to destroy non-existent graph");
    return 0;
} /* end of function DestroyGraph */



/* Free up all dynamically allocated data structures */
void FreeGraphs(void)
{
    /* Iterate over all hash buckets */
    GBUCKET *gbucket;
    for (gbucket = GBucket; gbucket < &GBucket[NUMGBUCKETS]; gbucket++) {
        LISTGRAPH * list = gbucket->list; /* linked list of graphs here */
        while (list) { /* Free each until end of list */
            LISTGRAPH *deadl = list;
            list = list->next;
            txfree(deadl);
        }
    }
} /* end of functdion FreeGraphs */



/* This function sets global varial currentgraph based on graphid */
void SetGraphContext(int graphid)
{
    currentgraph = FindGraph(graphid);
} /* end of function SetGraphContext */



/* Stack of graph objects implemented as a linked list */
typedef struct gcstack {
    GRAPH *pgraph;
    struct gcstack *next;
} GCSTACK;

static GCSTACK *gcstacktop; /* top of the stack of graphs */


/* note: This Push and Pop has tricky semantics.
   Push(graph) will push the currentgraph onto the stack
   and set currentgraph to graph.
   Pop() simply sets currentgraph to previous value at the top of the stack
   and pops stack.
*/
void PushGraphContext(GRAPH *graph)
{
    GCSTACK *gcstack = TMALLOC(GCSTACK, 1);

    if (!gcstacktop) {
        gcstacktop = gcstack;
    }
    else {
        gcstack->next = gcstacktop;
        gcstacktop = gcstack;
    }
    gcstacktop->pgraph = currentgraph;
    currentgraph = graph;
} /* end of function PushGraphContext */



void PopGraphContext(void)
{
    currentgraph = gcstacktop->pgraph; /* pop from stack, making current */
    GCSTACK *dead = gcstacktop; /* remove from stack */
    gcstacktop = gcstacktop->next;
    txfree(dead); /* free allocation */
} /* end of function PopGraphContext */



