/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

#ifndef ngspice_FTEDEBUG_H
#define ngspice_FTEDEBUG_H

/*
 *
 * Stuff for debugging a spice run. Debugging information will be printed
 * on the standard output.
 */

#define	DB_TRACENODE	1   /* Print the value of a node every iteration. */
#define	DB_TRACEALL	2   /* Trace all nodes. */
#define	DB_STOPAFTER	3   /* Break after this many iterations. */
#define	DB_STOPWHEN	4   /* Break when a node reaches this value. */
#define	DB_IPLOT	5   /* Incrementally plot the stuff. */
#define	DB_IPLOTALL	6   /* Incrementally plot everything. */
#define	DB_SAVE		7   /* Save a node. */
#define	DB_SAVEALL	8   /* Save everything. */
#define	DB_DEADIPLOT	9   /* Incrementally plot the stuff. */

/* These only make sense in the real case, so always use realpart(). It's too
 * bad that we have to say "ge" instead of ">=", etc, but cshpar takes
 * the <>'s.
 */

#define DBC_EQU  1   /* == (eq) */
#define DBC_NEQ  2   /* != (ne) */
#define DBC_GT    3   /* >  (gt) */
#define DBC_LT    4   /* <  (lt) */
#define DBC_GTE  5   /* >= (ge) */
#define DBC_LTE  6   /* <= (le) */

/* Below, members db_op and db_value1 are re-purposed by iplot options. */

struct dbcomm {
    int db_number;    /* The number of this debugging command. */
    char db_type;      /* One of the above. */
    char *db_nodename1; /* What node. */
    char *db_nodename2; /* What node. */
    char *db_analysis; /* for a specific analysis. */
    int db_iteration;   /* For the DB_STOPAFTER command. */
    int db_op;  /* For DB_STOPWHEN. */
    double db_value1;   /* If this is DB_STOPWHEN. */
    double db_value2;   /* If this is DB_STOPWHEN. */
    int db_graphid; /* If iplot, id of graph. */
    struct dbcomm *db_also; /* Link for conjunctions. */
    struct dbcomm *db_next; /* List of active debugging commands. */
} ;

#endif
