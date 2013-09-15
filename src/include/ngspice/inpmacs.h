/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

/* a couple of macros to make much of the input code
 * much much shorter and easier to handle.
 *
 * First, call the simulator function 'func', giving it the arglist
 * 'args' (must have its own parens!) and check the error returned.
 * If necessary, get the proper error message and tack it on the current
 * error
 */

#define IFC(func, args)                                                 \
    do {                                                                \
        error = (*(ft_sim->func)) args;                                 \
        if (error)                                                      \
            current->error = INPerrCat(current->error, INPerror(error)); \
    } while(0)


/* and one for calling more General functions that still return an
 * error code as above
 */

#define GCA(func, args)                                                 \
    do {                                                                \
        error = func args;                                              \
        if (error)                                                      \
            current->error = INPerrCat(current->error, INPerror(error)); \
    } while(0)


/* and one for putting our own error messages onto the current
 * line's error string
 */

#define LITERR(text)                                                    \
    do {                                                                \
        current->error = INPerrCat(current->error, INPmkTemp(text));    \
    } while(0)


/* and now a special one for calling INPdevParse which returns an
 * already concatenated list of error messages or NUL
 */

#define PARSECALL(args)                                                 \
    do {                                                                \
        current->error = INPerrCat(current->error, INPdevParse args);   \
    } while(0)
