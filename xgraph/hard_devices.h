/*
 * Hardcopy Device Header
 *
 * This file declares the types required for the hardcopy table
 * found in hard_devices.c.
 */

#ifndef _h_hard_devices
#define _h_hard_devices

#define MFNAME	25

typedef enum hard_dev_docu_defn {
    NONE, NO, YES
}       hard_dev_docu;

struct hard_dev {
    char   *dev_name;		/* Device name                */
    int     (*dev_init) ();	/* Initialization function    */
    char   *dev_spec;		/* Default pipe program       */
    char    dev_file[MFNAME];	/* Default file name          */
    char    dev_printer[MFNAME];/* Default printer name       */
    double  dev_max_dim;	/* Default maximum dimension (cm)    */
    char    dev_title_font[MFNAME];	/* Default name of title font        */
    double  dev_title_size;	/* Default size of title font (pnts) */
    char    dev_axis_font[MFNAME];	/* Default name of axis font         */
    double  dev_axis_size;	/* Default size of axis font (pnts)  */
    hard_dev_docu dev_docu;	/* Document predicate                */
};

extern int hard_count;
extern struct hard_dev hard_devices[];

#define D_HPGL 0
#define D_POSTSCRIPT 1
#define D_IDRAW 2
#define D_TGIF 3
#define D_XWINDOWS 4

extern void hard_init();
#endif /* _h_hard_devices */
