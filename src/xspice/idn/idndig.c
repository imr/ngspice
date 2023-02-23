/*============================================================================
FILE    IDNdig.c

MEMBER OF process XSPICE

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    This file contains the definition of the 'digital' node type
    used by 12-state digital models in the code model library.
    These functions are called exclusively through function
    pointers in an Evt_Udn_Info_t data structure.

INTERFACES

    Evt_Udn_Info_t idn_digital_info

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#include "ngspice/ngspice.h"

#include "ngspice/cm.h"

#include "ngspice/evtudn.h"
 
/* ************************************************************************ */

static void idn_digital_create(void **evt_struct)
{
    /* Malloc space for a digital struct */

	*evt_struct = TMALLOC(Digital_t, 1);
}


/* ************************************************************************ */

static void idn_digital_dismantle(void *evt_struct)
{
    NG_IGNORE(evt_struct);

    /* Do nothing.  There are no internally malloc'ed things to dismantle */
}


/* ************************************************************************ */

static void idn_digital_initialize(void *evt_struct)
{
    Digital_t  *dig_struct = (Digital_t *) evt_struct;


    /* Initialize to unknown state and strength */
    dig_struct->state = ZERO;
    dig_struct->strength = UNDETERMINED;
}


/* ************************************************************************ */

static void idn_digital_invert(void *evt_struct)
{
    Digital_t  *dig_struct = (Digital_t *) evt_struct;


    /* Invert the state */
    switch(dig_struct->state) {

    case ZERO:
        dig_struct->state = ONE;
        return;

    case ONE:
        dig_struct->state = ZERO;
        return;

    default:
        return;
    }

}


/* ************************************************************************ */

static void idn_digital_copy(void *evt_from_struct, void *evt_to_struct)
{
    Digital_t  *dig_from_struct = (Digital_t *) evt_from_struct;
    Digital_t  *dig_to_struct   = (Digital_t *) evt_to_struct;

    /* Copy the structure */
    dig_to_struct->state = dig_from_struct->state;
    dig_to_struct->strength = dig_from_struct->strength;
}


/* ************************************************************************ */

static void idn_digital_resolve(int num_struct,
    void **evt_struct_array, void *evt_struct)
{
    Digital_t   **dig_struct_array;
    Digital_t   *dig_struct;

#define e(v, s) (((s) * 3) + (v))
#define L ZERO
#define H ONE
#define U UNKNOWN
#define s STRONG
#define r RESISTIVE
#define z HI_IMPEDANCE
#define u UNDETERMINED

    static const int map[12][12] = {
// ------------ e(L,s), e(H,s), e(U,s), e(L,r), e(H,r), e(U,r), e(L,z), e(H,z), e(U,z), e(L,u), e(H,u), e(U,u)  -----------

/* e(L,s) */  { e(L,s), e(U,s), e(U,s), e(L,s), e(L,s), e(L,s), e(L,s), e(L,s), e(L,s), e(L,s), e(U,s), e(U,s)},  // e(L,s)
/* e(H,s) */  { e(U,s), e(H,s), e(U,s), e(H,s), e(H,s), e(H,s), e(H,s), e(H,s), e(H,s), e(U,s), e(H,s), e(U,s)},  // e(H,s)
/* e(U,s) */  { e(U,s), e(U,s), e(U,s), e(U,s), e(U,s), e(U,s), e(U,s), e(U,s), e(U,s), e(U,s), e(U,s), e(U,s)},  // e(U,s)
/* e(L,r) */  { e(L,s), e(H,s), e(U,s), e(L,r), e(U,r), e(U,r), e(L,r), e(L,r), e(L,r), e(L,u), e(U,u), e(U,u)},  // e(L,r)
/* e(H,r) */  { e(L,s), e(H,s), e(U,s), e(U,r), e(H,r), e(U,r), e(H,r), e(H,r), e(H,r), e(U,u), e(H,u), e(U,u)},  // e(H,r)
/* e(U,r) */  { e(L,s), e(H,s), e(U,s), e(U,r), e(U,r), e(U,r), e(U,r), e(U,r), e(U,r), e(U,u), e(U,u), e(U,u)},  // e(H,r)
/* e(L,z) */  { e(L,s), e(H,s), e(U,s), e(L,r), e(H,r), e(U,r), e(L,z), e(U,z), e(U,z), e(L,u), e(U,u), e(U,u)},  // e(L,z)
/* e(H,z) */  { e(L,s), e(H,s), e(U,s), e(L,r), e(H,r), e(U,r), e(U,z), e(H,z), e(U,z), e(U,u), e(H,u), e(U,u)},  // e(H,z)
/* e(U,z) */  { e(L,s), e(H,s), e(U,s), e(L,r), e(H,r), e(U,r), e(U,z), e(U,z), e(U,z), e(U,u), e(U,u), e(U,u)},  // e(U,z)
/* e(L,u) */  { e(L,s), e(U,s), e(U,s), e(L,u), e(U,u), e(U,u), e(L,u), e(U,u), e(U,u), e(L,u), e(U,u), e(U,u)},  // e(L,u)
/* e(H,u) */  { e(U,s), e(H,s), e(U,s), e(U,u), e(H,u), e(U,u), e(U,u), e(H,u), e(U,u), e(U,u), e(H,u), e(U,u)},  // e(H,u)
/* e(U,u) */  { e(U,s), e(U,s), e(U,s), e(U,u), e(U,u), e(U,u), e(U,u), e(U,u), e(U,u), e(U,u), e(U,u), e(U,u)}}; // e(U,u)

// ----------- e(L,s), e(H,s), e(U,s), e(L,r), e(H,r), e(U,r), e(L,z), e(H,z), e(U,z), e(L,u), e(H,u), e(U,u)  ------------
    int i;
    int index1;
    int index2;

    /* Cast the input void pointers to pointers of the digital type */
    dig_struct = (Digital_t *) evt_struct;
    dig_struct_array = (Digital_t **)evt_struct_array;

    if (num_struct < 2) {
        /* Copy the first member of the array directly to the output */
        dig_struct->state = dig_struct_array[0]->state;
	dig_struct->strength = dig_struct_array[0]->strength;
	return;
    }

    /* Convert struct to index into map */
    index1 = (int)(dig_struct_array[0]->state +
                   dig_struct_array[0]->strength * 3);

    /* For the remaining members, perform the resolution algorithm */
    for(i = 1; i < num_struct; i++) {
        /* Convert struct to index into map */
        index2 = (int)(dig_struct_array[i]->state +
                       dig_struct_array[i]->strength * 3);

        /* Compute the result */
        index1 = map[index1][index2];
    }

    /* Convert result back to state and strength */
    dig_struct->state = (Digital_State_t) (index1 % 3);
    dig_struct->strength = (Digital_Strength_t) (index1 / 3);
}


/* ************************************************************************ */

static void idn_digital_compare(void *evt_struct1, void *evt_struct2,
    Boolean_t *equal)
{
    Digital_t  *dig_struct1 = (Digital_t *) evt_struct1;
    Digital_t  *dig_struct2 = (Digital_t *) evt_struct2;

    /* Compare the structures in order of most likely differences */
    if(dig_struct1->state != dig_struct2->state)
        *equal = FALSE;
    else if(dig_struct1->strength != dig_struct2->strength)
        *equal = FALSE;
    else
        *equal = TRUE;
}


/* ************************************************************************ */

static void idn_digital_plot_val(void *evt_struct, char *member, double *val)
{
    Digital_t   *dig_struct = (Digital_t *) evt_struct;


    /* Output a value for the requested member of the digital struct */
    if(strcmp(member,"strength") == 0) {

        /* Choose values that will not make plots lie on state plots */
        switch(dig_struct->strength) {

        case STRONG:
            *val = 0.1;
            return;

        case RESISTIVE:
            *val = 0.6;
            return;

        case HI_IMPEDANCE:
            *val = 1.1;
            return;

        case UNDETERMINED:
            *val = -0.4;
            return;
        }
    }
    else {
        /* member = "state" or anything else */

        /* Pick reasonable values */
        switch(dig_struct->state) {

        case ZERO:
            *val = 0.0;
            return;

        case ONE:
            *val = 1.0;
            return;

        case UNKNOWN:
            *val = 0.5;
            return;
        }
    }
}


/* ************************************************************************ */

static void idn_digital_print_val(void *evt_struct, char *member, char **val)
{
    Digital_t   *dig_struct = (Digital_t *) evt_struct;

    int         index;

    static char *map[] = { "0s", "1s", "Us",
                           "0r", "1r", "Ur",
                           "0z", "1z", "Uz",
                           "0u", "1u", "Uu" };


    /* Output a value for the requested member of the digital struct */

    if(strcmp(member,"state") == 0) {

        /* Pick reasonable values */
        switch(dig_struct->state) {

        case ZERO:
            *val = "0";
            return;

        case ONE:
            *val = "1";
            return;

        case UNKNOWN:
            *val = "U";
            return;

        default:
            *val = "?";
            return;
        }
    }
    else if(strcmp(member,"strength") == 0) {

        /* Choose values that will not make plots lie on state plots */
        switch(dig_struct->strength) {

        case STRONG:
            *val = "s";
            return;

        case RESISTIVE:
            *val = "r";
            return;

        case HI_IMPEDANCE:
            *val = "z";
            return;

        case UNDETERMINED:
            *val = "u";
            return;

        default:
            *val = "?";
            return;
        }
    }
    else {
        index = (int)(dig_struct->state + dig_struct->strength * 3);
        if((index < 0) || (index > 11))
            *val = "??";
        else
            *val = map[index];
        return;
    }
}



/* ************************************************************************ */

static void idn_digital_ipc_val(void *evt_struct, void **ipc_val, int *ipc_val_size)
{
    /* Return the digital data structure and its size */
    *ipc_val = evt_struct;
    *ipc_val_size = sizeof(Digital_t);
}



Evt_Udn_Info_t idn_digital_info = {

"d",
"12 state digital data",
NULL,
idn_digital_create,
idn_digital_dismantle,
idn_digital_initialize,
idn_digital_invert,
idn_digital_copy,
idn_digital_resolve,
idn_digital_compare,
idn_digital_plot_val,
idn_digital_print_val,
idn_digital_ipc_val

};

