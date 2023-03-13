/* Definitions for inertial transport for XSPICE digital models. */

/* Determine whether inertial transport should be used. */

/* Values for control variable, "digital_delay_type". */

#define DEFAULT_TRANSPORT  0
#define DEFAULT_INERTIAL   1
#define OVERRIDE_TRANSPORT 2
#define OVERRIDE_INERTIAL  3

enum param_vals {Off, On, Not_set};
Mif_Boolean_t cm_is_inertial(enum param_vals param);

/* Extra state data for inertial delays, one structure per output. */

struct idata {
    double          when;
    Digital_State_t prev;
};


