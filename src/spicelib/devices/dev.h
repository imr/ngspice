#ifndef _DEV_H
#define _DEV_H


void spice_init_devices(void);
int num_devices(void);
IFdevice **devices_ptr(void);
SPICEdev **devices(void);
#ifdef XSPICE
int load_opus(char *);
int DEVflag(int type);
#endif
#ifdef DEVLIB
void load_alldevs(void);
int load_dev(char *name);
#endif
#endif

