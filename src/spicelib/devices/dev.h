#ifndef _DEV_H
#define _DEV_H


void spice_init_devices(void);
int num_devices(void);
IFdevice **devices_ptr(void);
SPICEdev **devices(void);

#endif

