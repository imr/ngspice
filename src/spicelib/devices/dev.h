#ifndef _DEV_H
#define _DEV_H


int num_devices(void);
IFdevice **devices_ptr(void);
SPICEdev **devices(void);
SPICEdev *first_device(void);
SPICEdev *next_device(SPICEdev *current);

#endif

