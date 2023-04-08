/**********
Copyright 2023 The ngspice team.  All rights reserved.
License: Three-clause BCD
Author: 2023 Holger Vogt
**********/

/*
  For dealing with compatibility transformations

  PSICE, LTSPICE and others
*/

extern void print_compat_mode(void);
extern void set_compat_mode(void);
extern void pspice_compat_a(struct card* oldcard);
extern void ltspice_compat_a(struct card* oldcard);
extern struct card* pspice_compat(struct card* newcard);
extern struct card* ltspice_compat(struct card* oldcard);


