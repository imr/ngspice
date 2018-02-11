// functionalise access to certain struct GENmodel and struct GENinstance slots

// (compile "ulimit -s 20000; git grep -l -e 'nextModel\\|nextInstance\\|instances\\|modPtr' -- '*.c' | xargs -P8 -n1 spatch --defined XSPICE --sp-file klag-4.cocci --in-place && git commit -am 'GENmodel, GENinstance, change layout, #1/4, coccinelle semantic patch'")

// (compile "spatch --sp-file klag-4.cocci src/spicelib/devices/isrc/isrcload.c --defined XSPICE")
// (compile "spatch --sp-file klag-4.cocci src/frontend/spiceif.c")
// (compile "spatch --sp-file klag-4.cocci src/spicelib/devices/bsimsoi/b4soild.c")
// (compile "spatch --sp-file klag-4.cocci src/spicelib/devices/cpl/*.c")
// (compile "spatch --sp-file klag-4.cocci src/spicelib/devices/cpl/*.c")


@anyid@
identifier el =~ "(nextModel|nextInstance|instances|modPtr)$";
identifier p;
@@

p->el


@functionalise_nextThing@
identifier anyid.el !~ "(INP|GEN)(nextModel|nextInstance|instances|modPtr)$";
identifier p;
@@
- p->el
+ el(p)
