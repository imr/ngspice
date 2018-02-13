// DEVdelete() change API
//   drop instance and model search loop
//   don't FREE the instance struct
//
// (compile "git grep -l -e 'delete' -- '*.[ch]' | xargs -P8 -n1 spatch --sp-file klag-1.cocci --in-place && git commit -am 'DEVdelete(), change API, #1/2, coccinelle semantic patch'")
//
// (compile "spatch --sp-file klag-1.cocci src/spicelib/devices/bjt/*.[ch]")
// (compile "spatch --sp-file klag-1.cocci src/spicelib/devices/asrc/asrcdel.c")
// (compile "spatch --sp-file klag-1.cocci src/spicelib/devices/bsim1/b1del.c")
// (compile "spatch --sp-file klag-1.cocci src/xspice/mif/mifdelete.c")


@unify_DEVdelete@
identifier fn =~ "delete$";
typedef GENmodel, IFuid, GENinstance;
identifier XinModel, Xname, Xkill;
type Tm, Ti;
identifier Xprev, Xhere, Xmod, Xinst;
identifier XXXnextModel, XXXinstances, XXXnextInstance, XXXname;
symbol OK, E_NODEV;
@@
  int fn (
- GENmodel *XinModel , IFuid Xname , GENinstance **Xkill
+ GENinstance *gen_inst
  )
  {
(
-     Ti **Xinst = (Ti **) Xkill;
-     Tm *Xmod = (Tm *) XinModel;
|
-     Tm *Xmod = (Tm *) XinModel;
-     Ti **Xinst = (Ti **) Xkill;
)
-     Ti **Xprev = NULL;
-     Ti *Xhere;
-     for (; Xmod; Xmod = Xmod->XXXnextModel) {
-         Xprev = &(Xmod->XXXinstances);
-         for (Xhere = *Xprev; Xhere; Xhere = *Xprev) {
-             if (Xhere->XXXname == Xname || (Xinst && Xhere == *Xinst)) {
-                 *Xprev = Xhere->XXXnextInstance;
-                 FREE(Xhere);
-                 return(OK);
-             }
-             Xprev = &(Xhere->XXXnextInstance);
-         }
-     }
-     return(E_NODEV);
+     NG_IGNORE(gen_inst);
+     return OK;
  }


@unify_DEVdelete_h@
identifier fn =~ "delete$";
typedef GENmodel, IFuid, GENinstance;
@@
  extern int fn (
- GENmodel *, IFuid, GENinstance *
+ GENinstance
  *
  );
