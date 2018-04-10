// (compile "for i in $(git ls-files -- 'src/spicelib/devices/*/*.c' | xargs -n 1 dirname | uniq); do spatch --in-place --sp-file tmp-empty-fun.cocci $i/*.[ch]; done")
// (compile "git commit -am 'spicelib/devices/*, drop empty destroy functions'")

// (compile "spatch --sp-file tmp-empty-fun.cocci src/spicelib/devices/mos1/*.[ch]")
// (compile "spatch --sp-file tmp-empty-fun.cocci src/spicelib/devices/ind/*.[ch]")

// fixme, beware of the xml files !!!

@thefun@
type T1;
identifier fun =~ "(destroy|mDelete|delete)$";
position pfun;
identifier arg;
symbol OK;
@@
  T1 fun@pfun(...)
  {
?   NG_IGNORE(arg);
?   return OK;
  }

@null_init@
identifier devinit;
identifier thefun.fun;
identifier member;
@@
  SPICEdev devinit = {
    ...,
-  .member = fun,
+  .member = NULL,
    ...
  };

@remove_proto@
type thefun.T1;
identifier thefun.fun;
@@
- T1 fun(...);

@initialize:python@
@@
import os
import subprocess
files = []

@script:python collect@
p << thefun.pfun;
@@
files.append(p[0].file)

@finalize:python@
@@
for f in files:
    print f
    os.remove(f)
    subprocess.call(["emacs", "-q", "--no-site-lisp", "--script", "tmp-empty-fun.el", f])
