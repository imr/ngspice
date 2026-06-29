simple test for names = ft_getpnames() versus free_pnode(names)
* altermod

R1 1 0 RE
V1 1 0 1

.model RE r r=1

.control
  op
  print all
  define gauss(nom, var, sig) (nom + (nom*var)/sig * sgauss(0))
  altermod r1 R = (1 + (1 * 2) / 3 * sgauss(0)) ; no leak
*  altermod @r1[r] = gauss(1,2,3) ; leak
*  altermod r1 r = gauss(1,2,3) ; leak
  op
  print all
  quit
.endc

.end
