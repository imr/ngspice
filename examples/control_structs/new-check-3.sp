new ft_getpnames parser check 3, try ternary

* (compile (concat "tmp-1/ng-spice-rework/src/ngspice "  buffer-file-name) t)

VIN  1 0  DC=0

.control

dc VIN 0 10 5

* trying the ternary

let checks = 0

let const0 = 0
let const5 = 5
let const6 = 6


let tmp = const0 ? const5 : const6
if tmp eq const6
  let checks = checks + 1
else
  echo "ERROR:"
end

let tmp = const6 ? const5 : const6
if tmp eq const5
  let checks = checks + 1
else
  echo "ERROR:"
end

define foo(a,b,d) a ? b : d

if foo(const0,const5,const6) eq const6
  let checks = checks + 1
else
  echo "ERROR:"
end

if foo(const6,const5,const6) eq const5
  let checks = checks + 1
else
  echo "ERROR:"
end

let vec7 = 7*unitvec(7)
let vec8 = 8*unitvec(8)

if length(const5 ? vec7 : vec8) eq 7
  let checks = checks + 1
else
  echo "ERROR:"
end

if length(const0 ? vec7 : vec8) eq 8
  let checks = checks + 1
else
  echo "ERROR:"
end

* FIXME, "1 ? 1:1" (without spaces around of ':') doesnt work,
*   "1:1" is a lexem, WHY !!!
*   ist that an old artifact, (ancient hierarchical name separator ':')
*
*print length(1?1:1)

*if (1 ? 1:1) eq 1
if (1 ? 1 : 1) eq 1
  let checks = checks + 1
else
  echo "ERROR:"
end

print @vin[dc]

* '"' survives, and will be processed in the ft_getpnames() lexer, that is PPlex()
*   where the string will be unqoted
*   thats used vor weired variable names, for example "zero(1)"
let foo = "vec8"
if foo eq vec8
  let checks = checks + 1
else
  echo "ERROR:"
end

if checks eq 8
  echo "INFO: ok"
else
  echo "ERROR:"
end

.endc

.end
