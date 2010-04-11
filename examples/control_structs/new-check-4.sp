demonstrate < etc in ft_getpnames

* (compile (concat "tmp-1/ng-spice-rework/src/ngspice " buffer-file-name) t)

VIN  1 0  DC=0

.control

dc VIN 0 10 5

let checks = 0

let const0 = 0
let const5 = 5
let const6 = 6

* check some relational operators, which are in danger to mixed up
*   with csh semantic, that is IO redirection

if const5 < const6
  let checks = checks + 1
else
  echo "ERROR:"
end

if const6 > const5
  let checks = checks + 1
else
  echo "ERROR:"
end

if const5 >= const5
  let checks = checks + 1
else
  echo "ERROR:"
end

if const5 <= const5
  let checks = checks + 1
else
  echo "ERROR:"
end

if const5 = const5
  let checks = checks + 1
else
  echo "ERROR:"
end

* check some wired non-equality operators
*   note: there are some awkward tranformations ahead of the ft_getpnames lexer
*     transforming "><" into "> <"
*     and          "<>" into "< >"
*   note: "!=" would have been in serious danger to be fooled up within
*     csh history mechanism

if const6 <> const5
  let checks = checks + 1
else
  echo "ERROR:"
end

if const6 >< const5
  let checks = checks + 1
else
  echo "ERROR:"
end


* check some boolean operators, which are in danger to be mixed up
*   with csh semantic, `&' background '|' pipe  '~' homedirectory

if const5 & const5
  let checks = checks + 1
else
  echo "ERROR:"
end

if const0 | const5
  let checks = checks + 1
else
  echo "ERROR:"
end

if ~ const0
  let checks = checks + 1
else
  echo "ERROR:"
end

* note:
*   "!=" would be in danger, '!' triggers the csh history mechanism
*if const5 != const6
*  echo "just trying"
*end


* Note: csh semantics swallows the '>' and '<' operators
*   on most of the com lines
* witnessed by
let tmp = const5 > unwanted_output_file_1
define foo(a,b) a > unwanted_output_file_2
print const0 > unwanted_output_file_3

if checks eq 10
  echo "INFO: ok"
end

.endc

.end
