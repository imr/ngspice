//! Subcircuit expansion with hierarchical parameter scoping — the `subckt.c`
//! replacement. Given the flattened, section-resolved card stream, this:
//!   * registers every `.subckt` definition (ports + default params + body),
//!   * walks the top-level circuit, and for each `X` instance recursively
//!     expands the subckt: binds instance params into a child [`Scope`],
//!     evaluates `.if/.elseif/.else/.endif` conditionals, renames internal nodes
//!     (`prefix.node`) and subckt-local models (`prefix:model`), and substitutes
//!     `{...}`/`'...'` param expressions to numbers,
//!   * emits a flat, resolved card stream ready for ngspice `INPpas1`.
//!
//! Scoping: a `Scope` holds this level's `.param` defs plus pre-bound instance
//! params, chained to its parent. Parameter lookup walks the chain, so a subckt's
//! local `.param leff='l-2*dl'` resolves against the instance's bound `l`.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use crate::config::Config;
use crate::expr::{eval, eval_builtin, parse, BinOp, Env, EvalError, Expr, UnOp};
use crate::params::{fmt_num, key, parse_assignments, parse_func_def, Assign, SubstStats};
use crate::reader::LogicalLine;

type FuncMap = HashMap<String, (Vec<String>, Arc<Expr>)>;

/// Result of partial evaluation: either a folded constant or a rewritten symbolic
/// expression string (with runtime refs — `v()`/`i()`/`temper`/`time` — preserved).
#[derive(Clone)]
enum Part {
    Const(f64),
    Sym(String),
}

fn part_str(p: &Part) -> String {
    match p {
        Part::Const(v) => fmt_num(*v),
        Part::Sym(s) => s.clone(),
    }
}

/// Extract identifier-like tokens (param names, function names) from an expression
/// string, normalized via `key()`. Used only for the global/local draw
/// classification in `compute_global_shared`.
fn ident_tokens(s: &str) -> HashSet<String> {
    let b = s.as_bytes();
    let mut out = HashSet::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_ascii_alphabetic() || c == '_' {
            let st = i;
            while i < b.len() && {
                let d = b[i] as char;
                d.is_ascii_alphanumeric() || d == '_'
            } {
                i += 1;
            }
            out.insert(key(&s[st..i]));
        } else {
            i += 1;
        }
    }
    out
}

/// Does this rhs contain a statistical draw function literally?
fn rhs_has_draw(s: &str) -> bool {
    let l = s.to_ascii_lowercase();
    l.contains("agauss(")
        || l.contains("gauss(")
        || l.contains("aunif(")
        || l.contains("unif(")
        || l.contains("limit(")
}

/// Classify the GLOBAL statistical params (shared process draws) out of the
/// top-level `.param` set. A param is global-shared iff it is (transitively) a
/// statistical draw AND it is referenced by at least one OTHER top-level `.param`.
/// See the `global_shared` field of `SubcktExpander` for the full rationale.
///
/// `raw` is the top-level param map (name -> rhs), already `key()`-normalized.
fn compute_global_shared(raw: &HashMap<String, String>) -> HashSet<String> {
    // 1. Statistical closure: params that literally draw, plus any param whose rhs
    //    (transitively) references a statistical param.
    let mut stat: HashSet<String> = raw
        .iter()
        .filter(|(_, rhs)| rhs_has_draw(rhs))
        .map(|(n, _)| key(n))
        .collect();
    loop {
        let mut added = false;
        for (n, rhs) in raw {
            let kn = key(n);
            if stat.contains(&kn) {
                continue;
            }
            if ident_tokens(rhs).iter().any(|id| stat.contains(id)) {
                stat.insert(kn);
                added = true;
            }
        }
        if !added {
            break;
        }
    }
    // 2. Params referenced by at least one OTHER top-level param's rhs.
    let mut referenced: HashSet<String> = HashSet::new();
    for (owner, rhs) in raw {
        let ko = key(owner);
        for id in ident_tokens(rhs) {
            if id != ko && raw.contains_key(&id) {
                referenced.insert(id);
            }
        }
    }
    // 3. Global-shared = statistical AND referenced-by-top-level.
    stat.intersection(&referenced).cloned().collect()
}

fn binop_str(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Rem => "%",
        BinOp::Pow => "**",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::Lt => "<",
        BinOp::Gt => ">",
        BinOp::Le => "<=",
        BinOp::Ge => ">=",
        BinOp::And => "&&",
        BinOp::Or => "||",
    }
}

/// A lexical scope in the subckt hierarchy.
pub struct Scope {
    /// param name (lower) -> raw rhs, evaluated lazily in THIS scope.
    raw: RefCell<HashMap<String, String>>,
    /// pre-bound values (instance params evaluated in the parent scope).
    locals: HashMap<String, f64>,
    parsed: RefCell<HashMap<String, Arc<Expr>>>,
    values: RefCell<HashMap<String, f64>>,
    resolving: RefCell<HashSet<String>>,
    /// Memoized partial-evaluation verdict per parameter (see partial()'s Var
    /// arm). The transitive-runtime check re-partials a param's raw definition
    /// at EVERY reference; on PDK decks whose params form deep chains that is
    /// exponential without this cache (foundry_b load went 2s -> 67s when the
    /// check landed). Sound per Scope instance: a Scope is only ever used with
    /// one (nmap, prefix) pair, so the rendered Sym strings cannot differ.
    /// Only consulted/populated when no function-arg locals are in effect.
    partial_cache: RefCell<HashMap<String, Part>>,
    parent: Option<Rc<Scope>>,
    funcs: Arc<FuncMap>,
}

impl Scope {
    fn root(funcs: Arc<FuncMap>) -> Rc<Scope> {
        Rc::new(Scope {
            raw: RefCell::new(HashMap::new()),
            locals: HashMap::new(),
            parsed: RefCell::new(HashMap::new()),
            values: RefCell::new(HashMap::new()),
            resolving: RefCell::new(HashSet::new()),
            partial_cache: RefCell::new(HashMap::new()),
            parent: None,
            funcs,
        })
    }

    fn child(parent: &Rc<Scope>, locals: HashMap<String, f64>) -> Rc<Scope> {
        Rc::new(Scope {
            raw: RefCell::new(HashMap::new()),
            locals,
            parsed: RefCell::new(HashMap::new()),
            values: RefCell::new(HashMap::new()),
            resolving: RefCell::new(HashSet::new()),
            partial_cache: RefCell::new(HashMap::new()),
            parent: Some(Rc::clone(parent)),
            funcs: Arc::clone(&parent.funcs),
        })
    }

    fn add_raw(&self, name: &str, rhs: String) {
        self.raw.borrow_mut().insert(key(name), rhs);
    }

    /// Look up a parameter's RAW (unevaluated) definition, walking the scope chain.
    /// Used by partial evaluation: when a param can't be folded to a constant
    /// (because its definition transitively depends on a runtime quantity such as
    /// `temper`), we partial-evaluate its definition instead of giving up.
    fn raw_lookup(&self, name: &str) -> Option<String> {
        let k = key(name);
        if let Some(r) = self.raw.borrow().get(&k) {
            return Some(r.clone());
        }
        match &self.parent {
            Some(p) => p.raw_lookup(name),
            None => None,
        }
    }
}

impl Env for Rc<Scope> {
    fn var(&self, name: &str) -> Result<f64, EvalError> {
        let k = key(name);
        if let Some(v) = self.locals.get(&k) {
            return Ok(*v);
        }
        if let Some(v) = self.values.borrow().get(&k) {
            return Ok(*v);
        }
        let raw = self.raw.borrow().get(&k).cloned();
        if let Some(raw) = raw {
            if self.resolving.borrow().contains(&k) {
                return Err(EvalError::Parse(format!("cyclic parameter `{k}`")));
            }
            let cached = self.parsed.borrow().get(&k).cloned();
            let expr = match cached {
                Some(e) => e,
                None => {
                    let e = Arc::new(parse(&raw)?);
                    self.parsed.borrow_mut().insert(k.clone(), Arc::clone(&e));
                    e
                }
            };
            self.resolving.borrow_mut().insert(k.clone());
            let r = eval(&expr, self);
            self.resolving.borrow_mut().remove(&k);
            let v = r?;
            self.values.borrow_mut().insert(k, v);
            return Ok(v);
        }
        match &self.parent {
            Some(p) => p.var(name),
            None => Err(EvalError::UnknownVar(name.to_string())),
        }
    }

    fn call_user(&self, name: &str, args: &[f64]) -> Result<Option<f64>, EvalError> {
        let k = key(name);
        let Some((argnames, body)) = self.funcs.get(&k).cloned() else {
            return Ok(None);
        };
        if argnames.len() != args.len() {
            return Err(EvalError::Arity {
                func: name.to_string(),
                got: args.len(),
            });
        }
        let mut locals = HashMap::new();
        for (n, v) in argnames.iter().zip(args) {
            locals.insert(key(n), *v);
        }
        let child = Scope::child(self, locals);
        Ok(Some(eval(&body, &child)?))
    }
}

/// Result of an expansion run.
pub struct Expanded {
    /// The resolved, flattened card stream.
    pub cards: Vec<String>,
    /// Parameters that could not be resolved and were dropped (device/model falls
    /// back to a DEFAULT). Non-empty means the output may be silently wrong.
    pub drops: Vec<String>,
}

/// A parsed `.subckt` definition.
struct SubcktDef {
    ports: Vec<String>,
    defaults: Vec<(String, String)>,
    body: Vec<String>,
}

/// Split a card's token stream (after the leading name) into leading positional
/// tokens (nodes / subckt name / model) and the trailing `name=value` assignment
/// text. The split point is just before the first top-level `=`'s parameter name.
fn split_positional(rest: &str) -> (Vec<String>, &str) {
    let b = rest.as_bytes();
    let mut depth = 0i32;
    let mut q = 0u8;
    let mut eq = None;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if q != 0 {
            if c == q {
                q = 0;
            }
            i += 1;
            continue;
        }
        match c {
            b'\'' | b'"' => q = c,
            b'(' | b'{' => depth += 1,
            b')' | b'}' => depth -= 1,
            b'=' if depth == 0 => {
                eq = Some(i);
                break;
            }
            _ => {}
        }
        i += 1;
    }
    let split = match eq {
        None => rest.len(),
        Some(e) => {
            let mut j = e;
            while j > 0 && (b[j - 1] as char).is_whitespace() {
                j -= 1;
            }
            while j > 0 && (b[j - 1].is_ascii_alphanumeric() || b[j - 1] == b'_') {
                j -= 1;
            }
            j
        }
    };
    let positional = split_ws_keep_exprs(&rest[..split]);
    (positional, &rest[split..])
}

/// Split on whitespace while keeping a quoted (`'...'`, `"..."`) or braced
/// (`{...}`) expression as ONE token even when it contains spaces. A behavioral
/// value like `'max(min(a,b), c)'` (TSMC resistor models) must survive as a
/// single positional token; plain `split_whitespace()` shatters it at the first
/// embedded space, truncating the equation. Bare parens are deliberately NOT
/// tracked, so `POLY( 2 )` and similar still split exactly as before.
fn split_ws_keep_exprs(s: &str) -> Vec<String> {
    let b = s.as_bytes();
    let n = b.len();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < n {
        while i < n && (b[i] as char).is_whitespace() {
            i += 1;
        }
        if i >= n {
            break;
        }
        let start = i;
        let mut q = 0u8;
        let mut brace = 0i32;
        while i < n {
            let c = b[i];
            if q != 0 {
                if c == q {
                    q = 0;
                }
            } else if c == b'\'' || c == b'"' {
                q = c;
            } else if c == b'{' {
                brace += 1;
            } else if c == b'}' {
                if brace > 0 {
                    brace -= 1;
                }
            } else if brace == 0 && (c as char).is_whitespace() {
                break;
            }
            i += 1;
        }
        toks.push(s[start..i].to_string());
    }
    toks
}

/// Is `s` a single identifier-like token — a string/keyword model value such as
/// an identifier or keyword — rather than an arithmetic expression? Used to tell a
/// literal token (keep verbatim) from a failed computation (drop). A leading digit
/// is excluded so a stray number never counts as a word.
fn is_bare_word(s: &str) -> bool {
    let s = s.trim();
    // A double-quoted string is a literal too: A-device file names such as
    // `file="my-source.txt"`, `input_file="./stim.txt"`.
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        return true;
    }
    let mut cs = s.chars();
    matches!(cs.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && cs.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
}

/// First whitespace-delimited token, lowercased.
fn kw(line: &str) -> String {
    line.split_whitespace().next().unwrap_or("").to_ascii_lowercase()
}

/// Split off the first whitespace-delimited token; returns (token, remainder).
fn split_first(s: &str) -> (&str, &str) {
    let s = s.trim_start();
    match s.find(char::is_whitespace) {
        Some(i) => (&s[..i], &s[i..]),
        None => (s, ""),
    }
}

/// How a device card's positional tokens divide into nodes and controlling-device
/// references.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DevShape {
    /// Leading positional tokens that are node names (renamed `prefix.node`).
    nodes: usize,
    /// Tokens directly after the nodes that name a *controlling device* — renamed
    /// like an instance (`v.prefix.vsen`), not like a node.
    ctrl: usize,
}

/// A bare numeric literal (`1e-6`, `2.0`, `10u`) is a value — never a node and
/// never a model name. This is the test ngspice reaches for in `get_number_terminals`
/// ("AREA may be assumed if we have a token with only digits"), except ngspice
/// implements it as "contains no alpha character", which misfires on `1e-6`
/// because of the `e`. Parsing the token is the same idea done correctly.
fn is_value_token(t: &str) -> bool {
    matches!(parse(t), Ok(Expr::Num(_)))
}

/// Index of the model-name token: the last positional token that is not a value,
/// a trailing flag, or a delimited expression. Everything before it is a node.
///
/// This is ngspice's own stated rule — "MNAME has to contain at least one alpha
/// character" (`inpcom.c::get_number_terminals`, case 'q') — and it is what the
/// per-device parsers do directly, by walking tokens until one resolves as a model
/// (`inp2m.c`/`inp2q.c`/`inp2d.c`: `if (i >= N && INPlookMod(token)) break`).
fn model_pos(positional: &[String]) -> Option<usize> {
    positional.iter().rposition(|t| {
        !matches!(
            t.to_ascii_lowercase().as_str(),
            "off" | "thermal" | "tnodeout"
        ) && !t.starts_with(['{', '\'', '"', '('])
            && !is_value_token(t)
    })
}

/// What `emit_device` must do with a single positional token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    /// A node name — rename `prefix.node`.
    Node,
    /// A controlling device's instance name — rename `v.prefix.vsen`.
    Inst,
    /// The `POLY(n)` group — re-emitted verbatim in ngspice's normalized spelling.
    Poly(usize),
    /// Consume and emit nothing (the redundant HSPICE source-type marker).
    Drop,
}

/// Recognize a `POLY(dim)` group at the head of `toks` → `(dim, tokens_consumed)`.
///
/// ngspice tokenizes parens separately, so the group reaches us in any spelling:
/// `POLY(2)`, `POLY( 2 )`, `POLY (2)`, `POLY ( 2 )`.
///
/// Matching is deliberately CASE-SENSITIVE on exactly `POLY`/`poly`: both readers
/// of this form use `strcmp`, not a case-insensitive compare — `subckt.c`'s
/// `translate()` and `enhtrans.c::get_poly_dimension()`. To ngspice, `Poly(2)` is
/// simply not a POLY, and we match that rather than silently building a working
/// circuit where the reference builds a broken one.
fn parse_poly(toks: &[String]) -> Option<(usize, usize)> {
    let head = toks.first()?;
    let word = head.split_once('(').map_or(head.as_str(), |(w, _)| w);
    if word != "POLY" && word != "poly" {
        return None;
    }
    let mut text = String::new();
    for (n, t) in toks.iter().enumerate().take(4) {
        text.push_str(t);
        if let (Some(o), Some(c)) = (text.find('('), text.find(')')) {
            return Some((text[o + 1..c].trim().parse().ok()?, n + 1));
        }
    }
    None
}

/// Tokenize an XSPICE A-device connection list exactly as ngspice's `MIFgettok`
/// does (`xspice/mif/mifutil.c`):
///
///   * whitespace and `=` `(` `)` `,` are delimiters and are DISCARDED — which is
///     why `%vd(a b)` expands to `%vd a b` with the parens gone;
///   * `<` `>` `[` `]` `~` `%` are single-character tokens;
///   * `"..."` yields the quoted contents, without the quotes;
///   * anything else runs to the next delimiter.
fn mif_tokens(s: &str) -> Vec<String> {
    const DELIM: [char; 4] = ['=', '(', ')', ','];
    let is_delim = |c: char| c.is_whitespace() || DELIM.contains(&c);
    let mut out = Vec::new();
    let mut it = s.chars().peekable();
    loop {
        while it.peek().is_some_and(|&c| is_delim(c)) {
            it.next();
        }
        let Some(&c) = it.peek() else { return out };
        if matches!(c, '<' | '>' | '[' | ']' | '~' | '%') {
            it.next();
            out.push(c.to_string());
        } else if c == '"' {
            it.next();
            let mut t = String::new();
            for ch in it.by_ref() {
                if ch == '"' {
                    break;
                }
                t.push(ch);
            }
            out.push(t);
        } else {
            let mut t = String::new();
            while let Some(&ch) = it.peek() {
                if is_delim(ch) || matches!(ch, '<' | '>' | '[' | ']' | '~' | '%' | '"') {
                    break;
                }
                t.push(ch);
                it.next();
            }
            out.push(t);
        }
    }
}

/// The role of every positional token on a device card.
///
/// `e`/`f`/`g`/`h` get their own layout because ngspice does too: `subckt.c`
/// handles them in a dedicated branch (line 1587) that never calls `numnodes()` —
/// "Control nodes for E and G sources are not counted as they vary in the case of
/// POLY". That branch is:
///   1. 2 output nodes;
///   2. optionally consume an HSPICE source-type marker (`vcvs`/`vccs`/`cccs`/`ccvs`)
///      — redundant with the device letter, so ngspice drops it;
///   3. optionally `POLY(dim)`, else `dim = 1`;
///   4. `dim * numdevs()` controlling tokens — E/G take 2 each as *nodes*,
///      F/H take 1 each as an *instance* name;
///   5. the rest are polynomial coefficients.
///
/// A POLY source is later rewritten into an XSPICE A-device by
/// `ENHtranslate_poly()` (`inp.c:994`), which runs AFTER expansion and before
/// `INPpas1` — so that stays ngspice's job; ours is only to hand it a correctly
/// expanded line.
fn dev_roles(first: u8, positional: &[String]) -> Vec<Role> {
    let f = first.to_ascii_lowercase();
    if !matches!(f, b'e' | b'f' | b'g' | b'h') {
        let s = dev_shape(f, positional);
        let mut roles = vec![Role::Node; s.nodes];
        roles.resize(s.nodes + s.ctrl, Role::Inst);
        return roles;
    }

    let mut roles = vec![Role::Node, Role::Node];
    let mut i = 2;

    let marker = match f {
        b'e' => "vcvs",
        b'g' => "vccs",
        b'f' => "cccs",
        _ => "ccvs",
    };
    if positional.get(i).is_some_and(|t| t.eq_ignore_ascii_case(marker)) {
        roles.push(Role::Drop);
        i += 1;
    }

    let dim = match positional.get(i..).and_then(parse_poly) {
        Some((dim, n)) => {
            roles.push(Role::Poly(dim));
            roles.resize(roles.len() + n - 1, Role::Drop);
            dim
        }
        None => 1,
    };

    // subckt.c::numdevs(): E/G sense 2 nodes per term, F/H sense 1 source per term.
    let (per_term, role) = match f {
        b'e' | b'g' => (2, Role::Node),
        _ => (1, Role::Inst),
    };
    roles.resize(roles.len() + dim * per_term, role);
    roles.truncate(positional.len());
    roles
}

/// Split a device card's positional tokens into nodes / controlling-device names,
/// mirroring the three functions ngspice uses when it renames a subckt body:
///
///   * `frontend/subckt.c::numnodes()` — the node count itself. It overrides
///     `e`/`g`/`w` to 2 and `k` to 0, resolves `x` from the `.subckt` header, and
///     otherwise defers to:
///   * `frontend/inpcom.c::get_number_terminals()` — the general table, including
///     the token scans for `d`/`m`/`q`/`p`/`n`;
///   * `frontend/subckt.c::numdevs()` — the trailing controlling V-source (`f`/`h`/`w`)
///     or coupled-inductor pair (`k`), which are renamed as instances.
///
/// `positional` is the tokens after the instance name, already cut at the first
/// `name=value` — which is exactly where ngspice's scans stop.
///
/// Why this is worth the precision: a node we fail to rename keeps the subckt's
/// *internal* name, so every instance of that subckt silently shorts together on
/// it. There is no error — just a wrong answer. (`t`/`o`/`s`/`y`/`u` were missing
/// from the old guess table entirely, and a 4-node `q` lost its substrate.)
fn dev_shape(first: u8, positional: &[String]) -> DevShape {
    let f = first.to_ascii_lowercase();
    let p = positional.len();

    // subckt.c::numdevs() — controlling sources / coupled inductors.
    let ctrl = match f {
        b'f' | b'h' | b'w' => 1, // one controlling V-source name
        b'k' => 2,               // two inductor instance names
        _ => 0,
    };

    let nodes = match f {
        b'r' | b'c' | b'l' | b'v' | b'i' | b'b' => 2,
        // 2 nodes + a controlling source. get_number_terminals says 3 for `w`,
        // but numnodes() overrides it to 2 and numdevs() takes the third token.
        b'f' | b'h' | b'w' => 2,
        // No nodes at all — just the two inductor names.
        b'k' => 0,
        // numnodes() -> 2 output nodes, numdevs() -> 2 controlling nodes; both are
        // node-translated, so 4 node tokens in a row.
        b'e' | b'g' => 4,
        b'j' | b'u' | b'z' => 3,
        b't' | b'o' | b's' | b'y' => 4,
        // Variable-node devices: nodes run up to the model name.
        //   d: 2, or 3 with a self-heating thermal node
        //   m: 3 (VDMOS d,g,s) .. 7 (B4SOI/B3SOI*)      -- inp2m.c::model_numnodes
        //   q: 3 .. 5 (substrate, then VBIC/hicum2 thermal) -- inp2q.c
        //   n: OSDI, p: coupled lines -- fully variable
        b'd' | b'm' | b'q' | b'n' | b'p' => {
            let (min, max) = match f {
                b'd' => (2, 3),
                b'm' => (3, 7),
                b'q' => (3, 5),
                _ => (1, p),
            };
            model_pos(positional)
                .unwrap_or(min)
                .clamp(min.min(p), max.min(p))
        }
        _ => 0,
    };

    DevShape {
        nodes: nodes.min(p),
        ctrl: ctrl.min(p - nodes.min(p)),
    }
}

/// The subckt expander.
pub struct SubcktExpander {
    /// Subckt definitions. `Arc` so a multi-core run can share them read-only
    /// across worker threads (they are immutable after construction).
    defs: Arc<HashMap<String, SubcktDef>>,
    /// Nodes declared `.global`: shared across the whole hierarchy, so they are
    /// never prefixed during subckt expansion (like ground `0`). `Arc` for the
    /// same reason as `defs`.
    globals: Arc<HashSet<String>>,
    funcs: Arc<FuncMap>,
    /// The global `.param` definitions (name -> raw rhs), as seeded into the root
    /// scope. Kept so a worker thread can rebuild an equivalent fresh root scope
    /// (the scope's memoization caches cannot cross threads, so each worker gets
    /// its own — same values, empty caches).
    global_raw: Arc<HashMap<String, String>>,
    /// GLOBAL statistical params: top-level `.param`s whose value is (transitively)
    /// a statistical draw AND which are referenced by at least one OTHER top-level
    /// `.param`. These represent a SHARED process draw (one per MC run, common to
    /// every device) — e.g. TSMC's `random1_res -> par1_res_rpo -> a1_disres_rpo`
    /// feeding `r_rnodwo_m`. They must NOT be inlined per instance (that gives each
    /// device its own draw -> global variation becomes uncorrelated). Instead they
    /// are emitted ONCE as shared `.param` cards and referenced by NAME everywhere,
    /// so ngspice draws each once and all instances share it. A draw consumed only
    /// inside a `.subckt` body (local mismatch, e.g. `par_res` in `factmis`) is NOT
    /// in this set and stays inlined per instance.
    global_shared: Arc<HashSet<String>>,
    /// The global-shared param definitions in DECK ORDER (name, raw rhs), used to
    /// emit the shared `.param` cards (see `global_shared`).
    global_shared_defs: Arc<Vec<(String, String)>>,
    root: Rc<Scope>,
    out: Vec<String>,
    /// Parameters that could not be resolved and were therefore DROPPED from the
    /// emitted card. A drop is NEVER silent: it means the device/model silently
    /// falls back to a DEFAULT, which has repeatedly produced wrong-but-converging
    /// results (foundry_a `u0` -> dead transistor; bxpressn-1 `v=` -> malformed
    /// B source). Reported by the CLI; `--strict` turns them into an error.
    drops: RefCell<Vec<(String, String)>>,
    /// `.option scale` (default 1) — the GLOBAL instance/model geometry scale.
    ///
    /// Model binning multiplies the X line's DRAWN l/w by this before comparing
    /// against the bins' post-shrink ranges (subckt.c:907 `csl = scale * c->l`).
    /// ngspice reads it with `cp_getvar("scale", ...)`, which `.option scale=X`
    /// sets via `cp_vset` (inp.c:913, hs/spe modes) and which defaults to 1 when
    /// no deck sets it.
    ///
    /// Held separately, NOT looked up through the active scope, because a subckt's
    /// own `scale` PARAM is a different mechanism — the HSPICE element scale, which
    /// scales that subckt's body geometry (see `geo_scale`). foundry_a's
    /// `pch_lvt_mac ... scale='scale_mos_lvt'` (0.9) would shadow the option and
    /// silently rebin every device if this were resolved lexically.
    option_scale: f64,
    /// Run configuration. Independent top-level cards are the fan-out unit for
    /// multi-core expansion (`expand_parallel`); `cfg.effective_cores()` drives it.
    cfg: Config,
    pub stats: SubstStats,
    pub inst_count: usize,
    depth_guard: usize,
    top_cards: Vec<String>,
}

impl SubcktExpander {
    /// Build from the flattened card stream: separate subckt defs, collect global
    /// `.param`s and functions into the root scope.
    pub fn new(lines: &[LogicalLine]) -> Self {
        SubcktExpander::with_config(lines, Config::default())
    }

    /// An expander running under `cfg`.
    pub fn with_config(lines: &[LogicalLine], cfg: Config) -> Self {
        // PSpice line-level conversions (AKO inheritance, d/q positional area
        // factors) run before anything else looks at the cards, as ngspice's
        // pspice_compat does.
        let ps_owned: Vec<LogicalLine>;
        let lines = if cfg.is_pspice() {
            ps_owned = pspice_line_rewrites(lines);
            &ps_owned[..]
        } else {
            lines
        };
        // Behavioral E/G split (every dialect, like inp_compat) — must happen
        // before subckt bodies are captured so the pair expands with ngspice's
        // own names. Gated on a cheap scan: decks without such cards keep the
        // exact same line vector.
        let eg_owned: Vec<LogicalLine>;
        let lines = if lines
            .iter()
            .any(|l| split_eg_value(&l.text).is_some() || split_eg_table(&l.text).is_some())
        {
            eg_owned = eg_value_rewrite(lines);
            &eg_owned[..]
        } else {
            lines
        };
        let mut defs = HashMap::new();
        let mut top: Vec<String> = Vec::new();
        let mut funcs: FuncMap = HashMap::new();

        // 1. split out subckt definitions (track nesting depth).
        let mut i = 0;
        let cards: Vec<&str> = lines.iter().map(|l| l.text.as_str()).collect();
        while i < cards.len() {
            let line = cards[i];
            if kw(line) == ".subckt" {
                let (name, ports, defaults) = parse_subckt_header(line);
                // capture body until matching .ends (respecting nesting)
                let mut body = Vec::new();
                let mut depth = 1;
                i += 1;
                while i < cards.len() && depth > 0 {
                    let bl = cards[i];
                    let k = kw(bl);
                    if k == ".subckt" {
                        depth += 1;
                    } else if k == ".ends" || k == ".eom" {
                        depth -= 1;
                        if depth == 0 {
                            i += 1;
                            break;
                        }
                    }
                    body.push(bl.to_string());
                    i += 1;
                }
                // Pull nested definitions out into scoped entries (`name/inner`)
                // and strip them from the body.
                let path = key(&name);
                let body = extract_nested(body, &path, &mut defs);
                defs.insert(path, SubcktDef { ports, defaults, body });
                continue;
            }
            top.push(line.to_string());
            i += 1;
        }

        // 2. collect global params/functions from top-level .param lines.
        let root_raw: HashMap<String, String> = HashMap::new();
        let mut global_params: Vec<(String, String)> = Vec::new();
        for line in &top {
            if kw(line) == ".param" {
                for a in parse_assignments(&line[".param".len()..]) {
                    match a.args {
                        Some(argnames) => {
                            if let Ok(body) = parse(&a.rhs) {
                                funcs.insert(key(&a.name), (argnames, Arc::new(body)));
                            }
                        }
                        None => global_params.push((a.name, a.rhs)),
                    }
                }
            } else if kw(line) == ".func" {
                // `.func` is ngspice's dedicated user-function spelling, equivalent
                // to `.param name(args)=body`. Without this, a B-source calling one
                // (`v='bar2(17.0)'`) cannot inline it and drops the whole `v=`.
                if let Some((name, argnames, body)) = parse_func_def(&line[".func".len()..]) {
                    if let Ok(b) = parse(&body) {
                        funcs.insert(key(&name), (argnames, Arc::new(b)));
                    }
                }
            }
        }
        // functions can also be defined inside subckts; harvest them globally too.
        for d in defs.values() {
            for line in &d.body {
                if kw(line) == ".param" {
                    for a in parse_assignments(&line[".param".len()..]) {
                        if let Some(argnames) = a.args {
                            if let Ok(body) = parse(&a.rhs) {
                                funcs.entry(key(&a.name)).or_insert((argnames, Arc::new(body)));
                            }
                        }
                    }
                } else if kw(line) == ".func" {
                    if let Some((name, argnames, body)) = parse_func_def(&line[".func".len()..]) {
                        if let Ok(b) = parse(&body) {
                            funcs.entry(key(&name)).or_insert((argnames, Arc::new(b)));
                        }
                    }
                }
            }
        }

        // `.global a b c` cards accumulate across the deck.
        let mut globals: HashSet<String> = HashSet::new();
        for line in &top {
            if kw(line) == ".global" {
                for n in line.split_whitespace().skip(1) {
                    globals.insert(key(n));
                }
            }
        }

        let funcs = Arc::new(funcs);
        let root = Scope::root(Arc::clone(&funcs));
        let _ = root_raw;
        // ngspice built-in parameters, seeded BEFORE the deck's own `.param`s so a
        // deck definition overrides them. `scale` (from `.option scale`, default 1)
        // is referenced by PDK LOD/stress expressions such as
        // `inv_sa='1/(sa*scale+0.5*l*scale)'`; without it the whole chain
        // (fu0_sa -> fu0_lod -> u0) fails to resolve and the param gets dropped.
        root.add_raw("scale", scale_option(&top).unwrap_or_else(|| "1".to_string()));
        // PSpice mode: ngspice's pspice_compat prepends `.param temp='temper'`,
        // `vt`, and `gmin` to the deck (inpcompat.c). PSpice libs reference TEMP
        // in behavioral sources (`VALUE={DC+POL*DRIFT*(TEMP-27)}`); without this
        // the expression cannot resolve and the whole VALUE= is dropped, leaving
        // a malformed E/G card. `temper` is runtime, so anything referencing
        // `temp` stays correctly symbolic.
        if cfg.is_pspice() {
            root.add_raw("temp", "temper".to_string());
            root.add_raw("vt", "(temper + 273.15) * 8.6173303e-5".to_string());
            root.add_raw("gmin", "1e-12".to_string());
        }
        for (n, rhs) in &global_params {
            root.add_raw(n, rhs.clone());
        }
        // Snapshot the root's raw params (scale + globals) so worker threads can
        // rebuild an equivalent fresh root; the caches are intentionally not
        // captured (they are per-thread memoization, rebuilt on demand).
        let global_raw = Arc::new(root.raw.borrow().clone());

        // Classify GLOBAL statistical params (shared process draws) vs local
        // (per-instance) ones (see the `global_shared` field).
        let global_shared = Arc::new(compute_global_shared(&global_raw));
        // Keep the shared defs in DECK ORDER for emitting `.param` cards later.
        let global_shared_defs: Arc<Vec<(String, String)>> = Arc::new(
            global_params
                .iter()
                .filter(|(n, _)| global_shared.contains(&key(n)))
                .cloned()
                .collect(),
        );

        // Resolved once, against the root, where `.option scale` cannot be shadowed
        // by any subckt's own `scale` param.
        let option_scale = root.var("scale").unwrap_or(1.0);

        let exp = SubcktExpander {
            defs: Arc::new(defs),
            globals: Arc::new(globals),
            funcs,
            global_raw,
            global_shared,
            global_shared_defs,
            root: Rc::clone(&root),
            out: Vec::new(),
            drops: RefCell::new(Vec::new()),
            option_scale,
            cfg,
            stats: SubstStats::default(),
            inst_count: 0,
            depth_guard: 0,
            top_cards: top,
        };
        exp
    }

    /// This expander's run configuration (see [`Config`]).
    pub fn config(&self) -> Config {
        self.cfg
    }

    /// Resolve a subckt name to its registry key using LEXICAL scoping: try the
    /// enclosing definition's scope first, then walk outwards, then global.
    /// (`def_path` is the definition path of the subckt currently being expanded,
    /// e.g. `sub1` or `sub1/sub`; empty at top level.)
    fn resolve_def(&self, name: &str, def_path: &str) -> Option<String> {
        let n = key(name);
        let mut p = def_path.to_string();
        loop {
            let cand = if p.is_empty() {
                n.clone()
            } else {
                format!("{p}/{n}")
            };
            if self.defs.contains_key(&cand) {
                return Some(cand);
            }
            if p.is_empty() {
                return None;
            }
            match p.rfind('/') {
                Some(i) => p.truncate(i),
                None => p.clear(),
            }
        }
    }

    fn process_top(&mut self) {
        let top = std::mem::take(&mut self.top_cards);
        let root = Rc::clone(&self.root);
        // Top-level local models (rare) + process.
        let active = resolve_conditionals(&top, &root);
        let local_models = collect_models(&active);
        self.process_lines(&active, &local_models);
    }

    /// Expand a slice of RESOLVED top-level cards into `self.out`. Split out from
    /// `process_top` so a multi-core run can give each worker its own slice —
    /// every card here expands independently EXCEPT a `.control ... .endc` block,
    /// which is stateful and so must lie wholly within one slice (the partitioner
    /// guarantees this). `.if`/`.param`/`.subckt` are already handled: conditionals
    /// were resolved away above, global params live in the root, and defs were
    /// extracted at registration.
    fn process_lines(&mut self, lines: &[String], local_models: &HashSet<String>) {
        let root = Rc::clone(&self.root);
        let empty_map: HashMap<String, String> = HashMap::new();
        // No instance geometry at top level -> no bin pruning (ngspice bins).
        let drop_bins: HashSet<String> = HashSet::new();
        // `.control ... .endc` is an ngspice command script, NOT netlist: emit it
        // verbatim. (Otherwise `let total_count = 0` would be parsed as a device —
        // leading `l` reads as an inductor — and silently mangled.)
        let mut in_control = false;
        for line in lines {
            let k = kw(line);
            if k == ".control" {
                in_control = true;
                self.out.push(line.clone());
                continue;
            }
            if k == ".endc" {
                in_control = false;
                self.out.push(line.clone());
                continue;
            }
            if in_control {
                self.out.push(line.clone());
                continue;
            }
            self.process_card(line, "", &root, &empty_map, local_models, &drop_bins, 1.0, 1.0, "");
        }
    }

    /// A partial-evaluated symbolic result must be KEPT (not folded to its nominal)
    /// if it is genuinely runtime (`has_runtime`) OR if it references a GLOBAL-shared
    /// param by NAME — that name is a deferred shared draw whose value is NOT the
    /// nominal. Without the second clause, a top-level computed param like
    /// `r_rnodwo_m = 131 + 6.5752*a1_disres_rpo` (where `a1_disres_rpo` is now a
    /// by-name global reference) folds to 131, collapsing the global process
    /// variation to zero.
    fn sym_keeps(&self, s: &str) -> bool {
        Self::has_runtime(s) || ident_tokens(s).iter().any(|id| self.global_shared.contains(id))
    }

    /// Build the shared `.param` cards for the global statistical params (see the
    /// `global_shared` field). Each rhs is partial-evaluated in the ROOT scope, so
    /// constants fold, references to OTHER global-shared params stay symbolic (by
    /// name), and the `agauss()` draw itself stays symbolic — leaving ngspice to
    /// draw it ONCE at top level and share it across every instance.
    fn global_shared_cards(&self) -> Vec<String> {
        let nmap: HashMap<String, String> = HashMap::new();
        let locals: HashMap<String, Part> = HashMap::new();
        let mut cards = Vec::with_capacity(self.global_shared_defs.len());
        for (name, rhs) in self.global_shared_defs.iter() {
            let val = match parse(rhs) {
                Ok(e) => part_str(&self.partial(&e, &self.root, &nmap, "", &locals, 0)),
                Err(_) => rhs.clone(),
            };
            cards.push(format!(".param {name}={val}"));
        }
        cards
    }

    /// Run expansion, returning the resolved flat card stream plus any parameters
    /// that had to be dropped (see [`Expanded::drops`]).
    pub fn expand(mut self) -> Expanded {
        let cores = self.cfg.effective_cores();
        // Multi-core is worth the thread setup only when there is enough
        // independent top-level work to divide; below that, run inline.
        if cores > 1 && self.top_cards.len() >= cores * 2 {
            self.expand_parallel(cores);
        } else {
            self.process_top();
        }
        // Dangling-passive removal (opt-in) runs first, so models used only by
        // removed devices fall to the unused-model prune below.
        let out_cards = std::mem::take(&mut self.out);
        let out_cards = if self.cfg.topo_reduce {
            reduce_dangling_passives(out_cards, &self.globals)
        } else {
            out_cards
        };
        // Prune unused models on the FLAT deck (see prune_unused_models): every
        // buried use is an explicit by-name reference by now, so this is sound —
        // and a wrong prune fails LOUDLY in ngspice ("can't find model X") rather
        // than silently using defaults.
        let (cards, pruned) = prune_unused_models(out_cards);
        // PSpice-dialect card-level rewrites, mirroring ngspice's pspice_compat
        // (inpcompat.c). ngspice runs that pass per `.include`d file, but we have
        // inlined every include, so it never fires on our output -- we do it here.
        let mut cards = if self.cfg.is_pspice() {
            pspice_rewrites(cards)
        } else {
            cards
        };
        // PSpice mode: emit the predefined params/funcs ngspice's pspice_compat
        // prepends to the deck. Kept-verbatim top-level `.param` cards may
        // reference them (`.param nz={0.3/(vt*log(1+5.0m/isz))}`); without the
        // definitions the downstream numparam pass hits "Undefined parameter
        // [vt]" and exits fatally. Inserted at position 0: `cards` does NOT
        // include the title (the reader consumed it; the CLI/glue prepend it),
        // so 0 is "right after the title" in the assembled deck. Inserting any
        // later can land INSIDE a leading `.control` block — the deck's first
        // real card may be `.control` — which corrupts the script. Unused
        // definitions are inert.
        if self.cfg.is_pspice() {
            let inject = [
                ".param temp = 'temper'",
                ".param vt = '(temper + 273.15) * 8.6173303e-5'",
                ".param gmin = 1e-12",
                ".func limit(x, a, b) { ternary_fcn(a > b, max(min(x, a), b), max(min(x, b), a)) }",
                ".func pwr(x, a) { pow(x, a) }",
                ".func pwrs(x, a) { sgn(x) * pow(x, a) }",
                ".func stp(x) { u(x) }",
                ".func if(a, b, c) {ternary_fcn( a , b , c )}",
                ".func int(x) { sgn(x)*floor(abs(x)) }",
            ];
            for (k, s) in inject.iter().enumerate() {
                cards.insert(k, s.to_string());
            }
        }
        // Emit the GLOBAL statistical params as SHARED `.param` cards: drawn ONCE
        // by ngspice (top-level, correlated across every device) and referenced by
        // name wherever they were used. Without this, each instance inlines its own
        // `agauss()` and global process variation becomes uncorrelated device-to-
        // device (the exact bug this fixes). Inserted at position 0 for the same
        // reason as the pspice injection above: `cards` excludes the title (the glue
        // prepends it), and a later position could land inside a leading `.control`.
        let shared = self.global_shared_cards();
        for (k, c) in shared.into_iter().enumerate() {
            cards.insert(k, c);
        }
        let drops = self
            .drops
            .into_inner()
            .into_iter()
            .filter(|(ctx, _)| !pruned.contains(&key(ctx)))
            .map(|(_, msg)| msg)
            .collect();
        Expanded { cards, drops }
    }

    /// Multi-core expansion. Resolve conditionals and collect the top-level model
    /// set ONCE (so every worker sees identical inputs), split the resolved cards
    /// into per-worker slices, expand them on scoped threads, and merge in order.
    ///
    /// The result is byte-identical to the single-core path: each top-level card
    /// expands independently of its siblings (global params live in the root,
    /// `.if` is already resolved, and the only cross-card state — a `.control`
    /// block — is kept whole in one slice by the partitioner). Counters like
    /// `inst_count` do not feed emitted text, so merging them is unnecessary for
    /// correctness. Each worker rebuilds its own root scope because the scope's
    /// memoization caches (RefCell) cannot cross threads; the values are the same.
    fn expand_parallel(&mut self, cores: usize) {
        let top = std::mem::take(&mut self.top_cards);
        let active = resolve_conditionals(&top, &self.root);
        let local_models = collect_models(&active);
        let chunks = partition_top(&active, cores);

        // Immutable shared state, borrowed into the worker threads.
        let defs = &self.defs;
        let globals = &self.globals;
        let funcs = &self.funcs;
        let global_raw = &self.global_raw;
        let global_shared = &self.global_shared;
        let global_shared_defs = &self.global_shared_defs;
        let local_models = &local_models;
        let option_scale = self.option_scale;
        let cfg = self.cfg;

        let results: Vec<(Vec<String>, Vec<(String, String)>)> = std::thread::scope(|s| {
            let handles: Vec<_> = chunks
                .into_iter()
                .map(|chunk| {
                    s.spawn(move || {
                        // Fresh per-worker root from the shared raw params.
                        let root = Scope::root(Arc::clone(funcs));
                        for (n, r) in global_raw.iter() {
                            root.add_raw(n, r.clone());
                        }
                        let mut w = SubcktExpander {
                            defs: Arc::clone(defs),
                            globals: Arc::clone(globals),
                            funcs: Arc::clone(funcs),
                            global_raw: Arc::clone(global_raw),
                            global_shared: Arc::clone(global_shared),
                            global_shared_defs: Arc::clone(global_shared_defs),
                            root,
                            out: Vec::new(),
                            drops: RefCell::new(Vec::new()),
                            option_scale,
                            cfg,
                            stats: SubstStats::default(),
                            inst_count: 0,
                            depth_guard: 0,
                            top_cards: Vec::new(),
                        };
                        w.process_lines(&chunk, local_models);
                        (w.out, w.drops.into_inner())
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        // Merge in chunk order -> identical order to a single-core run.
        for (out, drops) in results {
            self.out.extend(out);
            self.drops.get_mut().extend(drops);
        }
    }

    fn process_card(
        &mut self,
        line: &str,
        prefix: &str,
        scope: &Rc<Scope>,
        node_map: &HashMap<String, String>,
        local_models: &HashSet<String>,
        drop_bins: &HashSet<String>,
        mult: f64,
        geo_scale: f64,
        def_path: &str,
    ) {
        let k = kw(line);
        if k == ".param" {
            // Subckt-internal params are already folded into the expanded devices,
            // so drop them. But KEEP top-level `.param` lines: a `.control` block
            // (let/alter/print) or ngspice numparam may reference them by name.
            if prefix.is_empty() {
                self.out.push(line.to_string());
            }
            return;
        }
        if k == ".ends" || k == ".eom" || k == ".subckt" {
            return; // nested defs are extracted at registration; never emit these
        }
        if k == ".model" {
            // Drop non-selected bins of a binned set (pruned per instance geometry),
            // so ngspice bins the device to the single survivor and applies the
            // BSIM L/W/P binning coefficients.
            if let Some(name) = line.split_whitespace().nth(1) {
                if drop_bins.contains(&key(name)) {
                    return;
                }
            }
            self.out.push(self.emit_model(line, prefix, scope, local_models));
            return;
        }
        if k.starts_with(".if") || k.starts_with(".elseif") || k.starts_with(".else") || k.starts_with(".endif") {
            return; // handled by resolve_conditionals
        }
        if k == ".ic" || k == ".nodeset" {
            // Inside a subckt body ngspice renames the `v(node)` arguments
            // during expansion (ports map to the caller's nodes, internals get
            // the instance prefix). Left verbatim, every entry hits "IC on
            // non-existent node" and is silently ignored.
            let renamed = rename_vnode_args(line, node_map, prefix, &self.globals);
            let out = self.subst_exprs(&renamed, scope, node_map, prefix);
            self.out.push(out);
            return;
        }
        if k.starts_with('.') {
            // other dot-cards: partial-substitute, emit as-is
            let out = self.subst_exprs(line, scope, node_map, prefix);
            self.out.push(out);
            return;
        }
        // instance or device
        let first = line.as_bytes()[0].to_ascii_lowercase();
        if first == b'x' {
            self.expand_x(line, prefix, scope, node_map, local_models, mult, def_path);
        } else {
            // A behavioral resistor expands to several cards (B-source + noise
            // B/R/V), returned newline-joined; push each as its own card.
            let emitted = self.emit_device(line, prefix, scope, node_map, local_models, mult, geo_scale);
            if emitted.contains('\n') {
                for card in emitted.split('\n') {
                    self.out.push(card.to_string());
                }
            } else {
                self.out.push(emitted);
            }
        }
    }

    fn expand_x(
        &mut self,
        line: &str,
        prefix: &str,
        scope: &Rc<Scope>,
        node_map: &HashMap<String, String>,
        _local_models: &HashSet<String>,
        mult: f64,
        def_path: &str,
    ) {
        self.inst_count += 1;
        if self.depth_guard > 100 {
            self.out.push(format!("* ngparse: recursion limit at {line}"));
            return;
        }
        let mut parts = line.splitn(2, char::is_whitespace);
        let inst = parts.next().unwrap_or("");
        let rest = parts.next().unwrap_or("");
        // `x1 1 2 sub PARAMS: l=1` -> drop the keyword (see strip_params_kw).
        let rest_owned;
        let rest = if rest.to_ascii_lowercase().contains("params:") {
            rest_owned = strip_params_kw(rest);
            rest_owned.as_str()
        } else {
            rest
        };
        let (positional, assign_str) = split_positional(rest);
        if positional.is_empty() {
            return;
        }
        let subckt = positional.last().unwrap().clone();
        let nodes = &positional[..positional.len() - 1];
        // Lexical resolution: an inner definition of the same name wins over an
        // outer/global one (ngspice scopes subckt defs to their parent).
        let Some(def_key) = self.resolve_def(&subckt, def_path) else {
            // unknown subckt: best-effort emit (partial-substitute params)
            let out = self.subst_exprs(line, scope, node_map, prefix);
            self.out.push(out);
            return;
        };
        let Some(def) = self.defs.get(&def_key) else {
            let out = self.subst_exprs(line, scope, node_map, prefix);
            self.out.push(out);
            return;
        };
        // map the instance's actual nodes into the current namespace
        let actual: Vec<String> = nodes
            .iter()
            .map(|n| map_node_with(n, node_map, prefix, &self.globals))
            .collect();
        // build child scope: overrides evaluated in current scope
        let mut locals = HashMap::new();
        let mut override_raw: Vec<(String, String)> = Vec::new();
        for a in parse_assignments(assign_str) {
            match eval(&parse(&a.rhs).unwrap_or(Expr::Num(0.0)), scope) {
                Ok(v) => {
                    locals.insert(key(&a.name), v);
                }
                Err(_) => override_raw.push((a.name, a.rhs)),
            }
        }
        // instance geometry passed on the X line (used to prune binned models,
        // matching ngspice subckt.c which bins on the X-instance's l/w).
        let xl = locals.get("l").copied();
        let xw = locals.get("w").copied();
        let xnf = locals.get("nf").copied().unwrap_or(1.0);
        // `m` on an X line means one of two things, decided by the definition:
        //  * the subckt DECLARES an `m` parameter (e.g. SMIC `.subckt n12ll ... m=1`
        //    whose device does `m=m`, and whose mismatch uses `geo_fac='1/sqrt(lef*wef*m)'`):
        //    then `m` is an ordinary parameter — bind it and do NOT multiply, or we'd
        //    double-count (and stripping it would break `geo_fac`).
        //  * the subckt does NOT declare `m` (e.g. foundry_a `.subckt pch_lvt_mac ... multi='1'`
        //    whose device does `m=multi`): then `m` is the SPICE instance MULTIPLICITY,
        //    which ngspice multiplies into every device inside; remove it from the
        //    bindings and fold it into the multiplier instead.
        // An outer instance's multiplicity keeps propagating in both cases.
        let declares_m = def.defaults.iter().any(|(n, _)| key(n) == "m");
        let x_m = if declares_m {
            1.0
        } else {
            locals.remove("m").unwrap_or(1.0)
        };
        let child_mult = mult * x_m;
        let child = Scope::child(scope, locals);
        for (n, rhs) in &def.defaults {
            child.add_raw(n, rhs.clone());
        }
        for (n, rhs) in override_raw {
            child.add_raw(&n, rhs);
        }
        // node map: ports -> actual (extra ports without a node are ignored)
        let mut child_map = HashMap::new();
        for (p, a) in def.ports.iter().zip(actual.iter()) {
            child_map.insert(key(p), a.clone());
        }
        let child_prefix = if prefix.is_empty() {
            inst.to_string()
        } else {
            format!("{prefix}.{inst}")
        };

        // pre-add body .param to child scope so forward refs resolve
        for bl in &def.body {
            if kw(bl) == ".param" {
                for a in parse_assignments(&bl[".param".len()..]) {
                    if a.args.is_none() {
                        child.add_raw(&a.name, a.rhs);
                    }
                }
            }
        }
        // HSPICE element scale: a subckt that DECLARES a `scale` parameter scales
        // its OWN body's geometry (ngspice subckt.c::inp_apply_subckt_scale). It is
        // NOT inherited by nested subckts — each wraps its own body — so this is
        // computed per-definition rather than accumulated.
        let geo_scale = if def.defaults.iter().any(|(n, _)| key(n) == "scale") {
            child.var("scale").unwrap_or(1.0)
        } else {
            1.0
        };
        let active = resolve_conditionals(&def.body, &child);
        let local_models = collect_models(&active);
        let bins = collect_bins(&active, &child);
        let drop_bins = bins_to_drop(&bins, xl, xw, xnf, self.option_scale);
        self.depth_guard += 1;
        for bl in &active {
            self.process_card(
                bl,
                &child_prefix,
                &child,
                &child_map,
                &local_models,
                &drop_bins,
                child_mult,
                geo_scale,
                &def_key,
            );
        }
        self.depth_guard -= 1;
    }

    fn emit_model(
        &self,
        line: &str,
        prefix: &str,
        scope: &Rc<Scope>,
        _local_models: &HashSet<String>,
    ) -> String {
        // .model <name> <type> <params...>  (whitespace runs may be doubled).
        // Split by tokens (collapsing whitespace) to get name/type; params get
        // their own value-evaluation pass just like device instance params.
        let after_dot = line.trim_start()[".model".len()..].trim_start();
        let (name, rest) = split_first(after_dot);
        // The type token ends at whitespace OR at `(` — ngspice's INPgetTok
        // (inpgtok.c) treats `(`/`)`/`,`/`=` as separators, so `d(a=1 b=2)` is a
        // type of `d` with a parameter list, exactly like `d (a=1 b=2)`. Splitting
        // on whitespace alone made the type `d(a=1` and silently ate the params.
        let rest = rest.trim_start();
        let tend = rest
            .find(|c: char| c.is_whitespace() || c == '(')
            .unwrap_or(rest.len());
        let (mtype, params) = (&rest[..tend], rest[tend..].trim());
        // model params are often wrapped in `( ... )`
        let (open, inner, close) =
            if params.starts_with('(') && params.ends_with(')') && params.len() >= 2 {
                ("(", &params[1..params.len() - 1], ")")
            } else {
                ("", params, "")
            };
        let newname = if prefix.is_empty() {
            name.to_string()
        } else {
            format!("{prefix}:{name}")
        };
        // Leading bare KEYWORDS carry no `=` and must be kept verbatim: VDMOS
        // polarity is `.model M VDMOS nchan` (nchan/pchan), and there are similar
        // flags elsewhere. eval_assignments only understands `name=value`, so it
        // would drop them — and dropping `nchan` flips the device polarity and the
        // circuit's output by ~30x. Split them off; params start at the first
        // `name=value` token.
        let inner_t = inner.trim();
        let (keywords, param_part) = match inner_t.find('=') {
            Some(eq) => {
                // Back up over any whitespace before `=` (PDK models write
                // `level = 54`), THEN over the parameter name, to find where the
                // first name=value begins. Everything before it is keywords.
                // (Backing up only to the previous whitespace would wrongly treat
                // the name `level` as a keyword when there are spaces around `=`.)
                let name_end = inner_t[..eq].trim_end().len();
                let name_start = inner_t[..name_end]
                    .rfind(char::is_whitespace)
                    .map(|w| w + 1)
                    .unwrap_or(0);
                (inner_t[..name_start].trim_end(), &inner_t[name_start..])
            }
            None => (inner_t, ""), // no assignments at all — all keywords
        };
        let resolved = self.eval_assignments(scope, &HashMap::new(), prefix, param_part, 1.0, 1.0, &newname);
        let body = match (keywords.is_empty(), resolved.is_empty()) {
            (true, _) => resolved,
            (false, true) => keywords.to_string(),
            (false, false) => format!("{keywords} {resolved}"),
        };
        format!(".model {newname} {mtype} {open}{body}{close}")
    }

    fn emit_device(
        &self,
        line: &str,
        prefix: &str,
        scope: &Rc<Scope>,
        node_map: &HashMap<String, String>,
        local_models: &HashSet<String>,
        mult: f64,
        geo_scale: f64,
    ) -> String {
        let mut it = line.splitn(2, char::is_whitespace);
        let inst = it.next().unwrap_or("");
        let rest = it.next().unwrap_or("");
        let first = inst.as_bytes().first().copied().unwrap_or(b'?');
        // XSPICE A-devices have a connection syntax of their own and their own
        // tokenizer; ngspice gives them a dedicated branch, and so do we.
        if first.to_ascii_lowercase() == b'a' {
            return self.emit_a_device(inst, rest, prefix, node_map, local_models);
        }
        // Behavioral resistor `R n1 n2 {eq}` with a runtime equation: expand it to
        // ngspice's own B-source (+ noise B/R/V) form HERE, so the aux devices and
        // internal node take the resistor's LOCAL name (then our normal renaming
        // prefixes them). See emit_behavioral_resistor.
        if first.to_ascii_lowercase() == b'r' {
            if let Some(s) = self.emit_behavioral_resistor(inst, rest, prefix, scope, node_map) {
                return s;
            }
        }
        // E/G-source TABLE form: `E n+ n- TABLE {ctrl} = (x0,y0)(x1,y1)...`, a
        // piecewise-linear VCVS/VCCS common in op-amp models.  The `= (points)`
        // is NOT a name=value assignment, and `TABLE`/`{ctrl}` are not nodes, so
        // the normal positional/assignment split mangles it.  Handle it directly:
        // rename the two output nodes, then let subst_exprs rename the nodes
        // inside the `{ctrl}` expression while leaving TABLE/=/(points) verbatim.
        if matches!(first.to_ascii_lowercase(), b'e' | b'g') {
            let mut t = rest.split_whitespace();
            let (n1, n2, kw3) = (t.next(), t.next(), t.next());
            if let (Some(n1), Some(n2), Some(kw3)) = (n1, n2, kw3) {
                if kw3.eq_ignore_ascii_case("table") {
                    let head = format!(
                        "{} {} {} TABLE",
                        rename_inst(inst, prefix),
                        map_node_with(n1, node_map, prefix, &self.globals),
                        map_node_with(n2, node_map, prefix, &self.globals),
                    );
                    // Everything after the TABLE keyword: the {ctrl} + = + points.
                    let tail_start = rest.to_ascii_lowercase().find("table").unwrap() + 5;
                    let tail = self.subst_exprs(&rest[tail_start..], scope, node_map, prefix);
                    return format!("{head}{tail}");
                }
            }
        }
        let (positional, assign_str) = split_positional(rest);

        let roles = dev_roles(first, &positional);
        let nc = roles.len();
        let mut toks: Vec<String> = Vec::new();
        // Renamed instance name. Inside a subckt, ngspice names an expanded device
        // `<devletter>.<path>.<instname>` so the card still begins with the device
        // type letter (e.g. `m.x1.xmp1.main`) rather than the path's `x`.
        toks.push(if prefix.is_empty() {
            inst.to_string()
        } else {
            format!("{}.{}.{}", (first as char), prefix, inst)
        });
        // Nodes, controlling-device references, and the POLY group.
        //
        // A controlling reference (F/H/W sense a V-source; K names its two
        // inductors) is renamed as an INSTANCE, not a node — inside `x1`, `vsen`
        // becomes `v.x1.vsen` (subckt.c::numdevs + translate_inst_name). Left
        // alone it would resolve to some unrelated top-level device, or to
        // nothing at all.
        for (t, role) in positional.iter().zip(&roles) {
            match *role {
                Role::Node => toks.push(map_node_with(t, node_map, prefix, &self.globals)),
                Role::Inst => toks.push(rename_inst(t, prefix)),
                // Normalized to the exact spelling ngspice writes
                // (`bxx_printf("POLY( %d ) ")`), which is what the downstream
                // ENHtranslate_poly reads back.
                Role::Poly(dim) => toks.push(format!("POLY( {dim} )")),
                Role::Drop => {}
            }
        }
        // tail positional (value / model name):
        //   * subckt-local model name -> scope-rename (`prefix:model`)
        //   * bare token that folds to a constant (R/C/L value param like `prgate`)
        //     -> the number
        //   * delimited ({..}/'..') exprs -> left for subst_exprs (they may span
        //     whitespace once rejoined, e.g. V-source pulse args)
        //   * anything else (global model name, `dc`, `pulse`, node) -> kept
        for t in &positional[nc..] {
            let kt = key(t);
            if local_models.contains(&kt) {
                // A defined model name is kept verbatim, NEVER folded as a value.
                // This matters when the name looks like a number: `1n4002` (a real
                // diode part) parses as 1n = 1e-9, and folding it both corrupts the
                // device and orphans the `.model 1n4002` card (then pruned as
                // "unused"). Inside a subckt the reference is scope-renamed to the
                // BASE name so bin pruning + ngspice's L/W/P binning still apply;
                // at top level it stays as written.
                if prefix.is_empty() {
                    toks.push(t.clone());
                } else {
                    toks.push(format!("{prefix}:{t}"));
                }
            } else if t.contains(['{', '}', '\'', '"']) {
                toks.push(t.clone());
            } else {
                match parse(t) {
                    Ok(e) => match self.partial(&e, scope, node_map, prefix, &HashMap::new(), 0) {
                        Part::Const(v) => toks.push(fmt_num(v)),
                        Part::Sym(_) => toks.push(t.clone()),
                    },
                    Err(_) => toks.push(t.clone()),
                }
            }
        }
        // resolve delimited exprs in the head, partial-aware (folds params, keeps
        // runtime v()/temper, renames nodes inside v()).
        let mut base = self.subst_exprs(&toks.join(" "), scope, node_map, prefix);
        // trailing `name=value` params.
        let assigns = self.eval_assignments(scope, node_map, prefix, assign_str, mult, geo_scale, &toks[0]);
        if !assigns.is_empty() {
            base.push(' ');
            base.push_str(&assigns);
        }
        base
    }

    /// A behavioral resistor `R n1 n2 {eq}` / `'eq'` whose equation references a
    /// runtime quantity (`v()`, `i()`, `temper`, `time`, `hertz`) is turned by
    /// ngspice (inpcom.c) into a B-source, plus a noise B/R/V triple when
    /// `noisy=1`. ngspice does this BEFORE subckt expansion, so the aux devices
    /// carry the resistor's LOCAL name and the internal node has no leading type
    /// letter -- e.g. `b.x1.xr1.br1`, node `x1.xr1.r1_3`. We expand first, so if we
    /// handed ngspice a flat `R.x1.xr1.R1` it would name the internals
    /// `br.x1.xr1.r1` / `r.x1.xr1.r1_3` instead, reordering the matrix and shifting
    /// sensitive analog startups (bandgap: ~16 mV). So we emit ngspice's exact
    /// form here using the local `inst` name and let normal renaming prefix it.
    ///
    /// Returns None for a plain (numeric) resistor or one whose equation folds to a
    /// constant -- those go through the ordinary device path.
    fn emit_behavioral_resistor(
        &self,
        inst: &str,
        rest: &str,
        prefix: &str,
        scope: &Rc<Scope>,
        node_map: &HashMap<String, String>,
    ) -> Option<String> {
        let (positional, assign_str) = split_positional(rest);
        // Need n1, n2, and a value token.
        if positional.len() < 3 {
            return None;
        }
        let value = &positional[2];
        // Only an expression value (braced/quoted) can be behavioral.
        if !value.contains(['{', '\'', '"']) {
            return None;
        }
        // Resolve the equation (folds params, keeps runtime symbolic, renames the
        // nodes inside its v()/i()), then strip the surrounding delimiter.
        let resolved = self.subst_exprs(value, scope, node_map, prefix);
        let eq = resolved.trim();
        let eq = eq
            .strip_prefix('\'')
            .and_then(|s| s.strip_suffix('\''))
            .or_else(|| eq.strip_prefix('{').and_then(|s| s.strip_suffix('}')))
            .unwrap_or(eq)
            .trim();
        // ngspice's b_transformation_wanted: transform only a runtime equation.
        if !Self::has_runtime(eq) {
            return None;
        }
        let n1 = map_node_with(&positional[0], node_map, prefix, &self.globals);
        let n2 = map_node_with(&positional[1], node_map, prefix, &self.globals);
        // tc1/tc2 and m pass through (rare on behavioral R, absent in the PDK); the
        // resolved noisy flag decides the noise triple.
        let assigns = self.eval_assignments(scope, node_map, prefix, &assign_str, 1.0, 1.0, inst);
        let (tc, m, noisy) = split_resistor_params(&assigns);

        // Device name: `<letter>.<prefix>.<name>` (the normal renaming), NODE name:
        // `<prefix>.<name>` (no leading letter) -- matching ngspice post-expansion.
        let dev = |letter: char, name: &str| -> String {
            if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{letter}.{prefix}.{name}")
            }
        };
        let node = |name: &str| -> String {
            if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}.{name}")
            }
        };
        let mut cards = vec![format!(
            "{} {n1} {n2} i=v({n1},{n2})/({eq}){tc}{m} reciproctc=1 reciprocm=0",
            dev('b', &format!("b{inst}"))
        )];
        if noisy {
            let vsense = dev('v', &format!("v{inst}_3"));
            let n3 = node(&format!("{inst}_3"));
            cards.push(format!(
                "{} {n1} {n2} i=i({vsense})/sqrt({eq})",
                dev('b', &format!("b{inst}_1"))
            ));
            cards.push(format!("{} {n3} 0 1.0{tc}", dev('r', &format!("r{inst}_2"))));
            cards.push(format!("{vsense} {n3} 0 0"));
        }
        Some(cards.join("\n"))
    }

    /// Partial evaluation of an expression AST: fold parameter/constant subtrees
    /// to numbers, inline user functions, and keep runtime references — `v()`,
    /// `i()`, `temper`, `time`, `hertz`, and any identifier that isn't a defined
    /// parameter — symbolic. Node names inside `v()`/`i()` are renamed via the
    /// current `nmap`/`prefix`. `locals` binds inlined function arguments.
    fn partial(
        &self,
        e: &Expr,
        scope: &Rc<Scope>,
        nmap: &HashMap<String, String>,
        prefix: &str,
        locals: &HashMap<String, Part>,
        depth: usize,
    ) -> Part {
        match e {
            Expr::Num(n) => Part::Const(*n),
            // A string is a `table_param` filename, never a number. It can only be
            // consumed by the `table_param` fold below; if that fold fails, the
            // call stays symbolic and the string is re-emitted as written.
            Expr::Str(s) => Part::Sym(format!("\"{s}\"")),
            Expr::Var(name) => {
                let k = key(name);
                if let Some(p) = locals.get(&k) {
                    return p.clone();
                }
                // GLOBAL statistical param: a SHARED process draw. Reference it by
                // NAME rather than inlining its (statistical) definition, so every
                // instance that uses it points at the SAME `.param` card and ngspice
                // draws it once per run (correlated across devices). Inlining here is
                // exactly what made global MC uncorrelated. (A function-arg local of
                // the same name, handled just above, still wins.) See `global_shared`.
                if self.global_shared.contains(&k) {
                    return Part::Sym(name.clone());
                }
                // Memoized verdict (see Scope::partial_cache). Skipped whenever
                // function-arg locals are live: a raw definition partial-evaluated
                // under non-empty `locals` can bind identifiers to those args, so
                // its result is not a property of the scope alone.
                if locals.is_empty() {
                    if let Some(p) = scope.partial_cache.borrow().get(&k) {
                        return p.clone();
                    }
                }
                let out = match scope.var(name) {
                    Ok(v) => {
                        // A parameter whose definition is (transitively) a
                        // statistical draw must not fold to its nominal — that
                        // would collapse the MC distribution just as surely as
                        // folding the call directly.  A textual `has_runtime` on the
                        // raw definition is not enough: a PDK mismatch param reads
                        // `dvth='avth*geo_fac*sigma_b*mismatchflag'`, which mentions
                        // the draw only by NAME (`sigma_b`, itself `=agauss(...)`
                        // globally) — the literal `agauss(` is one level down.  So
                        // partial-evaluate the raw definition (which resolves
                        // `sigma_b`→`agauss(...)`) and, if the RESOLVED form is
                        // genuinely runtime, keep it symbolic instead of folding to
                        // the nominal `v`.  Folding it collapses the MC distribution
                        // to a single point (σ=0), the exact bug this guards.
                        let mut out = Part::Const(v);
                        if depth < 60 {
                            if let Some(raw) = scope.raw_lookup(name) {
                                if let Ok(e) = parse(&raw) {
                                    let p = self.partial(
                                        &e, scope, nmap, prefix, locals, depth + 1,
                                    );
                                    if matches!(&p, Part::Sym(s) if self.sym_keeps(s)) {
                                        out = p;
                                    }
                                }
                            }
                        }
                        out
                    }
                    Err(_e) => {
                        // Full evaluation failed — typically because the definition
                        // transitively depends on a runtime quantity (`temper`,
                        // `v()`, ...). Don't give up: partial-evaluate the param's
                        // own definition so the constant parts fold and the runtime
                        // parts survive symbolically (PDK LOD/stress params like
                        // `fu0_lod` are temperature-dependent and MUST stay symbolic).
                        // genuinely undefined (or too deep) -> keep symbolic
                        let mut out = Part::Sym(name.clone());
                        if depth < 60 {
                            if let Some(raw) = scope.raw_lookup(name) {
                                if let Ok(e) = parse(&raw) {
                                    out = self.partial(&e, scope, nmap, prefix, locals, depth + 1);
                                }
                            }
                        }
                        out
                    }
                };
                if locals.is_empty() {
                    scope.partial_cache.borrow_mut().insert(k, out.clone());
                }
                out
            }
            Expr::Unary(op, x) => {
                match self.partial(x, scope, nmap, prefix, locals, depth) {
                    Part::Const(v) => Part::Const(match op {
                        UnOp::Neg => -v,
                        UnOp::Pos => v,
                        UnOp::Not => if v == 0.0 { 1.0 } else { 0.0 },
                    }),
                    Part::Sym(s) => {
                        let o = match op { UnOp::Neg => "-", UnOp::Pos => "+", UnOp::Not => "!" };
                        Part::Sym(format!("{o}({s})"))
                    }
                }
            }
            Expr::Binary(op, l, r) => {
                let pl = self.partial(l, scope, nmap, prefix, locals, depth);
                let pr = self.partial(r, scope, nmap, prefix, locals, depth);
                if let (Part::Const(a), Part::Const(b)) = (&pl, &pr) {
                    // reuse the evaluator's semantics for a folded binary op
                    if let Ok(v) = eval(
                        &Expr::Binary(*op, Box::new(Expr::Num(*a)), Box::new(Expr::Num(*b))),
                        scope,
                    ) {
                        return Part::Const(v);
                    }
                }
                // A runtime/statistical term multiplied by a LITERAL 0 folds to 0.
                // A PDK mismatch param reads `...*sigma_b*mismatchflag`; with the
                // flag OFF (0) the whole term is nominal, and ngspice emits an
                // identical (unvaried) model for it. Without this fold ngparse would
                // keep a live `agauss(...)*0` — value-correct (finite draw * 0 == 0)
                // but it still DRAWS at runtime, needlessly perturbing the shared MC
                // PRNG stream and leaving the model spuriously "varied". Only an
                // exact literal-0 factor folds; a symbolic 0 is left alone (it could
                // carry a non-finite value where x*0 != 0).
                if matches!(op, BinOp::Mul)
                    && (matches!(&pl, Part::Const(z) if *z == 0.0)
                        || matches!(&pr, Part::Const(z) if *z == 0.0))
                {
                    return Part::Const(0.0);
                }
                // numparam's evaluator rejects a binary operator immediately
                // followed by a unary sign (`...e0+-(agauss(...))`, reported as
                // "wrongly determined negation"). Our right operand can start
                // with `-`/`+` — a Unary Neg renders `-(...)` and a negative
                // constant renders `-1.2e0`. Parenthesize such an operand so the
                // sign lands in a valid unary context (`... + (-(...))`) instead
                // of butting against the binary op. A leading-sign LEFT operand
                // is already safe: it sits right after the group's `(`.
                let rs = part_str(&pr);
                let rs = if rs.starts_with('-') || rs.starts_with('+') {
                    format!("({rs})")
                } else {
                    rs
                };
                Part::Sym(format!("({}{}{})", part_str(&pl), binop_str(*op), rs))
            }
            Expr::Ternary(c, t, f) => match self.partial(c, scope, nmap, prefix, locals, depth) {
                Part::Const(cv) => {
                    if cv != 0.0 {
                        self.partial(t, scope, nmap, prefix, locals, depth)
                    } else {
                        self.partial(f, scope, nmap, prefix, locals, depth)
                    }
                }
                Part::Sym(cs) => {
                    let pt = part_str(&self.partial(t, scope, nmap, prefix, locals, depth));
                    let pf = part_str(&self.partial(f, scope, nmap, prefix, locals, depth));
                    Part::Sym(format!("(({cs}) ? ({pt}) : ({pf}))"))
                }
            },
            Expr::Call(name, args) => {
                let lname = key(name);
                // node-voltage / branch-current refs: keep symbolic, rename nodes.
                if lname == "v" || lname == "i" {
                    // `i(...)` measures current THROUGH A DEVICE, so its argument
                    // is a device/instance name, renamed with the device-letter
                    // prefix (`i(rs2)` -> `i(r.x1.rs2)`), exactly like an F/H
                    // controlling source.  `v(...)` is a NODE, renamed as a node.
                    // Getting i() wrong left it pointing at a nonexistent node and
                    // ngspice failed with "unknown controlling source".
                    let is_i = lname == "i";
                    let rn: Vec<String> = args
                        .iter()
                        .map(|a| match a {
                            Expr::Var(n) if is_i => rename_inst(n, prefix),
                            Expr::Var(n) => map_node_with(n, nmap, prefix, &self.globals),
                            // A numeric node name — `v(1)` — parses as a Num. It is
                            // a node NAME, not a value: keep it (renamed), never
                            // fold it. `v(1)` folded to `v(1.0e0)` references a
                            // DIFFERENT, nonexistent node (ngspice matches nodes by
                            // string), which broke a B-source's operating point.
                            Expr::Num(x) if x.fract() == 0.0 && x.is_finite() && *x >= 0.0 => {
                                map_node_with(&format!("{}", *x as i64), nmap, prefix, &self.globals)
                            }
                            other => part_str(&self.partial(other, scope, nmap, prefix, locals, depth)),
                        })
                        .collect();
                    return Part::Sym(format!("{lname}({})", rn.join(",")));
                }
                let pargs: Vec<Part> = args
                    .iter()
                    .map(|a| self.partial(a, scope, nmap, prefix, locals, depth))
                    .collect();
                // user function: inline body with args bound (partial), even when
                // some args are symbolic (e.g. tcoef(temper)).
                if depth < 60 {
                    if let Some((argnames, body)) = self.funcs.get(&lname).cloned() {
                        if argnames.len() == pargs.len() {
                            let mut loc = HashMap::new();
                            for (an, pv) in argnames.iter().zip(&pargs) {
                                loc.insert(key(an), pv.clone());
                            }
                            return self.partial(&body, scope, nmap, prefix, &loc, depth + 1);
                        }
                    }
                }
                // PSpice `LIMIT(x,lo,hi)` (3 args) is a CLAMP, not HSPICE's 2-arg
                // MC distribution. ngspice handles it via pspice_compat's injected
                // `.func limit(x,a,b) {ternary_fcn(a>b, max(min(x,a),b),
                // max(min(x,b),a))}`, expanded by numparam before the B-source
                // parser (which has no `limit`). That injection never reaches our
                // flat deck, so emit the same form here; fold when all-const.
                if self.cfg.is_pspice() && lname == "limit" && pargs.len() == 3 {
                    if let [Part::Const(x), Part::Const(a), Part::Const(b)] = pargs[..] {
                        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                        return Part::Const(x.max(lo).min(hi));
                    }
                    let ss: Vec<String> = pargs.iter().map(part_str).collect();
                    let (x, a, b) = (&ss[0], &ss[1], &ss[2]);
                    return Part::Sym(format!(
                        "ternary_fcn({a}>{b},max(min({x},{a}),{b}),max(min({x},{b}),{a}))"
                    ));
                }
                // Statistical draws are NEVER folded — ngspice must draw a fresh
                // value per MC run. Resolve the arguments (so the distribution
                // parameters are numeric) but keep the call itself symbolic.
                if matches!(
                    lname.as_str(),
                    "agauss" | "gauss" | "aunif" | "unif" | "limit"
                ) {
                    let ss: Vec<String> = pargs.iter().map(part_str).collect();
                    return Part::Sym(format!("{lname}({})", ss.join(",")));
                }
                // builtin: fold if all args are constant.
                if pargs.iter().all(|p| matches!(p, Part::Const(_))) {
                    let vals: Vec<f64> = pargs
                        .iter()
                        .map(|p| if let Part::Const(v) = p { *v } else { 0.0 })
                        .collect();
                    if let Some(Ok(v)) = eval_builtin(&lname, &vals) {
                        return Part::Const(v);
                    }
                }
                let ss: Vec<String> = pargs.iter().map(part_str).collect();
                // PSpice/HSPICE `if(cond,a,b)` is the ternary selector. ngspice's
                // behavioral parser has no `if()` function -- its native name is
                // `ternary_fcn` (exactly what ngspice's own pspice-compat rewrites
                // `if` to). Emit the native name so the source parses on its own,
                // instead of relying on compat-mode injection that ngparse's
                // expanded deck may no longer trigger. `if` is the ternary in every
                // dialect, so this one is unconditional.
                if lname == "if" && ss.len() == 3 {
                    return Part::Sym(format!("ternary_fcn({})", ss.join(",")));
                }
                // The remaining PSpice behavioral functions are injected by ngspice
                // only via pspice_compat (`.func pwr/pwrs/stp/int`), which never
                // reaches our flat deck. In PSpice mode emit their native
                // equivalents directly, matching inpcompat.c exactly.
                if self.cfg.is_pspice() {
                    match (lname.as_str(), ss.len()) {
                        // pwr(x,a) -> pow(x,a)
                        ("pwr", 2) => return Part::Sym(format!("pow({},{})", ss[0], ss[1])),
                        // pwrs(x,a) -> sgn(x) * pow(x,a)
                        ("pwrs", 2) => {
                            return Part::Sym(format!("(sgn({0})*pow({0},{1}))", ss[0], ss[1]))
                        }
                        // stp(x) -> u(x)
                        ("stp", 1) => return Part::Sym(format!("u({})", ss[0])),
                        // int(x) -> sgn(x) * floor(abs(x))
                        ("int", 1) => {
                            return Part::Sym(format!("(sgn({0})*floor(abs({0})))", ss[0]))
                        }
                        _ => {}
                    }
                }
                Part::Sym(format!("{lname}({})", ss.join(",")))
            }
        }
    }

    /// Whether a symbolic string genuinely references runtime quantities (and so
    /// must be preserved for the simulator) vs. being merely an unresolved
    /// parameter (which we drop as a stopgap, e.g. `table_param`).
    fn has_runtime(s: &str) -> bool {
        let low = s.to_ascii_lowercase();
        low.contains("v(")
            || low.contains("i(")
            || low.contains("temper")
            || low.contains("time")
            || low.contains("hertz")
            // ngspice B-source functions that are inherently time-dependent and
            // can never be folded (inpptree.c): keep them for the simulator.
            || low.contains("ddt(")
            || low.contains("sdt(")
            || low.contains("pwl(")
            // Statistical draws MUST reach ngspice unfolded: it draws a fresh
            // value per Monte Carlo iteration, and that per-run variation IS the
            // point of MC. Folding them to nominal would collapse the whole
            // distribution to a point and make ngparse useless for MC. Kept
            // symbolic here (args still resolved) so ngspice's PRNG does the draw,
            // giving the same distribution as its own parser.
            || low.contains("agauss(")
            || low.contains("gauss(")   // also matches agauss (harmless)
            || low.contains("aunif(")
            || low.contains("unif(")    // also matches aunif (harmless)
            || low.contains("limit(")
    }

    /// Partial-substitute every `{..}`/`'..'` span in a line.
    fn subst_exprs(
        &self,
        line: &str,
        scope: &Rc<Scope>,
        nmap: &HashMap<String, String>,
        prefix: &str,
    ) -> String {
        let b = line.as_bytes();
        let mut out = String::with_capacity(line.len());
        let mut i = 0;
        while i < b.len() {
            let c = b[i];
            if c == b'{' || c == b'\'' {
                let close = if c == b'{' { b'}' } else { b'\'' };
                let start = i + 1;
                let mut k = start;
                let mut depth = 1;
                while k < b.len() {
                    if c == b'{' && b[k] == b'{' {
                        depth += 1;
                    } else if b[k] == close {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    k += 1;
                }
                let inner = &line[start..k.min(b.len())];
                match parse(inner) {
                    Ok(e) => match self.partial(&e, scope, nmap, prefix, &HashMap::new(), 0) {
                        Part::Const(v) => out.push_str(&fmt_num(v)),
                        Part::Sym(s) => {
                            out.push(c as char);
                            out.push_str(&s);
                            out.push(close as char);
                        }
                    },
                    Err(_) => {
                        out.push(c as char);
                        out.push_str(inner);
                        out.push(close as char);
                    }
                }
                i = (k + 1).min(b.len());
                continue;
            }
            out.push(c as char);
            i += 1;
        }
        out
    }

    /// Expand an XSPICE A-device (code model) card.
    ///
    /// ngspice gives these their own branch — subckt.c:1504, *"process A devices
    /// specially ... since they have a more involved and variable length node
    /// syntax"* — driven by a one-token look-ahead so the LAST token is always the
    /// model name. Per connection token:
    ///
    ///   * `[` `]` `~` — emitted verbatim (vector brackets, inversion);
    ///   * `%` — the following word is a port type (`vd`, `id`, `vnam`, …) and is
    ///     emitted glued to it as `%vd`. A port type of `vnam` means the NEXT token
    ///     names a V-source, so it is renamed as an INSTANCE, not a node;
    ///   * anything else — a node name (`null` and `0` pass through, being globals);
    ///   * the trailing model name is subckt-scoped (`prefix:model`), which ngspice
    ///     does separately in `devmodtranslate` (subckt.c:2105 — *"the name of the
    ///     model is always last"*).
    ///
    /// Unlike the other device branches this one never param-substitutes: ngspice's
    /// `case 'a'` consumes the whole line itself and never calls `finishLine`.
    fn emit_a_device(
        &self,
        inst: &str,
        rest: &str,
        prefix: &str,
        node_map: &HashMap<String, String>,
        local_models: &HashSet<String>,
    ) -> String {
        let toks = mif_tokens(rest);

        let mut out: Vec<String> = vec![if prefix.is_empty() {
            inst.to_string()
        } else {
            format!("{}.{}.{}", inst.chars().next().unwrap_or('a'), prefix, inst)
        }];

        let n = toks.len();
        let mut got_vnam = false;
        let mut i = 0;
        // Every token but the last is a connection; the last one is the model.
        while i + 1 < n {
            match toks[i].as_str() {
                "[" | "]" | "~" => {
                    out.push(toks[i].clone());
                    i += 1;
                }
                "%" => {
                    // NB: ngspice clears got_vnam after ONE translated token, so in
                    // `%vnam [ v1 v2 ]` only v1 is instance-renamed and v2 falls
                    // through to node renaming. Faithfully reproduced.
                    let pt = &toks[i + 1];
                    got_vnam = pt.eq_ignore_ascii_case("vnam");
                    out.push(format!("%{pt}"));
                    i += 2;
                }
                t => {
                    out.push(if got_vnam {
                        got_vnam = false;
                        rename_inst(t, prefix)
                    } else {
                        map_node_with(t, node_map, prefix, &self.globals)
                    });
                    i += 1;
                }
            }
        }
        if let Some(m) = toks.last() {
            out.push(if local_models.contains(&key(m)) && !prefix.is_empty() {
                format!("{prefix}:{m}")
            } else {
                m.clone()
            });
        }
        out.join(" ")
    }

    /// Evaluate `name=value` device/instance/model parameters, partial-aware.
    fn eval_assignments(
        &self,
        scope: &Rc<Scope>,
        nmap: &HashMap<String, String>,
        prefix: &str,
        assign_str: &str,
        mult: f64,
        geo_scale: f64,
        context: &str,
    ) -> String {
        let mut out: Vec<String> = Vec::new();
        let mut saw_m = false;
        for a in parse_assignments(assign_str) {
            let kn = key(&a.name);
            // An RHS that is not a numeric expression is a literal token — a
            // version string like `version=3.3.0`, or a keyword. ngspice passes
            // such values straight to the device/model parser, so we keep them
            // VERBATIM rather than dropping them (which would silently default the
            // parameter). Genuinely malformed text still surfaces loudly, as an
            // ngspice parse error, not a silent ngparse default. (The trailing-comma
            // bug that once justified dropping here is fixed upstream in
            // parse_assignments, so a `,` no longer reaches this point.)
            let e = match parse(&a.rhs) {
                Ok(e) => e,
                Err(_) => {
                    out.push(format!("{}={}", a.name, a.rhs));
                    continue;
                }
            };
            if kn == "m" {
                saw_m = true;
            }
            match self.partial(&e, scope, nmap, prefix, &HashMap::new(), 0) {
                Part::Const(v) => {
                    // `m` picks up the accumulated subckt-instance multiplicity;
                    // geometry picks up the enclosing subckt's HSPICE element scale.
                    let v = if kn == "m" {
                        v * mult
                    } else if let Some(p) = geo_power(&kn) {
                        v * geo_scale.powi(p)
                    } else {
                        v
                    };
                    out.push(format!("{}={}", a.name, fmt_num(v)));
                }
                Part::Sym(s) => {
                    // behavioral value OR a reference to a shared global draw -> keep
                    // braced; pure-unresolved param -> drop. (`sym_keeps` adds the
                    // global-shared case: a param like diode `area=...a1_disres_rpo...`
                    // references a by-name shared draw that ngspice resolves via the
                    // emitted `.param`, so it must NOT be dropped as "unresolved".)
                    if self.sym_keeps(&s) {
                        // scale/multiply symbolically so runtime exprs stay correct
                        let s = if kn == "m" && mult != 1.0 {
                            format!("({s})*{}", fmt_num(mult))
                        } else if let Some(p) = geo_power(&kn) {
                            if geo_scale != 1.0 {
                                format!("({s})*{}", fmt_num(geo_scale.powi(p)))
                            } else {
                                s
                            }
                        } else {
                            s
                        };
                        out.push(format!("{}={{{}}}", a.name, s));
                    } else if is_bare_word(&s) {
                        // A single unresolved identifier is a literal token, not a
                        // failed computation: a model string value (`mfg=acme_corp`),
                        // a keyword (`fraction=false`), a type name. ngspice passes
                        // these straight through, so keep it VERBATIM. If it is in
                        // fact a mistyped parameter name, ngspice reports the
                        // unknown parameter — loud, not a silent default.
                        out.push(format!("{}={}", a.name, a.rhs));
                    } else {
                        // A genuine EXPRESSION that could not resolve (references an
                        // undefined parameter inside arithmetic): the param is
                        // DROPPED and the device/model silently falls back to its
                        // DEFAULT. Never let that pass unreported — record it so the
                        // caller can warn (or fail under --strict).
                        let scope_name = if prefix.is_empty() { "<top>" } else { prefix };
                        self.drops.borrow_mut().push((
                            context.to_string(),
                            format!(
                                "{scope_name} {context}: {}={:?} (unresolved: {:?})",
                                a.name, a.rhs, s
                            ),
                        ));
                        if std::env::var_os("NGPARSE_DEBUG_DROP").is_some() {
                            eprintln!(
                                "ngparse DROP: prefix={prefix:?} {}={:?} -> unresolved {:?}",
                                a.name, a.rhs, s
                            );
                        }
                    }
                }
            }
        }
        // A device with no explicit `m` still inherits the instance multiplicity.
        if !saw_m && mult != 1.0 {
            out.push(format!("m={}", fmt_num(mult)));
        }
        out.join(" ")
    }
}

/// Power of the enclosing subckt's `scale` that a MOSFET geometry parameter takes
/// (ngspice subckt.c::inp_apply_subckt_scale): lengths/perimeters/spacings by
/// `scale`, areas by `scale^2`. `m`/`nf` (multipliers) and `nrd/nrs/sca-scc`
/// (dimensionless) are deliberately absent — they are never scaled.
fn geo_power(name: &str) -> Option<i32> {
    match name {
        "w" | "l" | "pd" | "ps" | "sa" | "sb" | "sc" | "sd" => Some(1),
        "ad" | "as" => Some(2),
        _ => None,
    }
}


/// Rename a device/instance name for a subckt body: `rs2` in `x1` -> `r.x1.rs2`,
/// keeping ngspice's `<device-letter>.<path>.<instance>` form.  At top level
/// (empty prefix) the name is unchanged.  Used for F/H/W/K controlling sources
/// and for the argument of `i()`.
fn rename_inst(name: &str, prefix: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{}.{}.{}", name.chars().next().unwrap_or('?'), prefix, name)
    }
}

/// Map a node name through the port map, else prefix it (internal node). Ground
/// (`0`) is global and never renamed.
/// Rename the node names inside every `v(...)` group of an `.ic`/`.nodeset`
/// card (the only node references such cards carry). Text outside `v(...)`
/// is preserved untouched; a comma-separated argument renames each part.
fn rename_vnode_args(
    line: &str,
    node_map: &HashMap<String, String>,
    prefix: &str,
    globals: &HashSet<String>,
) -> String {
    let b = line.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < b.len() {
        let is_v = (b[i] == b'v' || b[i] == b'V')
            && i + 1 < b.len()
            && b[i + 1] == b'('
            && (i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_'));
        if !is_v {
            out.push(b[i] as char);
            i += 1;
            continue;
        }
        let start = i + 2;
        let close = match line[start..].find(')') {
            Some(p) => start + p,
            None => {
                out.push_str(&line[i..]);
                break;
            }
        };
        let renamed: Vec<String> = line[start..close]
            .split(',')
            .map(|n| map_node_with(n.trim(), node_map, prefix, globals))
            .collect();
        out.push(b[i] as char);
        out.push('(');
        out.push_str(&renamed.join(","));
        out.push(')');
        i = close + 1;
    }
    out
}

fn map_node_with(
    n: &str,
    node_map: &HashMap<String, String>,
    prefix: &str,
    globals: &HashSet<String>,
) -> String {
    // Ground, `null`, and `.global` nodes are hierarchy-wide: never renamed.
    // ngspice seeds its global-node table with exactly these two before adding
    // the user's `.global` names (subckt.c::collect_global_nodes):
    //     nghash_insert(glonodes, "0", ...);
    //     nghash_insert(glonodes, "null", ...);   /* #ifdef XSPICE */
    // `null` marks an unconnected XSPICE port (`adiv2 d clk NULL NULL NULL q dff`);
    // renaming it to `x1.null` would invent a real node per subckt instance.
    if n == "0" || n.eq_ignore_ascii_case("null") || globals.contains(&key(n)) {
        return n.to_string();
    }
    if let Some(a) = node_map.get(&key(n)) {
        return a.clone();
    }
    if prefix.is_empty() {
        n.to_string()
    } else {
        format!("{prefix}.{n}")
    }
}

/// Collect the set of model names (lowercased) defined by active `.model` cards.
/// For length-binned models (`nfet.0`, `nfet.1`, …) the base name (`nfet`) is
/// also registered, so a device that references the base gets scoped the same
/// way and ngspice's bin matching still finds `prefix:nfet.0` … from `prefix:nfet`.
fn collect_models(active: &[String]) -> HashSet<String> {
    let mut s = HashSet::new();
    for line in active {
        if kw(line) == ".model" {
            if let Some(name) = line.split_whitespace().nth(1) {
                let k = key(name);
                // strip a trailing `.<digits>` bin suffix to get the base name
                if let Some(dot) = k.rfind('.') {
                    if k[dot + 1..].chars().all(|c| c.is_ascii_digit()) && dot > 0 {
                        s.insert(k[..dot].to_string());
                    }
                }
                s.insert(k);
            }
        }
    }
    s
}

/// A length/width bin of a binned model set.
struct BinDef {
    name: String, // lowercased full model name, e.g. `nfet.1`
    lmin: f64,
    lmax: f64,
    wmin: f64,
    wmax: f64,
}

/// Extract a named parameter's value from a `.model` card's params, evaluated in
/// `scope`. Returns None if absent/unevaluable.
fn model_param(params: &[Assign], name: &str, scope: &Rc<Scope>) -> Option<f64> {
    let k = key(name);
    for a in params {
        if key(&a.name) == k {
            return parse(&a.rhs).ok().and_then(|e| eval(&e, scope).ok());
        }
    }
    None
}

/// Group binned `.model <base>.<N>` cards among the active lines by base name,
/// resolving each bin's `lmin/lmax/wmin/wmax` in `scope`. Only sets whose cards
/// carry all four bounds are recorded (unbinned models are ignored).
fn collect_bins(active: &[String], scope: &Rc<Scope>) -> HashMap<String, Vec<BinDef>> {
    let mut m: HashMap<String, Vec<BinDef>> = HashMap::new();
    for line in active {
        if kw(line) != ".model" {
            continue;
        }
        let mut tok = line.split_whitespace();
        let _dot = tok.next();
        let name = match tok.next() {
            Some(n) => key(n),
            None => continue,
        };
        // base = name minus a trailing `.<digits>` bin suffix
        let base = match name.rfind('.') {
            Some(d) if d > 0 && name[d + 1..].chars().all(|c| c.is_ascii_digit()) => {
                name[..d].to_string()
            }
            _ => continue,
        };
        // params start after `.model <name> <type>`
        let after = line.trim_start()[".model".len()..].trim_start();
        let (_n, rest) = split_first(after);
        let (_ty, params_txt) = split_first(rest.trim_start());
        let params_txt = params_txt.trim().trim_start_matches('(').trim_end_matches(')');
        let params = parse_assignments(params_txt);
        if let (Some(lmin), Some(lmax), Some(wmin), Some(wmax)) = (
            model_param(&params, "lmin", scope),
            model_param(&params, "lmax", scope),
            model_param(&params, "wmin", scope),
            model_param(&params, "wmax", scope),
        ) {
            m.entry(base).or_default().push(BinDef {
                name,
                lmin,
                lmax,
                wmin,
                wmax,
            });
        }
    }
    m
}

/// From a behavioral resistor's resolved `name=value` tail, pull out the
/// temperature-coefficient string (` tc1=.. tc2=..`), the multiplier (` m=..`),
/// and the `noisy` flag -- the pieces ngspice's inpcom.c resistor transform
/// consumes. `noisy`/`noise` is consumed (not re-emitted); everything else is
/// ignored (ngspice's transform handles only these).
fn split_resistor_params(assigns: &str) -> (String, String, bool) {
    let (mut tc1, mut tc2, mut m, mut noisy) = (None, None, None, false);
    for tok in assigns.split_whitespace() {
        if let Some((k, v)) = tok.split_once('=') {
            match k.to_ascii_lowercase().as_str() {
                "tc1" => tc1 = Some(v.to_string()),
                "tc2" => tc2 = Some(v.to_string()),
                "m" => m = Some(v.to_string()),
                "noisy" | "noise" => noisy = v.parse::<f64>().map_or(false, |x| x != 0.0),
                _ => {}
            }
        }
    }
    let tc = match (tc1, tc2) {
        (Some(a), Some(b)) => format!(" tc1={a} tc2={b}"),
        (Some(a), None) => format!(" tc1={a}"),
        _ => String::new(),
    };
    let m = m.map(|z| format!(" m={z}")).unwrap_or_default();
    (tc, m, noisy)
}

/// Split resolved top-level cards into up to `parts` ordered slices for parallel
/// expansion. Preserves order and NEVER splits a `.control ... .endc` block: that
/// block is stateful (a `let`/`print`/`alter` inside would be mangled if cut), so
/// a chunk boundary is only taken at `.control` nesting depth 0. Chunks are kept
/// roughly equal; at most `parts` are produced.
fn partition_top(active: &[String], parts: usize) -> Vec<Vec<String>> {
    let n = active.len();
    // roughly equal, rounding up (avoid usize::div_ceil for a lower MSRV)
    let target = if parts == 0 { n } else { (n + parts - 1) / parts };
    let mut chunks: Vec<Vec<String>> = Vec::new();
    let mut cur: Vec<String> = Vec::new();
    let mut in_control = false;
    for line in active {
        let k = kw(line);
        if k == ".control" {
            in_control = true;
        }
        cur.push(line.clone());
        if k == ".endc" {
            in_control = false;
        }
        // Close only at a safe boundary, once big enough, while still leaving
        // room for the final chunk (so we never exceed `parts`).
        if !in_control && cur.len() >= target && chunks.len() + 1 < parts {
            chunks.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

/// ngspice inp_compat's behavioral E/G split, done PRE-expansion so the
/// derived names match ngspice's own expansion exactly:
///
///   Exxx n1 n2 VALUE|VOL = {expr}        ->  Exxx n1 n2 Exxx_int1 0 1
///                                            bExxx Exxx_int1 0 v = {expr}
///   Gxxx n1 n2 VALUE|CUR = {expr} [m=X]  ->  Gxxx n1 n2 Gxxx_int1 0 X
///                                            bGxxx Gxxx_int1 0 v = {expr}
///
/// Downstream ngspice performs this same split (inpcom.c inp_compat, every
/// dialect but s3) — but on ngparse's output it ran on the FLAT names, so the
/// internal node came out `e.x1.e1_int1` where ngspice's own expansion makes
/// `x1.e1_int1`, and the B source `be.x1.e1` instead of `b.x1.be1`. The value
/// set is identical; the difference reorders the sparse matrix, which flips
/// convergence-marginal decks (same class as the behavioral-resistor naming).
/// Split here, inside the subckt body, and expansion renames both cards the
/// way ngspice's own flow does; downstream inp_compat then finds nothing left
/// to convert. A `VALUE={TABLE(...)}` card is NOT split — the TABLE form has
/// its own conversion (replace_table_fn in ps mode; inp_compat's otherwise).
fn eg_value_rewrite(lines: &[LogicalLine]) -> Vec<LogicalLine> {
    let mut out: Vec<LogicalLine> = Vec::with_capacity(lines.len());
    let mut in_ctl = false;
    for l in lines {
        let k = kw(&l.text);
        if k == ".control" {
            in_ctl = true;
        }
        let first = l.text.trim_start().as_bytes().first().map(|b| b.to_ascii_lowercase());
        let candidate = !in_ctl && (first == Some(b'e') || first == Some(b'g'));
        if k == ".endc" {
            in_ctl = false;
        }
        if candidate {
            if let Some((c1, c2)) = split_eg_value(&l.text) {
                for text in [c1, c2] {
                    let mut nl = l.clone();
                    nl.text = text;
                    out.push(nl);
                }
                continue;
            }
            if let Some(cards) = split_eg_table(&l.text) {
                for text in cards {
                    let mut nl = l.clone();
                    nl.text = text;
                    out.push(nl);
                }
                continue;
            }
        }
        out.push(l.clone());
    }
    out
}

/// Split one `E/G n1 n2 TABLE {expr} = (x0,y0) (x1,y1) ..` card into ngspice
/// inp_compat's four-card XSPICE pwl form (same pre-expansion naming argument
/// as eg_value_rewrite):
///
///   Exxx n1 n2 Exxx_int1 0 1
///   bExxx Exxx_int2 0 v={expr}
///   aExxx %v(Exxx_int2) %v(Exxx_int1) xfer_Exxx
///   .model xfer_Exxx pwl(x_array=[..] y_array=[..] input_domain=0.1 fraction=TRUE)
///
/// The 4-node `nc1 nc2 TABLE = (..)` variant is left alone (downstream
/// converts it as before). Returns None when the card is not of this form.
fn split_eg_table(line: &str) -> Option<Vec<String>> {
    let t = line.trim_start();
    let first = t.as_bytes().first()?.to_ascii_lowercase();
    if first != b'e' && first != b'g' {
        return None;
    }
    let (name, rest) = split_first(t);
    let (n1, rest) = split_first(rest);
    let (n2, rest) = split_first(rest);
    let rest = rest.trim_start();
    if n2.is_empty() || !rest.to_ascii_lowercase().starts_with("table") {
        return None;
    }
    let after = rest["table".len()..].trim_start();
    let after = after.strip_prefix('=').unwrap_or(after).trim_start();
    // expression in braces, then `= (x,y) (x,y) ..` pairs
    if !after.starts_with('{') {
        return None;
    }
    let close = matching_brace(after.as_bytes(), 0)?;
    let expr = after[1..close].trim();
    // pairs: strip separators, tokens then alternate x,y (as inp_compat does)
    let pairs: Vec<&str> = after[close + 1..]
        .split(|c: char| c.is_whitespace() || matches!(c, '(' | ')' | ',' | '='))
        .filter(|s| !s.is_empty())
        .collect();
    if pairs.len() < 4 || pairs.len() % 2 != 0 {
        return None;
    }
    let xs: Vec<&str> = pairs.iter().step_by(2).copied().collect();
    let ys: Vec<&str> = pairs.iter().skip(1).step_by(2).copied().collect();
    Some(vec![
        format!("{name} {n1} {n2} {name}_int1 0 1"),
        format!("b{name} {name}_int2 0 v={{{expr}}}"),
        format!("a{name} %v({name}_int2) %v({name}_int1) xfer_{name}"),
        // `limit=TRUE` clamps the pwl beyond the table endpoints — inpcom.c's
        // ACTUAL tprintf (6532) emits it even though its comment blocks don't.
        // Without it a DC sweep past the table range extrapolates unbounded
        // and a marginal deck loses the operating point.
        format!(
            ".model xfer_{name} pwl(x_array=[{}] y_array=[{}] input_domain=0.1 fraction=TRUE limit=TRUE)",
            xs.join(" "),
            ys.join(" ")
        ),
    ])
}

/// Byte offset of the `}` matching the `{` at `open`.
fn matching_brace(b: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0;
    for (i, &c) in b.iter().enumerate().skip(open) {
        match c {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Split one `E/G ... VALUE|VOL|CUR = {expr}` card (see eg_value_rewrite).
/// Returns None when the card is not of that form.
fn split_eg_value(line: &str) -> Option<(String, String)> {
    let t = line.trim_start();
    let first = t.as_bytes().first()?.to_ascii_lowercase();
    if first != b'e' && first != b'g' {
        return None;
    }
    let (name, rest) = split_first(t);
    let (n1, rest) = split_first(rest);
    let (n2, rest) = split_first(rest);
    let rest = rest.trim_start();
    if n2.is_empty() || rest.is_empty() {
        return None;
    }
    // keyword directly after the nodes, '=' attached or spaced (ngspice
    // matches the token in front of the line's first '=')
    let low = rest.to_ascii_lowercase();
    let kws: &[&str] = if first == b'e' { &["value", "vol"] } else { &["value", "cur"] };
    if !kws
        .iter()
        .any(|w| low.starts_with(w) && low[w.len()..].trim_start().starts_with('='))
    {
        return None;
    }
    // equation: from the first '{' to end of line (as inp_compat takes it)
    let open = rest.find('{')?;
    let mut equation = rest[open..].trim();
    // TABLE form has its own conversion path — leave it alone
    if equation.to_ascii_lowercase().contains("table(") {
        return None;
    }
    // G only: a trailing multiplier is moved onto the VCCS gain
    let mut gain = "1".to_string();
    if first == b'g' {
        if let Some(mp) = equation.to_ascii_lowercase().rfind(" m=") {
            gain = equation[mp + 3..].trim().to_string();
            equation = equation[..mp].trim_end();
        }
    }
    Some((
        format!("{name} {n1} {n2} {name}_int1 0 {gain}"),
        format!("b{name} {name}_int1 0 v = {equation}"),
    ))
}

/// Pre-expansion PSpice line rewrites: AKO model inheritance and the d/q
/// positional area factor. `.control` blocks pass through untouched.
fn pspice_line_rewrites(lines: &[LogicalLine]) -> Vec<LogicalLine> {
    let mut out = if lines.iter().any(|l| l.text.to_ascii_lowercase().contains("ako:")) {
        ako_rewrite(lines)
    } else {
        lines.to_vec()
    };
    let mut in_control = false;
    for l in &mut out {
        let k = kw(&l.text);
        if k == ".control" {
            in_control = true;
        } else if k == ".endc" {
            in_control = false;
        } else if !in_control {
            if let Some(nl) = pspice_dq_area(&l.text) {
                l.text = nl;
            }
        }
    }
    out
}

/// PSpice diodes/BJTs take a bare positional AREA factor after the model name
/// (`d1 n1 n2 dmod 7`, `q2 n1 n2 n3 [n4] bjtmod 1.35`); ngspice's device
/// parser instead mistakes the number for the model name. Convert it to the
/// named form `area=<n>` and strip the `[..]` substrate brackets, mirroring
/// inpcompat.c. Returns None when the card needs no change.
fn pspice_dq_area(line: &str) -> Option<String> {
    let t = line.trim_start();
    let first = t.as_bytes().first()?.to_ascii_lowercase();
    if first != b'd' && first != b'q' {
        return None;
    }
    let mut toks: Vec<String> = t.split_whitespace().map(str::to_string).collect();
    // name + nodes; a `[sub]` group may itself contain spaces ("[ 100 ]").
    let mut i = 1 + if first == b'd' { 2 } else { 3 };
    if first == b'q' && i < toks.len() {
        if toks[i].starts_with('[') {
            // substrate node in brackets: strip them (ngspice blanks the chars)
            while i < toks.len() && !toks[i].ends_with(']') {
                toks[i] = toks[i].trim_start_matches('[').to_string();
                i += 1;
            }
            if i < toks.len() {
                toks[i] = toks[i]
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .to_string();
                i += 1;
            }
            // emptied bracket tokens ("[ sub ]") just join as extra spaces
        } else if !toks[i].is_empty() && toks[i].bytes().all(|c| c.is_ascii_digit()) {
            i += 1; // an all-digit token is the (numeric) substrate node
        }
    }
    i += 1; // model name
    if i >= toks.len() {
        return None;
    }
    let a = &toks[i];
    if a.parse::<f64>().map(|v| v > 0.0).unwrap_or(false) || a.starts_with('{') {
        toks[i] = format!("area={a}");
        Some(toks.join(" "))
    } else {
        None
    }
}

/// Split a plain `.model <name> <type><body>` card. `body` is everything after
/// the type token (usually `(<params>)`), trimmed. Returns None on malformed.
fn split_model_card(line: &str) -> Option<(String, String, String)> {
    let t = line.trim_start();
    if t.len() < ".model".len() || !t[..".model".len()].eq_ignore_ascii_case(".model") {
        return None;
    }
    let after = &t[".model".len()..];
    let (name, rest) = split_first(after);
    let rest = rest.trim_start();
    let type_end = rest
        .find(|c: char| c == '(' || c.is_whitespace())
        .unwrap_or(rest.len());
    if name.is_empty() || type_end == 0 {
        return None;
    }
    Some((
        name.to_string(),
        rest[..type_end].to_string(),
        rest[type_end..].trim().to_string(),
    ))
}

/// Strip one outer `(...)` layer, if present.
fn strip_outer_parens(s: &str) -> &str {
    let t = s.trim();
    t.strip_prefix('(')
        .and_then(|u| u.strip_suffix(')'))
        .map(str::trim)
        .unwrap_or(t)
}

/// Resolve PSpice `.MODEL <new> AKO:<base> <type>(<overrides>)` inheritance,
/// mirroring inpcompat.c ako_model/find_model: the base model is looked up
/// among models of the SAME enclosing subckt first, then at top level; the
/// resolved card is `.model <new> <type> (<base params> <overrides>)` — a
/// duplicated parameter's LAST occurrence (the override) wins downstream.
/// Cards processed in deck order, so an AKO of an earlier AKO resolves too.
/// A card whose base cannot be found (or whose type disagrees) is left
/// untouched — ngspice then reports it loudly.
fn ako_rewrite(lines: &[LogicalLine]) -> Vec<LogicalLine> {
    // innermost enclosing .subckt line index per line (usize::MAX = top level)
    let mut scopes = Vec::with_capacity(lines.len());
    let mut stack: Vec<usize> = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        let k = kw(&l.text);
        if k == ".subckt" {
            stack.push(i);
        }
        scopes.push(stack.last().copied().unwrap_or(usize::MAX));
        if k == ".ends" || k == ".eom" {
            stack.pop();
        }
    }
    // plain models: (name, scope) -> (type, body-inside-parens)
    let mut models: HashMap<(String, usize), (String, String)> = HashMap::new();
    for (i, l) in lines.iter().enumerate() {
        if kw(&l.text) != ".model" || l.text.to_ascii_lowercase().contains("ako:") {
            continue;
        }
        if let Some((name, ty, body)) = split_model_card(&l.text) {
            models
                .entry((key(&name), scopes[i]))
                .or_insert((ty, strip_outer_parens(&body).to_string()));
        }
    }
    let mut out = lines.to_vec();
    for (i, l) in lines.iter().enumerate() {
        if kw(&l.text) != ".model" {
            continue;
        }
        // `.MODEL <new> AKO:<base> <type>(<overrides>)`
        let after = match l.text.trim_start().get(".model".len()..) {
            Some(a) => a,
            None => continue,
        };
        let (newname, rest) = split_first(after);
        let (akotok, over_rest) = split_first(rest);
        if !akotok.to_ascii_lowercase().starts_with("ako:") {
            continue;
        }
        let base = key(&akotok[4..]);
        let over_rest = over_rest.trim_start();
        let type_end = over_rest
            .find(|c: char| c == '(' || c.is_whitespace())
            .unwrap_or(over_rest.len());
        let (ty, overrides) = (&over_rest[..type_end], &over_rest[type_end..]);
        let found = models
            .get(&(base.clone(), scopes[i]))
            .or_else(|| models.get(&(base.clone(), usize::MAX)))
            .cloned();
        let Some((base_ty, base_body)) = found else { continue };
        if !base_ty.eq_ignore_ascii_case(ty) {
            continue; // type disagreement: leave for ngspice to report
        }
        let merged = format!(
            ".model {newname} {ty} ({} {})",
            base_body,
            strip_outer_parens(overrides)
        );
        out[i].text = merged.clone();
        // resolved AKO models can themselves serve as later bases
        if let Some((name, ty2, body)) = split_model_card(&merged) {
            models
                .entry((key(&name), scopes[i]))
                .or_insert((ty2, strip_outer_parens(&body).to_string()));
        }
    }
    out
}

/// PSpice card-level rewrites applied to the flat deck in [`Compat::Pspice`].
///
/// ngspice's `pspice_compat` (inpcompat.c) runs these per `.include`d file; we
/// inline every include, so we replicate the ones the corpus needs here. Kept as
/// a card-list pass, exactly as ngspice does it, so the output matches.
fn pspice_rewrites(cards: Vec<String>) -> Vec<String> {
    let cards = replace_table_fn(cards);
    let cards = replace_vswitch(cards);
    rename_pspice_model_temps(cards)
}

/// PSpice thermal .model parameters -> ngspice names, mirroring inpcompat.c:
/// `T_ABS`->`temp`, `T_REL_GLOBAL`->`dtemp`, `T_MEASURED`->`tnom`. Left alone,
/// ngspice warns "unrecognized parameter - ignored" and e.g. a noiseless
/// resistor (`T_ABS=-273.15`) silently becomes a noisy one at circuit temp.
fn rename_pspice_model_temps(mut cards: Vec<String>) -> Vec<String> {
    for card in &mut cards {
        if kw(card) != ".model" {
            continue;
        }
        let low = card.to_ascii_lowercase();
        if !(low.contains("t_abs") || low.contains("t_rel_global") || low.contains("t_measured"))
        {
            continue;
        }
        for (from, to) in
            [("t_abs", "temp"), ("t_rel_global", "dtemp"), ("t_measured", "tnom")]
        {
            *card = replace_word_ci(card, from, to);
        }
    }
    cards
}

/// Replace whole-word, case-insensitive occurrences of `from` (an identifier)
/// with `to`.
fn replace_word_ci(s: &str, from: &str, to: &str) -> String {
    let low = s.to_ascii_lowercase();
    let b = low.as_bytes();
    let is_ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while let Some(pos) = low[i..].find(from) {
        let start = i + pos;
        let end = start + from.len();
        let bounded = (start == 0 || !is_ident(b[start - 1]))
            && (end == b.len() || !is_ident(b[end]));
        out.push_str(&s[i..start]);
        out.push_str(if bounded { to } else { &s[start..end] });
        i = end;
    }
    out.push_str(&s[i..]);
    out
}

/// PSpice `VSWITCH` voltage switches -> ngspice equivalents, mirroring
/// inpcompat.c. Two forms:
///
/// * `.model M VSWITCH(vt=.. vh=..)` (short-transition) -> the classical
///   voltage-controlled switch `.model M sw(..)`; the `S` instance is unchanged.
/// * `.model M VSWITCH(von=.. voff=..)` -> the `pswitch` code model
///   `.model aM pswitch(log=TRUE ..)` with `von/voff/ron/roff` remapped to
///   `cntl_on/cntl_off/r_on/r_off`; every `S` instance calling it becomes an
///   `A` device `aS.. %gd(nc+ nc-) %gd(n+ n-) aM`.
///
/// Missing parameters get PSpice's defaults, exactly as inpcompat.c fills them.
fn replace_vswitch(cards: Vec<String>) -> Vec<String> {
    let mut pswitch: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::with_capacity(cards.len());
    for card in cards {
        if kw(&card) == ".model" && card.to_ascii_lowercase().contains("vswitch") {
            if let Some((newcard, needs_inst)) = convert_vswitch_model(&card) {
                if let Some(name) = needs_inst {
                    pswitch.insert(key(&name));
                }
                out.push(newcard);
                continue;
            }
        }
        out.push(card);
    }
    // No pswitch models -> no instance rewrites needed (sw-form S stays as-is).
    if pswitch.is_empty() {
        return out;
    }
    for card in &mut out {
        if let Some(nc) = rewrite_switch_instance(card, &pswitch) {
            *card = nc;
        }
    }
    out
}

/// Locate an `Assign` by case-insensitive name.
fn find_assign<'a>(assigns: &'a [Assign], name: &str) -> Option<&'a Assign> {
    assigns.iter().find(|a| a.name.eq_ignore_ascii_case(name))
}

/// Convert one `.model .. VSWITCH(..)` card. Returns the rewritten card and, for
/// the `pswitch` (von/voff) form, the ORIGINAL model name whose `S` instances must
/// then be rewritten to `A` devices. Returns None if the card is not a VSWITCH
/// model we understand (left untouched by the caller).
fn convert_vswitch_model(card: &str) -> Option<(String, Option<String>)> {
    let t = card.trim_start();
    let after = t[".model".len()..].trim_start();
    let (name, after) = split_first(after);
    let after = after.trim_start();
    // model type, up to '(' or whitespace
    let type_end = after
        .find(|c: char| c == '(' || c.is_whitespace())
        .unwrap_or(after.len());
    if !after[..type_end].eq_ignore_ascii_case("vswitch") {
        return None;
    }
    let params_raw = after[type_end..].trim();
    let params = params_raw
        .strip_prefix('(')
        .map(|s| s.strip_suffix(')').unwrap_or(s))
        .unwrap_or(params_raw)
        .trim();
    let assigns = parse_assignments(params);

    // vt/vh -> sw (model only); von/voff -> pswitch (model + instance). Prefer the
    // vt/vh test first, matching inpcompat.c's order.
    if find_assign(&assigns, "vt").is_some() || find_assign(&assigns, "vh").is_some() {
        // ron, roff, vt, vh -- native names unchanged, fill defaults.
        let body = build_switch_params(
            &assigns,
            &[
                ("ron", "ron", "1.0"),
                ("roff", "roff", "1.0e12"),
                ("vt", "vt", "0"),
                ("vh", "vh", "0"),
            ],
            "",
        );
        Some((format!(".model {name} sw ({body})"), None))
    } else if find_assign(&assigns, "von").is_some() || find_assign(&assigns, "voff").is_some() {
        // ron->r_on, roff->r_off, von->cntl_on, voff->cntl_off; add log=TRUE.
        let body = build_switch_params(
            &assigns,
            &[
                ("ron", "r_on", "1.0"),
                ("roff", "r_off", "1.0e6"),
                ("von", "cntl_on", "1"),
                ("voff", "cntl_off", "0"),
            ],
            "log=TRUE",
        );
        Some((format!(".model a{name} pswitch({body})"), Some(name.to_string())))
    } else {
        None
    }
}

/// Rebuild a switch model's parameter body: for each (pspice, native, default)
/// mapping emit `native=value` using the found value or the default, then append
/// any leftover params (unmapped) verbatim, then `extra` (e.g. `log=TRUE`).
fn build_switch_params(assigns: &[Assign], maps: &[(&str, &str, &str)], extra: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    for (ps, native, default) in maps {
        let val = find_assign(assigns, ps).map(|a| a.rhs.as_str()).unwrap_or(default);
        parts.push(format!("{native}={val}"));
    }
    // carry through any params not covered by the mapping (e.g. td is dropped by
    // ngspice today, but anything else the model set should survive)
    for a in assigns {
        if !maps.iter().any(|(ps, _, _)| a.name.eq_ignore_ascii_case(ps)) {
            parts.push(format!("{}={}", a.name, a.rhs));
        }
    }
    if !extra.is_empty() {
        parts.push(extra.to_string());
    }
    parts.join(" ")
}

/// If `card` is an `S` instance calling a `pswitch`-converted model, rewrite it to
/// the `A`-device form `a<inst> %gd(nc+ nc-) %gd(n+ n-) a<model>`. Returns None if
/// the card is not such an instance.
fn rewrite_switch_instance(card: &str, pswitch: &HashSet<String>) -> Option<String> {
    let t = card.trim_start();
    let first = t.as_bytes().first().copied().unwrap_or(0);
    if first != b's' && first != b'S' {
        return None;
    }
    // S instance: inst n+ n- nc+ nc- model [on|off]. Need at least 6 tokens.
    let toks: Vec<&str> = t.split_whitespace().collect();
    if toks.len() < 6 {
        return None;
    }
    let model = toks[5];
    if !pswitch.contains(&key(model)) {
        return None;
    }
    Some(format!(
        "a{inst} %gd({ncp} {ncn}) %gd({np} {nn}) a{model}",
        inst = toks[0],
        ncp = toks[3],
        ncn = toks[4],
        np = toks[1],
        nn = toks[2],
        model = model,
    ))
}

/// Index of the `)` matching the `(` at `open`, honoring nesting (`v(a,b)` inside
/// the table args has its own parens). None if unbalanced.
fn matching_paren(b: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut i = open;
    while i < b.len() {
        match b[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// PSpice `E/G ... {.. TABLE(ctrl, x1,y1, ..) ..}` -> a helper node driven by a
/// B-source using ngspice's native `pwl()`, mirroring inpcompat.c::replace_table:
///
///   e1 a b value={.. v(table_new_0) ..}
///   btable_new_0 table_new_0 0 v=pwl(ctrl, x1,y1, ..)
///
/// The `TABLE()` *function* (paren immediately after) is the PSpice interpolation
/// form; ngspice's behavioral parser has no such function, but `pwl()` takes the
/// identical `(ctrl, x1,y1, ..)` argument list. The native `E .. TABLE {ctrl}=(..)`
/// keyword form (handled elsewhere) is untouched -- it has no `table(`.
fn replace_table_fn(cards: Vec<String>) -> Vec<String> {
    let mut out = Vec::with_capacity(cards.len());
    let mut n = 0usize;
    for card in cards {
        let first = card
            .trim_start()
            .as_bytes()
            .first()
            .copied()
            .unwrap_or(0)
            .to_ascii_lowercase();
        let low = card.to_ascii_lowercase();
        // Only e/g behavioral sources carrying a table() function.
        if !(first == b'e' || first == b'g') || !low.contains("table(") {
            out.push(card);
            continue;
        }
        let mut line = card;
        let mut blines: Vec<String> = Vec::new();
        loop {
            let ll = line.to_ascii_lowercase();
            let Some(pos) = ll.find("table(") else { break };
            let open = pos + 5; // the '(' after "table"
            let Some(close) = matching_paren(line.as_bytes(), open) else {
                break;
            };
            let begline = &line[..pos];
            let args = &line[open..=close]; // "(ctrl, x1,y1, ..)" incl. parens
            let rest = &line[close + 1..];
            blines.push(format!("btable_new_{n} table_new_{n} 0 v=pwl{args}"));
            line = format!("{begline}v(table_new_{n}){rest}");
            n += 1;
        }
        out.push(line);
        out.extend(blines);
    }
    out
}

/// Prune `.model` cards never referenced by any device, so a deck that pulls in a
/// 5000-model PDK library but instantiates ten of them does not carry the other
/// 4990 into ngspice, where each would be set up and waste time and memory. A
/// model kept only because ngparse could not prove it unused is a safe, cheap
/// defaults.
/// Dangling-passive topology reduction (opt-in via [`Config::topo_reduce`]).
///
/// Removes a two-terminal R/C when (a) its NAME is referenced by no other card
/// (`.save @r1[i]`, `i(r1)`, an F/H controlling source, `.probe r1` all
/// protect it) and (b) one of its terminals — other than ground, `null`, or a
/// `.global` node — is referenced by no other card in the whole FLAT deck.
/// References are counted conservatively: every whitespace token AND every
/// identifier run inside expression text (`v(x)`, `.ic v(x)=..`, `.control`
/// script lines) protects a name, so over-protection is possible but a
/// wrong removal is not. Repeats to a fixpoint, so a dead-end chain
/// (`in -- R -- x -- C -- y`, nothing else on x or y) collapses entirely.
///
/// Runs on the flat deck BEFORE unused-model pruning, so a model used only by
/// removed devices is pruned along with them. ngspice tried this during
/// circuit setup (commit aac195, since reverted) and hit `.probe`/AC/XSPICE
/// ordering problems; done at parse time the simulator only ever sees the
/// surviving devices and the matrix shrinks. Never silent: removals are
/// summarized in a `* ngparse:` comment card.
fn reduce_dangling_passives(cards: Vec<String>, globals: &HashSet<String>) -> Vec<String> {
    // Every way a card can reference a name: normalized whitespace tokens plus
    // maximal runs of node-name characters (so `v(a,b)` yields `a` and `b`).
    fn refs(c: &str) -> HashSet<String> {
        let mut s = HashSet::new();
        for tok in c.split_whitespace() {
            let t = key(tok.trim_matches(|ch: char| {
                matches!(ch, '{' | '}' | '\'' | '"' | '(' | ')' | ',' | '=')
            }));
            if !t.is_empty() {
                s.insert(t);
            }
        }
        let mut run = String::new();
        for ch in c.chars() {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '+' | '-') {
                run.push(ch.to_ascii_lowercase());
            } else if !run.is_empty() {
                s.insert(std::mem::take(&mut run));
            }
        }
        if !run.is_empty() {
            s.insert(run);
        }
        s
    }
    let toksets: Vec<HashSet<String>> = cards.iter().map(|c| refs(c)).collect();
    // name -> number of CARDS referencing it (presence, not multiplicity)
    let mut count: HashMap<String, usize> = HashMap::new();
    for ts in &toksets {
        for t in ts {
            *count.entry(t.clone()).or_insert(0) += 1;
        }
    }
    // `.control` script lines are never device candidates (`reset`, `run`, ..)
    let mut in_ctl = vec![false; cards.len()];
    let mut ctl = false;
    for (i, c) in cards.iter().enumerate() {
        let k = kw(c);
        if k == ".control" {
            ctl = true;
        }
        in_ctl[i] = ctl;
        if k == ".endc" {
            ctl = false;
        }
    }
    let mut removed = vec![false; cards.len()];
    let mut removed_names: Vec<String> = Vec::new();
    loop {
        let mut changed = false;
        for i in 0..cards.len() {
            if removed[i] || in_ctl[i] {
                continue;
            }
            let c = &cards[i];
            let first = c.as_bytes().first().map(|b| b.to_ascii_lowercase());
            if first != Some(b'r') && first != Some(b'c') {
                continue;
            }
            // behavioral / still-symbolic values: leave alone
            if c.contains('{') || c.contains('\'') {
                continue;
            }
            let toks: Vec<&str> = c.split_whitespace().collect();
            if toks.len() < 4 {
                continue;
            }
            if count.get(&key(toks[0])).copied().unwrap_or(0) > 1 {
                continue; // the device itself is referenced somewhere
            }
            let dangling = toks[1..3].iter().any(|n| {
                let nk = key(n);
                nk != "0"
                    && nk != "null"
                    && !globals.contains(&nk)
                    && count.get(&nk).copied().unwrap_or(0) <= 1
            });
            if !dangling {
                continue;
            }
            removed[i] = true;
            changed = true;
            removed_names.push(toks[0].to_string());
            for t in &toksets[i] {
                if let Some(n) = count.get_mut(t) {
                    *n -= 1;
                }
            }
        }
        if !changed {
            break;
        }
    }
    if removed_names.is_empty() {
        return cards;
    }
    let list = removed_names.iter().take(10).cloned().collect::<Vec<_>>().join(" ");
    let more = if removed_names.len() > 10 {
        format!(" (+{} more)", removed_names.len() - 10)
    } else {
        String::new()
    };
    let note = format!(
        "* ngparse: topo-reduce removed {} dangling passive(s): {list}{more}",
        removed_names.len()
    );
    let mut out: Vec<String> = Vec::with_capacity(cards.len());
    for (i, c) in cards.into_iter().enumerate() {
        if !removed[i] {
            out.push(c);
        }
        if i == 0 {
            // after the first card, so a deck-leading title line stays first
            out.push(note.clone());
        }
    }
    out
}

fn prune_unused_models(cards: Vec<String>) -> (Vec<String>, HashSet<String>) {
    // model name -> defined
    let mut defined: HashSet<String> = HashSet::new();
    // base name -> its binned members (nfet -> {nfet.0, nfet.1, ...})
    let mut bins: HashMap<String, Vec<String>> = HashMap::new();
    for c in &cards {
        if kw(c) == ".model" {
            if let Some(n) = c.split_whitespace().nth(1) {
                let k = key(n);
                if let Some(d) = k.rfind('.') {
                    if d > 0 && k[d + 1..].chars().all(|ch| ch.is_ascii_digit()) {
                        bins.entry(k[..d].to_string()).or_default().push(k.clone());
                    }
                }
                defined.insert(k);
            }
        }
    }
    if defined.is_empty() {
        return (cards, HashSet::new());
    }

    let mut used: HashSet<String> = HashSet::new();
    for c in &cards {
        if kw(c) == ".model" {
            continue; // a model card doesn't "use" a model
        }
        for tok in c.split_whitespace() {
            // strip delimiters an expression/value might carry
            let t = key(tok.trim_matches(|ch: char| {
                matches!(ch, '{' | '}' | '\'' | '"' | '(' | ')' | ',' | '=')
            }));
            if t.is_empty() {
                continue;
            }
            if defined.contains(&t) {
                used.insert(t.clone());
            }
            // a reference to the BASE of a binned set keeps every bin
            if let Some(members) = bins.get(&t) {
                for m in members {
                    used.insert(m.clone());
                }
            }
        }
    }

    let pruned: HashSet<String> = defined.difference(&used).cloned().collect();
    if pruned.is_empty() {
        return (cards, pruned);
    }
    let kept = cards
        .into_iter()
        .filter(|c| {
            if kw(c) != ".model" {
                return true;
            }
            match c.split_whitespace().nth(1) {
                Some(n) => !pruned.contains(&key(n)),
                None => true,
            }
        })
        .collect();
    (kept, pruned)
}

/// Parse a `.subckt` header line into (name, ports, defaults).
/// Remove PSpice's `PARAMS:` keyword from a `.subckt` header or `X` instance
/// line. ngspice strips it in EVERY dialect (inpcom.c inp_fix_params:
/// `.subckt name 1 2 3 params: l=1 w=2` -> `.subckt name 1 2 3 l=1 w=2`);
/// left in place it reads as a positional token, so an X line resolves the
/// subckt name as literally `params:` ("unknown subckt") and a header gains a
/// phantom port. Case-insensitive, outside quotes/braces only.
fn strip_params_kw(line: &str) -> String {
    let low = line.to_ascii_lowercase();
    let b = line.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(b.len());
    let (mut depth, mut q, mut i) = (0i32, 0u8, 0usize);
    while i < b.len() {
        let c = b[i];
        if q != 0 {
            if c == q {
                q = 0;
            }
        } else {
            match c {
                b'\'' | b'"' => q = c,
                b'(' | b'{' => depth += 1,
                b')' | b'}' => depth -= 1,
                _ => {}
            }
            if depth == 0
                && q == 0
                && low[i..].starts_with("params:")
                && (i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_'))
            {
                i += "params:".len();
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| line.to_string())
}

fn parse_subckt_header(line: &str) -> (String, Vec<String>, Vec<(String, String)>) {
    let line = if line.to_ascii_lowercase().contains("params:") {
        strip_params_kw(line)
    } else {
        line.to_string()
    };
    let after = &line.trim_start()[".subckt".len()..];
    let (hdr, assign_str) = split_positional(after);
    let name = hdr.first().cloned().unwrap_or_default();
    let ports = hdr[1.min(hdr.len())..].to_vec();
    let defaults = parse_assignments(assign_str)
        .into_iter()
        .map(|a| (a.name, a.rhs))
        .collect();
    (name, ports, defaults)
}

/// Extract nested `.subckt ... .ends` blocks out of `body`, registering each in
/// `defs` under a scoped path `{path}/{name}` (recursively), and return the body
/// with those blocks REMOVED.
///
/// ngspice scopes a subckt definition to its enclosing subckt, so two different
/// parents may define the same name with different contents (see
/// tests/regression/lib-processing/scope-1.cir, where `sub1` and `sub2` each
/// define their own `sub`). Registering them all in one flat namespace would let
/// the last definition win — silently wrong. Leaving them in the body would also
/// emit stray `.subckt`/`.ends` cards and unbalance the deck.
fn extract_nested(
    body: Vec<String>,
    path: &str,
    defs: &mut HashMap<String, SubcktDef>,
) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < body.len() {
        if kw(&body[i]) == ".subckt" {
            let (name, ports, defaults) = parse_subckt_header(&body[i]);
            let mut inner = Vec::new();
            let mut depth = 1;
            i += 1;
            while i < body.len() && depth > 0 {
                let k = kw(&body[i]);
                if k == ".subckt" {
                    depth += 1;
                } else if k == ".ends" || k == ".eom" {
                    depth -= 1;
                    if depth == 0 {
                        i += 1;
                        break;
                    }
                }
                inner.push(body[i].clone());
                i += 1;
            }
            let child_path = if path.is_empty() {
                key(&name)
            } else {
                format!("{path}/{}", key(&name))
            };
            let inner = extract_nested(inner, &child_path, defs);
            defs.insert(
                child_path,
                SubcktDef {
                    ports,
                    defaults,
                    body: inner,
                },
            );
            continue;
        }
        out.push(body[i].clone());
        i += 1;
    }
    out
}

/// Find `scale=<value>` on any top-level `.option`/`.options` card. ngspice's
/// device-geometry scale factor; defaults to 1 when absent.
fn scale_option(cards: &[String]) -> Option<String> {
    for line in cards {
        let k = kw(line);
        if k == ".option" || k == ".options" {
            let (_kw, rest) = split_first(line.trim_start());
            for a in parse_assignments(rest) {
                if key(&a.name) == "scale" {
                    return Some(a.rhs);
                }
            }
        }
    }
    None
}

/// Select the bin for a device geometry, replicating ngspice `subckt.c`:
/// `csl = scale*l`, `csw = scale*w/nf`, match `csl>=lmin && csl<lmax &&
/// csw>=wmin && csw<wmax` (upper-exclusive). Returns the chosen bin's full name,
/// or None if no bin matches (caller then leaves ngspice to bin the full set).
fn select_bin(bins: &[BinDef], l: f64, w: f64, nf: f64, scale: f64) -> Option<&str> {
    let csl = scale * l;
    let csw = if nf != 0.0 { scale * w / nf } else { scale * w };
    bins.iter()
        .find(|b| csl >= b.lmin && csl < b.lmax && csw >= b.wmin && csw < b.wmax)
        .map(|b| b.name.as_str())
}

/// For each binned model set, select the bin for the instance geometry and
/// return the names of all the OTHER (non-selected) bins, so they can be pruned.
/// A set with no matching bin is left intact (ngspice bins it after any shrink).
fn bins_to_drop(
    bins: &HashMap<String, Vec<BinDef>>,
    l: Option<f64>,
    w: Option<f64>,
    nf: f64,
    scale: f64,
) -> HashSet<String> {
    let mut drop = HashSet::new();
    let (Some(l), Some(w)) = (l, w) else {
        return drop;
    };
    for set in bins.values() {
        if let Some(sel) = select_bin(set, l, w, nf, scale) {
            let sel = sel.to_string();
            for b in set {
                if b.name != sel {
                    drop.insert(b.name.clone());
                }
            }
        }
    }
    drop
}

/// Resolve `.if/.elseif/.else/.endif` blocks against `scope`, returning only the
/// active (kept) lines. Conditions that fail to evaluate are treated as false.
fn resolve_conditionals(body: &[String], scope: &Rc<Scope>) -> Vec<String> {
    struct Frame {
        active: bool,
        taken: bool,
        parent: bool,
    }
    let mut stack: Vec<Frame> = Vec::new();
    let mut out = Vec::new();
    let cur = |st: &[Frame]| st.last().map(|f| f.active).unwrap_or(true);

    for line in body {
        let k = kw(line);
        // `.if(sel == 1)` — the paren may be attached to the keyword, so the
        // first whitespace token is `.if(sel`; match on the prefix.
        let k = if k.starts_with(".if(") {
            ".if".to_string()
        } else if k.starts_with(".elseif(") {
            ".elseif".to_string()
        } else {
            k
        };
        if k == ".if" {
            let parent = cur(&stack);
            let cond = parent && eval_cond(line, scope);
            stack.push(Frame { active: cond, taken: cond, parent });
        } else if k == ".elseif" {
            if let Some(f) = stack.last_mut() {
                if f.taken {
                    f.active = false;
                } else {
                    let c = f.parent && eval_cond(line, scope);
                    f.active = c;
                    f.taken = c;
                }
            }
        } else if k == ".else" {
            if let Some(f) = stack.last_mut() {
                f.active = f.parent && !f.taken;
                f.taken = true;
            }
        } else if k == ".endif" {
            stack.pop();
        } else if cur(&stack) {
            out.push(line.clone());
        }
    }
    out
}

/// Evaluate the `(...)` condition of a `.if`/`.elseif` line. Missing/failed -> false.
fn eval_cond(line: &str, scope: &Rc<Scope>) -> bool {
    let Some(open) = line.find('(') else { return false };
    let Some(close) = line.rfind(')') else { return false };
    if close <= open {
        return false;
    }
    let cond = normalize_eq(&line[open + 1..close]);
    match parse(&cond).and_then(|e| eval(&e, scope)) {
        Ok(v) => v != 0.0,
        Err(_) => false,
    }
}

/// In a `.if`/`.elseif` condition a lone `=` means equality, not assignment:
/// `.elseif (select2 = 3)` is `select2 == 3`.  ngspice accepts this because it
/// hands the condition to numparam, which treats `=` as `==`; a condition is a
/// boolean expression with no assignments, so every `=` that is not already part
/// of `==`/`!=`/`<=`/`>=` is an equality.  Doubles those, leaving the compound
/// operators untouched.
fn normalize_eq(cond: &str) -> String {
    let b = cond.as_bytes();
    let mut out = String::with_capacity(cond.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'=' {
            let prev = i.checked_sub(1).map(|j| b[j]);
            let next = b.get(i + 1).copied();
            // already `==`: copy both and skip past.
            if next == Some(b'=') {
                out.push_str("==");
                i += 2;
                continue;
            }
            // tail of `!=`/`<=`/`>=`: leave as-is.
            if matches!(prev, Some(b'!') | Some(b'<') | Some(b'>')) {
                out.push('=');
                i += 1;
                continue;
            }
            // lone `=` used as equality -> `==`.
            out.push_str("==");
            i += 1;
            continue;
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A LogicalLine from literal text, for building small decks in tests.
    fn ll(text: &str) -> LogicalLine {
        LogicalLine {
            text: text.to_string(),
            file: std::sync::Arc::from("test"),
            line_no: 1,
        }
    }

    /// Roles of a device card's positional tokens, exactly as `emit_device` computes them.
    fn roles(line: &str) -> Vec<Role> {
        let (inst, rest) = split_first(line);
        let (positional, _) = split_positional(rest);
        dev_roles(inst.as_bytes()[0], &positional)
    }

    /// Tokens renamed as nodes.
    fn nodes(line: &str) -> usize {
        roles(line).iter().filter(|r| **r == Role::Node).count()
    }

    /// Tokens renamed as controlling-device instance names.
    fn ctrl(line: &str) -> usize {
        roles(line).iter().filter(|r| **r == Role::Inst).count()
    }

    /// Every expectation below was read off the reference ngspice's own
    /// `listing expand` output for the same card inside a subckt.
    /// `.option scale` multiplies the DRAWN l/w before the bin comparison
    /// (subckt.c:907 `csl = scale * c->l`), defaulting to 1 when unset.
    #[test]
    fn option_scale_shifts_bin_selection() {
        let bins = vec![
            BinDef { name: "n.0".into(), lmin: 0.0, lmax: 1e-7, wmin: 0.0, wmax: 1.0 },
            BinDef { name: "n.1".into(), lmin: 1e-7, lmax: 1e-6, wmin: 0.0, wmax: 1.0 },
        ];
        // scale=1: a drawn l of 1.2e-7 lands in the upper bin
        assert_eq!(select_bin(&bins, 1.2e-7, 1e-6, 1.0, 1.0), Some("n.1"));
        // scale=0.5: the same drawn l scales to 0.6e-7 -> the LOWER bin
        assert_eq!(select_bin(&bins, 1.2e-7, 1e-6, 1.0, 0.5), Some("n.0"));
        // upper bound is exclusive, lower inclusive
        assert_eq!(select_bin(&bins, 1e-7, 1e-6, 1.0, 1.0), Some("n.1"));
    }

    /// A subckt's own `scale` param is the HSPICE ELEMENT scale — a different
    /// mechanism from `.option scale`, and it must NOT affect bin selection.
    /// foundry_a's `pch_lvt_mac ... scale='scale_mos_lvt'` (0.9) would otherwise
    /// silently rebin every device in the deck.
    #[test]
    fn subckt_scale_param_does_not_leak_into_binning() {
        let lines = vec![
            ll("* t"),
            ll(".param scale_mos=0.5"),
            ll(".subckt m1 d g s scale='scale_mos'"),
            ll("mn d g s s nch l=1u w=1u"),
            ll(".ends"),
            ll("x1 a b 0 m1"),
        ];
        let se = SubcktExpander::new(&lines);
        // no `.option scale` anywhere -> the global stays 1, despite the subckt
        // declaring scale=0.5
        assert_eq!(se.option_scale, 1.0);
    }

    /// A numeric node name inside `v()`/`i()` must be kept as the node, not
    /// folded to a float: `v(1)` stays `v(1)`, never `v(1.000...e0)` (which is a
    /// different, nonexistent node — ngspice matches nodes by string).
    #[test]
    fn numeric_node_in_v_is_not_folded() {
        let lines = vec![
            ll("* t"),
            ll("v1 1 0 dc 2.7"),
            ll("b1 b1 0 v=ln(v(1))"),
            ll("r1 b1 0 1k"),
        ];
        let out = SubcktExpander::new(&lines).expand();
        let b = out.cards.iter().find(|c| c.starts_with("b1 ")).unwrap();
        assert!(b.contains("v(1)"), "node folded: {b}");
        assert!(!b.contains("v(1."), "node folded to float: {b}");
    }

    /// PSpice/HSPICE `if(cond,a,b)` is the ternary selector; ngspice's behavioral
    /// parser has no `if()` (it only knows `ternary_fcn`, which is what ngspice's
    /// own pspice-compat rewrites `if` to). We must emit the native name because
    /// that compat pass does not reach ngparse's already-flat deck.
    #[test]
    fn if_becomes_ternary_fcn() {
        let out = SubcktExpander::new(&[
            ll("* t"),
            ll(".subckt s 1 2"),
            ll(".param g=2"),
            ll("g1 1 2 value={if(v(1,2)>0, g*v(1,2), 0)}"),
            ll(".ends"),
            ll("xs a b s"),
        ])
        .expand();
        // the VALUE= expression now lives on the split-off B source
        // (eg_value_rewrite): g.xs.g1 is the linear VCCS, b.xs.bg1 the equation
        let g = out.cards.iter().find(|c| c.starts_with("b.xs.bg1")).unwrap();
        assert!(g.contains("ternary_fcn("), "if not rewritten: {g}");
        assert!(!g.contains("if("), "stray if( left: {g}");
        // a user function or variable literally named `if`-prefixed is untouched;
        // only the exact 3-arg `if` selector is rewritten.
    }

    /// PSpice behavioral functions (pwr/pwrs/stp/int) have no ngspice equivalent
    /// except via pspice_compat's injected .func, which never reaches our flat
    /// deck; in Pspice mode we emit the native form. Gated on the mode -- default
    /// mode leaves them alone.
    #[test]
    fn pspice_functions_rewritten() {
        let expand = |compat| {
            let cfg = crate::config::Config::default().with_compat(compat);
            SubcktExpander::with_config(
                &[
                    ll("* t"),
                    ll(".subckt s 1 2"),
                    ll("b1 1 2 v={pwr(v(1),2)+pwrs(v(1),3)+stp(v(1))+int(v(1))}"),
                    ll(".ends"),
                    ll("xs a b s"),
                ],
                cfg,
            )
            .expand()
        };
        let ps = expand(crate::config::Compat::Pspice);
        let b = ps.cards.iter().find(|c| c.starts_with("b")).unwrap();
        assert!(b.contains("pow("), "pwr/pwrs not rewritten: {b}");
        assert!(b.contains("u(") && b.contains("floor("), "stp/int not rewritten: {b}");
        assert!(!b.contains("pwr"), "stray pwr left: {b}");
        // default mode is untouched -- the functions pass through verbatim
        let df = expand(crate::config::Compat::Default);
        let b = df.cards.iter().find(|c| c.starts_with("b")).unwrap();
        assert!(b.contains("pwr("), "default mode should not rewrite: {b}");
    }

    /// Opt-in dangling-passive reduction: removes two-terminal R/C on nodes
    /// referenced nowhere else (the sourceforge thread's reproducer), cascades
    /// down dead-end chains, and NEVER touches a device or node referenced
    /// anywhere — `.save`, `i()`, `v()` in a B-source, `.control` text,
    /// `.global`. Off by default (deck unchanged).
    #[test]
    fn topo_reduce_dangling_passives() {
        let deck = [
            ll("* t"),
            ll("r2 0 R2_2 10e3"),
            ll("r1 R1_1 R1_2 1000"),
            ll("v1 IN 0 SIN(0 1 1000 0 0 0) AC 1"),
            ll("rload IN 0 1k"),
            ll("rchain IN chx 1k"),
            ll("cchain chx chy 1p"),
            ll("rsaved IN saved_n 1k"),
            ll("rcurr IN curr_n 1k"),
            ll("bsrc bs 0 v='2*v(bref_n)'"),
            ll("rbref IN bref_n 1k"),
            ll(".save v(saved_n)"),
            ll(".probe i(rcurr)"),
            ll(".ac dec 20 10 100e3"),
        ];
        // default: OFF, nothing removed
        let off = SubcktExpander::new(&deck).expand();
        assert!(off.cards.iter().any(|c| c.starts_with("r1 ")), "default must not reduce");
        // on: r1/r2 and the dead-end chain go; everything referenced stays
        let cfg = crate::config::Config::default().with_topo_reduce(true);
        let on = SubcktExpander::with_config(&deck, cfg).expand();
        let has = |p: &str| on.cards.iter().any(|c| c.starts_with(p));
        assert!(!has("r1 ") && !has("r2 "), "dangling r1/r2 kept: {:?}", on.cards);
        assert!(!has("rchain") && !has("cchain"), "dead-end chain kept: {:?}", on.cards);
        assert!(has("rload") && has("v1"), "connected devices removed");
        assert!(has("rsaved"), ".save-referenced node not protected");
        assert!(has("rcurr"), ".probe i()-referenced device not protected");
        assert!(has("rbref"), "B-source v()-referenced node not protected");
        assert!(
            on.cards.iter().any(|c| c.starts_with("* ngparse: topo-reduce removed 4 ")),
            "removal note missing/wrong: {:?}",
            on.cards
        );
    }

    /// `.if(cond)` with the paren attached to the keyword resolves like
    /// `.if (cond)`. Unhandled, EVERY branch's contents leaked into the deck
    /// ("device already exists" on same-named instances per branch).
    #[test]
    fn if_with_attached_paren() {
        let out = SubcktExpander::new(&[
            ll("* t"),
            ll(".param sel = 0"),
            ll(".if(sel == 1)"),
            ll("r1 a 0 111"),
            ll(".elseif(sel == 2)"),
            ll("r1 a 0 222"),
            ll(".else"),
            ll("r1 a 0 333"),
            ll(".endif"),
        ])
        .expand();
        let rs: Vec<&String> = out.cards.iter().filter(|c| c.starts_with("r1")).collect();
        assert_eq!(rs.len(), 1, "exactly one branch must survive: {rs:?}");
        assert!(rs[0].contains("3.33"), "else branch expected: {}", rs[0]);
    }

    /// PSpice `AKO:` model inheritance (ps mode): base params inherited, the
    /// override appended so its value wins; scoped lookup (same subckt first).
    /// And the d/q positional area factor becomes `area=<n>`.
    #[test]
    fn pspice_ako_and_area() {
        let cfg = crate::config::Config::default().with_compat(crate::config::Compat::Pspice);
        let out = SubcktExpander::with_config(
            &[
                ll("* t"),
                ll(".subckt amp 1 2 3"),
                ll(".MODEL QP350 PNP(IS=1.4E-15 BF=70 RB=350)"),
                ll(".MODEL QP AKO:QP350 PNP(BF=150 VA=100)"),
                ll("Q1 1 2 3 QP 2"),
                ll(".ends"),
                ll("x1 a b c amp"),
                ll("d1 a b dm 7"),
                ll(".model dm d(is=1e-14)"),
            ],
            cfg,
        )
        .expand();
        let m = out
            .cards
            .iter()
            .find(|c| c.to_ascii_lowercase().starts_with(".model x1:qp "))
            .unwrap_or_else(|| panic!("AKO model missing: {:?}", out.cards));
        assert!(m.contains("PNP") || m.contains("pnp"), "type lost: {m}");
        assert!(m.to_lowercase().contains("rb=") && m.to_lowercase().contains("va="),
            "base/override params missing: {m}");
        // override BF must come AFTER the inherited BF (last wins in ngspice)
        let low = m.to_ascii_lowercase();
        let b70 = low.find("bf=7").expect("inherited bf");
        let b150 = low.find("bf=1.5").expect("override bf");
        assert!(b150 > b70, "override must follow base: {m}");
        let q = out.cards.iter().find(|c| c.to_ascii_lowercase().starts_with("q.x1.q1")).unwrap();
        assert!(q.to_lowercase().contains("area=2"), "q area factor: {q}");
        let d = out.cards.iter().find(|c| c.to_ascii_lowercase().starts_with("d1")).unwrap();
        assert!(d.to_lowercase().contains("area=7"), "d area factor: {d}");
    }

    /// PSpice `PARAMS:` in `.SUBCKT` headers and X lines is stripped in every
    /// dialect (ngspice inp_fix_params). Left in place, the X line resolves
    /// the subckt name as literally `params:` -> "unknown subckt", including
    /// for NESTED calls inside another subckt.
    #[test]
    fn params_keyword_stripped() {
        let out = SubcktExpander::new(&[
            ll("* t"),
            ll(".SUBCKT inner 1 2 PARAMS: r=1k"),
            ll("r1 1 2 'r'"),
            ll(".ENDS"),
            ll(".SUBCKT outer a b PARAMS: rr=2k"),
            ll("x1 a b inner PARAMS: r={rr}"),
            ll(".ENDS"),
            ll("xo n1 n2 outer PARAMS: rr=5k"),
        ])
        .expand();
        assert!(
            !out.cards.iter().any(|c| c.to_ascii_lowercase().contains("params:")),
            "params: leaked: {:?}",
            out.cards
        );
        let r = out
            .cards
            .iter()
            .find(|c| c.to_ascii_lowercase().starts_with("r.xo.x1"))
            .unwrap_or_else(|| panic!("nested inner not expanded: {:?}", out.cards));
        assert!(r.contains("5.0") || r.contains("5e3") || r.contains("5.000"), "rr not bound: {r}");
    }

    /// XSPICE bracketed vector parameters (`cntl_array = [-2 -1 1 2]`) are ONE
    /// value token, spaces included; and an `.ic`/`.nodeset` inside a subckt
    /// body gets its `v(node)` arguments renamed like any other node reference
    /// (ports to caller nodes, internals prefixed).
    #[test]
    fn bracketed_arrays_and_scoped_ic() {
        let out = SubcktExpander::new(&[
            ll("* t"),
            ll(".model var_clock d_osc(cntl_array = [-2 -1 1 2] freq_array = [1e3 1e3 10e3 10e3]"),
            ll("+ duty_cycle = 0.1)"),
            ll("a5 cntl clk var_clock"),
            ll(".subckt filt in out"),
            ll("r1 in mid 1k"),
            ll("r2 mid out 1k"),
            ll(".ic v(out)=2.5 v(mid)=1.25"),
            ll(".ends"),
            ll("x1 a b filt"),
        ])
        .expand();
        // NB the reader joins `+` continuations before SubcktExpander runs; join
        // manually here to keep the fixture faithful to one logical card.
        let out2 = SubcktExpander::new(&[
            ll("* t"),
            ll(".model var_clock d_osc(cntl_array = [-2 -1 1 2] freq_array = [1e3 1e3 10e3 10e3] duty_cycle = 0.1)"),
            ll("a5 cntl clk var_clock"),
        ])
        .expand();
        let m = out2.cards.iter().find(|c| c.starts_with(".model var_clock")).unwrap();
        assert!(m.contains("[-2 -1 1 2]"), "cntl_array mangled: {m}");
        assert!(m.contains("[1e3 1e3 10e3 10e3]"), "freq_array mangled: {m}");
        assert!(m.contains("duty_cycle"), "params after array lost: {m}");
        let ic = out.cards.iter().find(|c| c.starts_with(".ic")).unwrap();
        assert!(
            ic.contains("v(b)") && ic.contains("v(x1.mid)"),
            "scoped .ic not renamed: {ic}"
        );
    }

    /// PSpice mode predefines `temp='temper'`/`vt`/`gmin` (as ngspice's
    /// pspice_compat does), rewrites 3-arg `LIMIT` to the ternary clamp, and
    /// renames thermal .model params (`t_abs`->`temp` etc). Without the first,
    /// a `VALUE={..TEMP..}` cannot resolve and the whole VALUE= is dropped.
    #[test]
    fn pspice_temp_limit_and_model_thermals() {
        let cfg = crate::config::Config::default().with_compat(crate::config::Compat::Pspice);
        let out = SubcktExpander::with_config(
            &[
                ll("* t"),
                ll(".param drift='2e-6*(TEMP-27)'"),
                ll(".param clamped='limit(5,0,3)'"),
                ll("e1 a 0 VALUE={0.5+drift}"),
                ll("g1 a 0 VALUE={LIMIT(v(a),-1,1)}"),
                ll("r2 a 0 'clamped'"),
                ll(".model rn res(t_abs=-273.15)"),
                ll("r1 a 0 rn 1k"),
            ],
            cfg,
        )
        .expand();
        assert!(out.drops.is_empty(), "drops: {:?}", out.drops);
        // VALUE= cards are split (eg_value_rewrite); the expressions live on
        // the be1/bg1 sources
        let e = out.cards.iter().find(|c| c.starts_with("be1")).unwrap();
        assert!(e.contains("temper"), "TEMP not mapped to temper: {e}");
        let g = out.cards.iter().find(|c| c.starts_with("bg1")).unwrap();
        assert!(
            g.contains("ternary_fcn") && !g.to_lowercase().contains("limit("),
            "3-arg LIMIT not rewritten: {g}"
        );
        let r2 = out.cards.iter().find(|c| c.starts_with("r2")).unwrap();
        assert!(r2.contains("3"), "const LIMIT clamp wrong: {r2}");
        let m = out.cards.iter().find(|c| c.starts_with(".model rn")).unwrap();
        assert!(
            m.contains("temp=") && !m.to_lowercase().contains("t_abs"),
            "t_abs not renamed: {m}"
        );
        // default (hs) mode: TEMP stays an ordinary (undefined) param and
        // 2-arg HSPICE limit() remains a symbolic MC distribution.
        let df = SubcktExpander::with_config(
            &[
                ll("* t"),
                ll(".param lm='limit(0.5,0.1)'"),
                ll(".model nch nmos (vth0=lm)"),
                ll("m1 d g s b nch"),
            ],
            crate::config::Config::default(),
        )
        .expand();
        let m = df.cards.iter().find(|c| c.starts_with(".model nch")).unwrap();
        assert!(m.contains("limit("), "hs-mode MC limit must stay symbolic: {m}");
    }

    /// PSpice `TABLE()` E/G form -> a helper node driven by a `pwl()` B-source,
    /// matching inpcompat.c::replace_table. Only in Pspice mode.
    #[test]
    fn pspice_table_to_pwl() {
        let cfg = crate::config::Config::default().with_compat(crate::config::Compat::Pspice);
        let out = SubcktExpander::with_config(
            &[
                ll("* t"),
                ll("e1 out 0 value={table(v(a,b), 1, 10, 3, 30)}"),
                ll("va a 0 1"),
                ll("vb b 0 0"),
            ],
            cfg,
        )
        .expand();
        let e = out.cards.iter().find(|c| c.to_lowercase().starts_with("e1")).unwrap();
        assert!(e.contains("v(table_new_0)"), "e-source not repointed: {e}");
        let b = out.cards.iter().find(|c| c.starts_with("btable_new_0")).unwrap();
        assert!(b.contains("v=pwl(v(a,b)"), "pwl b-source wrong: {b}");
        assert!(!b.to_lowercase().contains("table("), "table( left in b: {b}");
    }

    /// PSpice `VSWITCH` von/voff form -> pswitch code model + S-device becomes an
    /// A-device with %gd ports (inpcompat.c). vt/vh form -> `sw`, instance kept.
    #[test]
    fn pspice_vswitch() {
        let cfg = crate::config::Config::default().with_compat(crate::config::Compat::Pspice);
        // von/voff -> pswitch, and its S instance -> A device
        let out = SubcktExpander::with_config(
            &[
                ll("* t"),
                ll(".model msw VSWITCH(Ron=1 Roff=1e9 Von=0.9 Voff=0.8)"),
                ll("s1 outp outn ctlp ctln msw"),
                ll("v1 ctlp 0 1"),
            ],
            cfg,
        )
        .expand();
        let m = out.cards.iter().find(|c| c.to_lowercase().contains("pswitch")).unwrap();
        assert!(m.contains(".model amsw pswitch("), "model not converted: {m}");
        assert!(m.contains("cntl_on=") && m.contains("r_on=") && m.contains("log=TRUE"), "remap wrong: {m}");
        let a = out.cards.iter().find(|c| c.starts_with("as1")).unwrap();
        assert_eq!(a, "as1 %gd(ctlp ctln) %gd(outp outn) amsw", "instance rewrite wrong: {a}");

        // vt/vh -> sw, instance unchanged
        let out = SubcktExpander::with_config(
            &[
                ll("* t"),
                ll(".model msw2 VSWITCH(vt=1.5 vh=0.3 ron=1 roff=1e9)"),
                ll("s1 a b c d msw2"),
                ll("v1 c 0 1"),
            ],
            cfg,
        )
        .expand();
        let m = out.cards.iter().find(|c| c.contains(" sw ")).unwrap();
        assert!(m.contains(".model msw2 sw (") && m.contains("vt=") && m.contains("vh="), "sw model wrong: {m}");
        assert!(out.cards.iter().any(|c| c.starts_with("s1 ")), "sw instance should stay an S device");
    }

    /// Multi-core expansion must be byte-identical to single-core: the whole
    /// project rests on it. Many independent top-level instances + a .control
    /// block (which must stay whole across the split).
    #[test]
    fn parallel_matches_single_core() {
        let mut lines = vec![
            ll("* t"),
            ll(".param g=2"),
            ll(".subckt cell a b"),
            ll("r1 a b {1k*g}"),
            ll("c1 a b 1p"),
            ll(".ends"),
        ];
        for i in 0..50 {
            lines.push(ll(&format!("x{i} n{i} 0 cell")));
            lines.push(ll(&format!("v{i} n{i} 0 {i}")));
        }
        lines.push(ll(".control"));
        lines.push(ll("let x = 0"));
        lines.push(ll("run"));
        lines.push(ll(".endc"));
        let one = SubcktExpander::with_config(&lines, Config::default()).expand();
        for cores in [2usize, 3, 4, 8] {
            let cfg = Config::with_cores(std::num::NonZeroUsize::new(cores).unwrap());
            let many = SubcktExpander::with_config(&lines, cfg).expand();
            assert_eq!(one.cards, many.cards, "cores={cores} output differs from single-core");
            assert_eq!(one.drops, many.drops, "cores={cores} drops differ");
        }
    }

    /// A behavioral resistor is expanded to ngspice's own B-source + noise B/R/V
    /// form using the resistor's LOCAL name, so after subckt-prefixing the names
    /// match ngspice's during-expansion transform (b.x1.xr1.br1, node x1.xr1.r1_3)
    /// rather than the flat-name form (br.x1.xr1.r1). A plain resistor is untouched.
    #[test]
    fn behavioral_resistor_matches_ngspice_form() {
        let out = SubcktExpander::new(&[
            ll("* t"),
            ll(".subckt rmac a b"),
            ll("r1 a b 'v(a,b)*10+50' noisy=1"),
            ll(".ends"),
            ll("xr1 n1 0 rmac"),
            ll("v1 n1 0 1"),
        ])
        .expand();
        let has = |p: &str| out.cards.iter().any(|c| c.to_lowercase().starts_with(p));
        assert!(has("b.xr1.br1 "), "main B-source missing/misnamed: {:?}", out.cards);
        assert!(has("b.xr1.br1_1 "), "noise B-source missing: {:?}", out.cards);
        assert!(has("r.xr1.rr1_2 "), "noise R missing: {:?}", out.cards);
        assert!(has("v.xr1.vr1_3 "), "sense V missing: {:?}", out.cards);
        // internal node is x1.xr1.r1_3 style -- no leading type letter
        let vcard = out.cards.iter().find(|c| c.to_lowercase().starts_with("v.xr1.vr1_3")).unwrap();
        assert!(vcard.to_lowercase().contains("xr1.r1_3 0 0"), "internal node wrong: {vcard}");
        // no leftover plain behavioral R card
        assert!(!out.cards.iter().any(|c| c.to_lowercase().starts_with("r.xr1.r1 ")), "stray R: {:?}", out.cards);

        // a plain (numeric) resistor is NOT transformed
        let out2 = SubcktExpander::new(&[ll("* t"), ll(".subckt s a b"), ll("r1 a b 1k"), ll(".ends"), ll("xr1 n 0 s")]).expand();
        assert!(out2.cards.iter().any(|c| c.to_lowercase().starts_with("r.xr1.r1 ")), "plain R changed: {:?}", out2.cards);
    }

    /// Single `|`/`&` are logical or/and in SPICE behavioral expressions.
    #[test]
    fn single_pipe_amp_are_logical() {
        use crate::expr::parse;
        assert!(parse("(a>0 | b>0)").is_ok());
        assert!(parse("(a>0 & b>0)").is_ok());
    }

    /// A bare keyword in a model card (no `=`) is a flag, kept verbatim, not
    /// dropped: `.model M VDMOS nchan` -- dropping `nchan` flips VDMOS polarity.
    #[test]
    fn model_bare_keyword_kept() {
        for (name, spec, kw) in [
            ("mn", ".model mn VDMOS nchan Vto=4 Kp=5.9", "nchan"),
            ("mp", ".model mp VDMOS pchan Vto=-4", "pchan"),
            ("mb", ".model mb VDMOS nchan", "nchan"),
        ] {
            let dev = format!("m1 d g s {name}");
            let out = SubcktExpander::new(&[ll("* t"), ll(spec), ll(&dev)]).expand();
            let m = out.cards.iter().find(|c| c.starts_with(".model")).unwrap();
            assert!(m.contains(kw), "keyword {kw} dropped: {m}");
        }
        // normal name=value models are unaffected
        let out = SubcktExpander::new(&[ll("* t"), ll(".model dm d (is=1e-14 n=2)"), ll("d1 1 0 dm")]).expand();
        let m = out.cards.iter().find(|c| c.starts_with(".model")).unwrap();
        assert!(m.contains("is=") && m.contains("n="), "params lost: {m}");
    }

    /// E/G-source TABLE form: `E n+ n- TABLE {ctrl} = (pts)` is split
    /// pre-expansion into ngspice inp_compat's four-card XSPICE pwl form, so
    /// the derived names match ngspice's own expansion (e1_int1/e1_int2 under
    /// the SUBCKT path, no type-letter prefix; b/a devices named be1/ae1).
    #[test]
    fn e_source_table_form_split() {
        let lines = vec![
            ll("* t"),
            ll(".subckt s a b"),
            ll("e1 a b TABLE {v(a)} = (0,0) (1,2)"),
            ll(".ends"),
            ll("x1 n1 n2 s"),
            ll("v1 n1 0 1"),
        ];
        let out = SubcktExpander::new(&lines).expand();
        let find = |p: &str| {
            out.cards
                .iter()
                .find(|c| c.to_lowercase().starts_with(p))
                .unwrap_or_else(|| panic!("missing {p}: {:?}", out.cards))
        };
        let e = find("e.x1.e1 ");
        assert!(e.to_lowercase().contains("x1.e1_int1 0 1"), "gain card wrong: {e}");
        let b = find("b.x1.be1 ");
        assert!(
            b.to_lowercase().contains("x1.e1_int2") && b.contains("v(n1)"),
            "b card wrong (ctrl node must rename): {b}"
        );
        // `%v(n)` may render as `%v n` — equivalent XSPICE port syntax
        let a = find("a.x1.ae1 ").to_lowercase();
        let i2 = a.find("x1.e1_int2").expect("int2 port missing");
        let i1 = a.find("x1.e1_int1").expect("int1 port missing");
        assert!(a.matches("%v").count() == 2 && i2 < i1, "a-device ports wrong: {a}");
        let m = find(".model x1:xfer_e1 ");
        assert!(
            m.contains("[0 1]") && m.contains("[0 2]") && m.contains("fraction=TRUE"),
            "pwl model wrong: {m}"
        );
    }

    /// i(dev) inside a subckt renames its argument as a DEVICE (with the
    /// device-letter prefix), not a node -- it measures current through a device.
    /// v(node) stays a node. Getting i() wrong left ngspice with an "unknown
    /// controlling source".
    #[test]
    fn i_of_device_renamed_as_instance() {
        let lines = vec![
            ll("* t"),
            ll(".subckt s a b"),
            ll("rs1 a b '6*i(rs2)'"),
            ll("rs2 a b 1"),
            ll(".ends"),
            ll("x1 n1 0 s"),
            ll("v1 n1 0 1"),
        ];
        let out = SubcktExpander::new(&lines).expand();
        let r = out.cards.iter().find(|c| c.to_lowercase().contains("rs1")).unwrap();
        assert!(r.contains("i(r.x1.rs2)"), "i() arg not instance-renamed: {r}");
        assert!(!r.contains("i(x1.rs2)"), "i() arg renamed as a node: {r}");
        assert_eq!(rename_inst("rs2", "x1.x2"), "r.x1.x2.rs2");
        assert_eq!(rename_inst("rs2", ""), "rs2");
    }

    /// Statistical draws must be kept SYMBOLIC, args resolved, so ngspice draws
    /// per Monte Carlo run -- folding them to nominal would collapse the
    /// distribution. Both the direct call and a `.param` reference to one.
    #[test]
    fn agauss_kept_symbolic_for_monte_carlo() {
        // direct in a model card
        let lines = vec![
            ll("* t"),
            ll(".model nch nmos (vth0=agauss(0.5,0.01,3))"),
            ll("m1 d g s b nch"),
        ];
        let out = SubcktExpander::new(&lines).expand();
        let m = out.cards.iter().find(|c| c.starts_with(".model nch")).unwrap();
        assert!(m.contains("agauss("), "agauss folded away: {m}");
        assert!(!m.contains("vth0=5") && !m.contains("vth0=0.5"), "folded to nominal: {m}");

        // via a .param reference
        let lines = vec![
            ll("* t"),
            ll(".param vm=agauss(0.5,0.01,3)"),
            ll(".model nch nmos (vth0=vm)"),
            ll("m1 d g s b nch"),
        ];
        let out = SubcktExpander::new(&lines).expand();
        let m = out.cards.iter().find(|c| c.starts_with(".model nch")).unwrap();
        assert!(m.contains("agauss("), "param-referenced agauss folded away: {m}");

        assert!(SubcktExpander::has_runtime("agauss(0.5,0.01,3)"));
        assert!(SubcktExpander::has_runtime("aunif(1,2)"));
        assert!(SubcktExpander::has_runtime("limit(1,0.1)"));
    }

    /// A negated statistical term must not emit `<binop>-` (`...e0+-(agauss(...))`).
    /// numparam rejects the operator-then-unary-sign sequence ("wrongly
    /// determined negation"); the right operand's leading sign must be
    /// parenthesized so it reads `... + (-(agauss(...)))`.
    #[test]
    fn no_binop_followed_by_unary_sign() {
        // vth0 = base + (a NEGATED statistical draw): 0.5 + -(dvth)
        let lines = vec![
            ll("* t"),
            ll(".param dvth=-agauss(0,0.01,3)"),
            ll(".param vm='0.5+dvth'"),
            ll(".model nch nmos (vth0=vm)"),
            ll("m1 d g s b nch"),
        ];
        let out = SubcktExpander::new(&lines).expand();
        let m = out.cards.iter().find(|c| c.starts_with(".model nch")).unwrap();
        assert!(m.contains("agauss("), "statistical draw folded away: {m}");
        assert!(!m.contains("+-"), "emitted binop-then-unary-sign `+-`: {m}");
        assert!(!m.contains("--"), "emitted `--`: {m}");
        // sanity: the sign survives, just parenthesized
        assert!(m.contains("(-("), "expected parenthesized unary sign: {m}");
    }

    /// String / keyword / version model-parameter values are literals kept
    /// verbatim, not folded or dropped: `version=3.3.0`, `mfg=acme_corp`,
    /// `fraction=false`, `file="x.txt"`. Only a failed arithmetic EXPRESSION drops.
    #[test]
    fn literal_param_values_kept_verbatim() {
        for (spec, want) in [
            ("d (version=3.3.0)", "version=3.3.0"),
            ("d (mfg=acme_corp)", "mfg=acme_corp"),
            ("d (fraction=false)", "fraction=false"),
        ] {
            let lines = vec![
                ll("* t"),
                ll(&format!(".model dm {spec}")),
                ll("d1 n1 0 dm"),
            ];
            let out = SubcktExpander::new(&lines).expand();
            let m = out.cards.iter().find(|c| c.starts_with(".model dm")).unwrap();
            assert!(m.contains(want), "want {want:?} in {m:?}");
            assert!(out.drops.is_empty(), "unexpected drop for {spec}: {:?}", out.drops);
        }
        // is_bare_word covers identifiers and quoted strings; `3.3.0` is kept via
        // the separate unparsable path (it starts with a digit), as the version
        // case above confirms.
        assert!(is_bare_word("acme_corp") && is_bare_word("false"));
        assert!(is_bare_word("\"x.txt\""));
        assert!(!is_bare_word("3.3.0") && !is_bare_word("1+nosuch") && !is_bare_word("a*b"));
    }

    /// In a `.if` condition a lone `=` is equality; `==`/`!=`/`<=`/`>=` are left.
    #[test]
    fn normalize_eq_doubles_lone_equals() {
        assert_eq!(normalize_eq("select2 = 3"), "select2 == 3");
        assert_eq!(normalize_eq("a==b"), "a==b");
        assert_eq!(normalize_eq("a != b"), "a != b");
        assert_eq!(normalize_eq("a <= b"), "a <= b");
        assert_eq!(normalize_eq("a>=b && c=d"), "a>=b && c==d");
    }

    #[test]
    fn fixed_node_counts() {
        assert_eq!(nodes("r1 a b 1k"), 2);
        assert_eq!(nodes("c1 a b 1p"), 2);
        assert_eq!(nodes("l1 a b 1n"), 2);
        assert_eq!(nodes("v1 a b dc 1"), 2);
        assert_eq!(nodes("i1 a b dc 1"), 2);
        assert_eq!(nodes("b1 a b v=1"), 2);
        assert_eq!(nodes("j1 d g s jmod"), 3);
        assert_eq!(nodes("z1 d g s zmod"), 3);
        assert_eq!(nodes("u1 a b umod"), 3);
        // e/g: 2 output nodes + 2 controlling nodes, all node-translated
        assert_eq!(nodes("e1 a b c d 2.0"), 4);
        assert_eq!(nodes("g1 a b c d 3.0"), 4);
        // these four were missing from the old table entirely -> 0 nodes renamed
        assert_eq!(nodes("t1 a b c d z0=50"), 4);
        assert_eq!(nodes("o1 a b c d omod"), 4);
        assert_eq!(nodes("s1 a b c d smod"), 4);
        assert_eq!(nodes("y1 a b c d ymod"), 4);
    }

    #[test]
    fn controlling_devices_are_instances_not_nodes() {
        // F/H/W sense a V-source; K names two inductors. subckt.c::numdevs().
        for l in ["f1 a b vsen 1.0", "h1 a b vsen 1.0", "w1 a b vsen wmod"] {
            assert_eq!((nodes(l), ctrl(l)), (2, 1), "{l}");
        }
        assert_eq!((nodes("k1 l1 l2 0.5"), ctrl("k1 l1 l2 0.5")), (0, 2));
    }

    /// ngspice: 2 output nodes, then `dim * numdevs()` controlling terms —
    /// E/G take 2 nodes per term, F/H take 1 source name per term.
    #[test]
    fn poly_controlling_terms_scale_with_dim() {
        assert_eq!(nodes("e1 a b poly(2) c1 c2 c3 c4 0 1 1"), 2 + 4);
        assert_eq!(nodes("g1 a b POLY(2) c1 c2 c3 c4 0 1 1"), 2 + 4);
        assert_eq!(nodes("e2 a b poly(3) c1 c2 c3 c4 c5 c6 0 1"), 2 + 6);
        // F/H: the controlling terms are V-source names, not nodes
        assert_eq!((nodes("f1 a b poly(2) vs1 vs2 0 1 1"), ctrl("f1 a b poly(2) vs1 vs2 0 1 1")), (2, 2));
        assert_eq!((nodes("h1 a b poly(1) vs1 0 1"), ctrl("h1 a b poly(1) vs1 0 1")), (2, 1));
        // non-POLY => dim 1
        assert_eq!(nodes("e5 a b c1 c2 2.0"), 4);
    }

    /// ngspice's tokenizer splits parens off, so every spelling must work.
    #[test]
    fn poly_spellings() {
        for l in [
            "e1 a b poly(2) c1 c2 c3 c4 0 1 1",
            "e1 a b poly( 2 ) c1 c2 c3 c4 0 1 1",
            "e1 a b POLY (2) c1 c2 c3 c4 0 1 1",
            "e1 a b POLY ( 2 ) c1 c2 c3 c4 0 1 1",
        ] {
            assert_eq!(nodes(l), 6, "{l}");
            assert!(roles(l).contains(&Role::Poly(2)), "{l}");
        }
        // Case-sensitive, like subckt.c and enhtrans.c: `Poly` is not POLY.
        assert!(!roles("e1 a b Poly(2) c1 c2 c3 c4 0 1 1")
            .contains(&Role::Poly(2)));
    }

    /// The HSPICE source-type marker is redundant with the device letter, so
    /// ngspice consumes it and it does not appear in the expanded card.
    #[test]
    fn source_type_marker_is_dropped() {
        for (l, n, c) in [
            ("e4 a b vcvs c1 c2 2.0", 4, 0),
            ("g4 a b vccs c1 c2 3.0", 4, 0),
            ("f4 a b cccs vs1 2.0", 2, 1),
            ("h4 a b ccvs vs1 2.0", 2, 1),
        ] {
            assert_eq!((nodes(l), ctrl(l)), (n, c), "{l}");
            assert!(roles(l).contains(&Role::Drop), "{l}");
        }
        // a marker only counts for its own device letter
        assert!(!roles("e4 a b vccs c1 c2 2.0").contains(&Role::Drop));
        assert_eq!(nodes("e6 a b vcvs poly(2) c1 c2 c3 c4 0 1 1"), 6);
    }

    #[test]
    fn bjt_is_three_four_or_five_nodes() {
        assert_eq!(nodes("q1 c b e qmod"), 3);
        // the old table hard-coded 3, so the substrate kept its subckt-internal
        // name and every instance shorted together on it
        assert_eq!(nodes("q2 c b e s qmod"), 4);
        assert_eq!(nodes("q3 c b e s t qmod"), 5); // VBIC/hicum2 thermal
        // a trailing area value is not a node
        assert_eq!(nodes("q4 c b e s qmod 2.0"), 4);
        assert_eq!(nodes("q5 c b e qmod 2.0"), 3);
        assert_eq!(nodes("q6 c b e qmod off"), 3);
        // `1e-6` fools ngspice's own "contains no alpha" area test; we parse it
        assert_eq!(nodes("q7 c b e s qmod 1e-6"), 4);
    }

    #[test]
    fn mos_is_four_to_seven_nodes() {
        assert_eq!(nodes("m1 d g s b nch l=1u w=2u"), 4);
        assert_eq!(nodes("m2 d g s b nch"), 4);
        assert_eq!(nodes("m3 d g s b nch off"), 4);
        assert_eq!(nodes("m4 d g s e t bsimbulk"), 5); // bsimbulk/bsimcmg thermal
        assert_eq!(nodes("m5 d g s e p1 p2 hv2"), 6);  // HiSIMHV/SOI3
        assert_eq!(nodes("m6 d g s e p1 p2 p3 b4soi"), 7); // B4SOI/B3SOI*
        assert_eq!(nodes("m7 d g s nch"), 3); // VDMOS
        assert_eq!(nodes("m8 d g s b nch.1 l=1u"), 4); // binned model name
    }

    #[test]
    fn diode_is_two_or_three_nodes() {
        assert_eq!(nodes("d1 a b dmod"), 2);
        assert_eq!(nodes("d2 a b dmod area=2e-6"), 2);
        assert_eq!(nodes("d3 a b t dmod"), 3); // self-heating
        assert_eq!(nodes("d4 a b t dmod thermal"), 3);
        assert_eq!(nodes("d5 a b dmod off"), 2);
        // ngspice miscounts this one as 3 nodes and dies at pass 2 with
        // "could not find a valid modelname", so no working deck contains it;
        // we read the bare positional area correctly instead.
        assert_eq!(nodes("d6 a b dmod 1e-6"), 2);
    }

    #[test]
    fn numeric_node_names_are_still_nodes() {
        // node "0"/"5" are numbers; only the token *after* the last node is the model
        assert_eq!(nodes("d1 1 2 dmod"), 2);
        assert_eq!(nodes("q1 1 2 3 qmod"), 3);
        assert_eq!(nodes("m1 1 2 3 4 nch"), 4);
    }

    /// MIFgettok (xspice/mif/mifutil.c) treats `=` `(` `)` `,` as whitespace and
    /// makes `[ ] ~ % < >` single-character tokens.
    #[test]
    fn mif_tokenizer() {
        assert_eq!(mif_tokens("%vd(a b) %vd(c d) amod"),
                   ["%","vd","a","b","%","vd","c","d","amod"]);
        assert_eq!(mif_tokens("%vnam ( Vsin1 ) %id ( out 0 ) m"),
                   ["%","vnam","Vsin1","%","id","out","0","m"]);
        assert_eq!(mif_tokens("[p1] [enable] atod"),
                   ["[","p1","]","[","enable","]","atod"]);
        assert_eq!(mif_tokens("~in out inv"), ["~","in","out","inv"]);
        assert_eq!(mif_tokens("d clk NULL NULL NULL q dff"),
                   ["d","clk","NULL","NULL","NULL","q","dff"]);
        assert_eq!(mif_tokens("a \"quoted str\" b"), ["a","quoted str","b"]);
    }

    /// `null` is a built-in global in ngspice (subckt.c::collect_global_nodes
    /// seeds the table with "0" and "null"), so it is never renamed.
    #[test]
    fn null_and_ground_are_global() {
        let nm = HashMap::new();
        let g = HashSet::new();
        for n in ["0", "null", "NULL", "Null"] {
            assert_eq!(map_node_with(n, &nm, "x1", &g), n, "{n}");
        }
        assert_eq!(map_node_with("foo", &nm, "x1", &g), "x1.foo");
    }

    #[test]
    fn k_has_no_nodes() {
        assert_eq!(nodes("k1 l1 l2 0.5"), 0);
    }

    #[test]
    fn xspice_a_device_claims_no_nodes() {
        // ngspice's get_number_terminals also returns 0 here; A-devices are
        // handled by a dedicated branch we don't implement, so emit_device
        // records a loud drop instead of silently shorting the connections.
        assert_eq!(nodes("a1 %v(a b) %v(c d) amod"), 0);
    }
}
