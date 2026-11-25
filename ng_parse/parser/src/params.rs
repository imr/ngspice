//! Parameter collection, resolution, and `{...}`/`'...'` substitution — the
//! numparam replacement. Given the flattened card stream, this:
//!   1. collects every `.param` definition (multi-assignment lines and
//!      `name(args)=expr` function definitions),
//!   2. resolves parameter values lazily with memoization + cycle detection
//!      (so forward references and inter-parameter dependencies just work),
//!   3. substitutes inline `{expr}` / `'expr'` on device/model lines with the
//!      evaluated numeric value,
//! and emits a fully-numeric card stream plus resolution statistics.
//!
//! Values are `f64`. Anything that fails to evaluate is reported and (in numeric
//! output mode) replaced with a placeholder so downstream ngspice sees no
//! leftover `{}` — this is what lets us measure the model-ingest floor before the
//! evaluator is fully complete.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::expr::{eval, parse, Env, EvalError, Expr};
use crate::reader::LogicalLine;

/// One `.param` assignment parsed from a card.
pub(crate) struct Assign {
    pub(crate) name: String,
    pub(crate) args: Option<Vec<String>>,
    pub(crate) rhs: String,
}

/// Case-insensitive key.
pub(crate) fn key(s: &str) -> String {
    s.to_ascii_lowercase()
}

/// Parse the text after `.func`: `name(args) [=] body`.  Returns
/// `(name, arg names, body)` with the body's surrounding `'…'`/`{…}` stripped.
/// `.func` bodies are always a single function, so unlike `.param` there is no
/// multi-assignment to handle.
pub(crate) fn parse_func_def(rest: &str) -> Option<(String, Vec<String>, String)> {
    let s = rest.trim();
    let open = s.find('(')?;
    let name = s[..open].trim().to_string();
    if name.is_empty() {
        return None;
    }
    // Match the args paren (they never nest in a func header).
    let close = s[open..].find(')')? + open;
    let argnames: Vec<String> = s[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|a| !a.is_empty())
        .map(str::to_string)
        .collect();
    // Body: skip an optional `=`, then strip one layer of quotes/braces.
    let mut body = s[close + 1..].trim();
    if let Some(rest) = body.strip_prefix('=') {
        body = rest.trim();
    }
    let body = body
        .strip_prefix('\'')
        .and_then(|b| b.strip_suffix('\''))
        .or_else(|| body.strip_prefix('{').and_then(|b| b.strip_suffix('}')))
        .unwrap_or(body)
        .trim()
        .to_string();
    if body.is_empty() {
        return None;
    }
    Some((name, argnames, body))
}

/// Parse the text following `.param` into assignments. Handles:
///   * multiple `name = value` pairs on one line
///   * whitespace around `=`
///   * values quoted with `'...'`, braced `{...}`, or a bare token (with
///     paren-balancing so `max(a, b)` survives even unquoted)
///   * `name(a,b) = ...` function-definition headers
pub(crate) fn parse_assignments(rest: &str) -> Vec<Assign> {
    let b = rest.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    let skip_ws = |b: &[u8], mut i: usize| {
        while i < b.len() && (b[i] as char).is_whitespace() {
            i += 1;
        }
        i
    };

    while i < b.len() {
        i = skip_ws(b, i);
        if i >= b.len() {
            break;
        }
        // name
        let name_start = i;
        while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
            i += 1;
        }
        if i == name_start {
            // not an identifier start; skip a char to avoid infinite loop
            i += 1;
            continue;
        }
        let name = rest[name_start..i].to_string();

        // optional (args)
        let mut args = None;
        let j = skip_ws(b, i);
        if j < b.len() && b[j] == b'(' {
            let mut depth = 0;
            let mut k = j;
            while k < b.len() {
                match b[k] {
                    b'(' => depth += 1,
                    b')' => {
                        depth -= 1;
                        if depth == 0 {
                            k += 1;
                            break;
                        }
                    }
                    _ => {}
                }
                k += 1;
            }
            let arglist = &rest[j + 1..k - 1];
            args = Some(
                arglist
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
            );
            i = k;
        }

        // expect '='
        i = skip_ws(b, i);
        if i >= b.len() || b[i] != b'=' {
            // malformed; bail on this assignment
            break;
        }
        i += 1; // consume '='
        i = skip_ws(b, i);
        if i >= b.len() {
            break;
        }

        // value
        let rhs = match b[i] {
            b'\'' | b'{' => {
                let close = if b[i] == b'\'' { b'\'' } else { b'}' };
                let start = i + 1;
                let mut k = start;
                let mut depth = 1;
                while k < b.len() {
                    if b[i] == b'{' && b[k] == b'{' {
                        depth += 1;
                    } else if b[k] == close {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    k += 1;
                }
                let v = rest[start..k.min(b.len())].to_string();
                i = (k + 1).min(b.len());
                v
            }
            b'[' => {
                // Bracketed vector value (XSPICE array parameters):
                // `cntl_array = [-2 -1 1 2]`. The whole `[...]` group, spaces
                // included, is ONE value token; cutting it at the first space
                // truncated the model body and every remaining parameter with
                // it ("Too few values for parameter 'cntl_array'").
                let start = i;
                let mut depth = 0;
                while i < b.len() {
                    match b[i] {
                        b'[' => depth += 1,
                        b']' => {
                            depth -= 1;
                            if depth == 0 {
                                i += 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                    i += 1;
                }
                rest[start..i].to_string()
            }
            _ => {
                // Bare token, paren-balanced. A `,` ends it just like whitespace:
                // ngspice's INPgetTok (inpgtok.c) treats `=`, `(`, `)` and `,` as
                // separators — skipped before a token, terminating after one — so
                // `.model dm d (a=500.0, b=-500.0)` has TWO parameters, not a value
                // of `"500.0,"`. Depth-guarded, so an unquoted `max(a, b)` keeps its
                // internal commas.
                //
                // Whitespace does NOT end the value while the expression is
                // visibly unfinished: a binary operator stands on one side of the
                // gap. An unquoted behavioural source `i = -2m*v(a) + 1m*v(a)`
                // is one value; cutting it at the first space silently dropped
                // every term after the first. A following `name=value` starts
                // with an identifier, never an operator, so it still ends the
                // value.
                let is_op = |c: u8| matches!(c, b'+' | b'-' | b'*' | b'/' | b'^' | b'?' | b':' | b'<' | b'>' | b'&' | b'|' | b'%');
                let start = i;
                let mut depth = 0;
                while i < b.len() {
                    let c = b[i];
                    if c == b'(' {
                        depth += 1;
                    } else if c == b')' {
                        depth -= 1;
                    } else if c == b',' && depth == 0 {
                        break;
                    } else if (c as char).is_whitespace() && depth == 0 {
                        let next = skip_ws(b, i);
                        let before = b[i - 1];
                        let goes_on = next < b.len()
                            && i > start
                            && (is_op(before)
                                || matches!(before, b'=' | b'!')
                                || is_op(b[next])
                                || (matches!(b[next], b'=' | b'!')
                                    && next + 1 < b.len()
                                    && b[next + 1] == b'='));
                        if !goes_on {
                            break;
                        }
                        i = next;
                        continue;
                    }
                    i += 1;
                }
                rest[start..i].trim_end().to_string()
            }
        };

        out.push(Assign { name, args, rhs });
    }
    out
}

/// The resolved/lazy parameter + function table.
pub struct ParamTable {
    raw: HashMap<String, String>,
    funcs: HashMap<String, (Vec<String>, Rc<Expr>)>,
    parsed: RefCell<HashMap<String, Rc<Expr>>>,
    values: RefCell<HashMap<String, f64>>,
    resolving: RefCell<HashSet<String>>,
}

impl ParamTable {
    pub fn new() -> Self {
        ParamTable {
            raw: HashMap::new(),
            funcs: HashMap::new(),
            parsed: RefCell::new(HashMap::new()),
            values: RefCell::new(HashMap::new()),
            resolving: RefCell::new(HashSet::new()),
        }
    }

    /// Collect all `.param` and `.func` definitions from the flattened deck.
    pub fn collect(&mut self, lines: &[LogicalLine]) {
        for l in lines {
            let t = l.text.trim_start();
            if t.len() >= 6 && t[..6].eq_ignore_ascii_case(".param") {
                let rest = &t[6..];
                for a in parse_assignments(rest) {
                    let k = key(&a.name);
                    match a.args {
                        Some(argnames) => {
                            if let Ok(body) = parse(&a.rhs) {
                                self.funcs.insert(k, (argnames, Rc::new(body)));
                            }
                        }
                        None => {
                            // last definition wins (matches ngspice override order)
                            self.raw.insert(k, a.rhs);
                        }
                    }
                }
            } else if t.len() >= 5 && t[..5].eq_ignore_ascii_case(".func") {
                // `.func name(args) [=] body` -- ngspice's dedicated spelling for
                // a user function, equivalent to `.param name(args) = body`, which
                // it in fact rewrites .func into.  The `=` is optional and the body
                // may be 'quoted', {braced} or bare.  Collected the same way, so a
                // B-source `v='bar2(17.0)'` can inline bar2 instead of dropping it.
                if let Some((name, argnames, body)) = parse_func_def(&t[5..]) {
                    if let Ok(b) = parse(&body) {
                        self.funcs.insert(key(&name), (argnames, Rc::new(b)));
                    }
                }
            }
        }
    }

    /// Resolve a parameter to its value (memoized; detects cycles).
    fn resolve(&self, name: &str) -> Result<f64, EvalError> {
        let k = key(name);
        if let Some(v) = self.values.borrow().get(&k) {
            return Ok(*v);
        }
        if self.resolving.borrow().contains(&k) {
            return Err(EvalError::Parse(format!("cyclic parameter `{k}`")));
        }
        let raw = self
            .raw
            .get(&k)
            .ok_or_else(|| EvalError::UnknownVar(name.to_string()))?
            .clone();

        // parse (cache). Clone the Rc out of the borrow before any borrow_mut.
        let cached = self.parsed.borrow().get(&k).cloned();
        let expr: Rc<Expr> = match cached {
            Some(e) => e,
            None => {
                let e = Rc::new(parse(&raw)?);
                self.parsed.borrow_mut().insert(k.clone(), Rc::clone(&e));
                e
            }
        };

        self.resolving.borrow_mut().insert(k.clone());
        let scope = Scope {
            table: self,
            locals: HashMap::new(),
        };
        let res = eval(&expr, &scope);
        self.resolving.borrow_mut().remove(&k);

        let v = res?;
        self.values.borrow_mut().insert(k, v);
        Ok(v)
    }

    /// Number of scalar parameters collected (for diagnostics).
    pub fn param_count(&self) -> usize {
        self.raw.len()
    }

    /// Evaluate an inline expression string against this table.
    pub fn eval_str(&self, s: &str) -> Result<f64, EvalError> {
        let e = parse(s)?;
        let scope = Scope {
            table: self,
            locals: HashMap::new(),
        };
        eval(&e, &scope)
    }
}

impl Default for ParamTable {
    fn default() -> Self {
        Self::new()
    }
}

/// `ParamTable` is itself an environment (the global scope).
impl Env for ParamTable {
    fn var(&self, name: &str) -> Result<f64, EvalError> {
        self.resolve(name)
    }
    fn call_user(&self, name: &str, args: &[f64]) -> Result<Option<f64>, EvalError> {
        let k = key(name);
        let Some((argnames, body)) = self.funcs.get(&k) else {
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
        let inner = Scope {
            table: self,
            locals,
        };
        Ok(Some(eval(body, &inner)?))
    }
}

/// An evaluation scope: the global table plus function-argument locals.
struct Scope<'a> {
    table: &'a ParamTable,
    locals: HashMap<String, f64>,
}

impl Env for Scope<'_> {
    fn var(&self, name: &str) -> Result<f64, EvalError> {
        let k = key(name);
        if let Some(v) = self.locals.get(&k) {
            return Ok(*v);
        }
        self.table.resolve(name)
    }

    fn call_user(&self, name: &str, args: &[f64]) -> Result<Option<f64>, EvalError> {
        let k = key(name);
        let Some((argnames, body)) = self.table.funcs.get(&k) else {
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
        let body = Rc::clone(body);
        let inner = Scope {
            table: self.table,
            locals,
        };
        Ok(Some(eval(&body, &inner)?))
    }
}

/// Statistics from a substitution pass.
#[derive(Debug, Default, Clone)]
pub struct SubstStats {
    pub exprs_total: usize,
    pub exprs_failed: usize,
}

/// Format a resolved value the way ngspice prints numeric params.
pub(crate) fn fmt_num(v: f64) -> String {
    format!("{v:.15e}")
}

/// Substitute every `{expr}` (and single-quoted `'expr'`) in a device/model line
/// with its evaluated value against `env`. On failure the original delimited
/// expression is kept intact (so behavioral `v()`/`i()` refs and not-yet-resolved
/// params survive for a later stage or for ngspice).
pub fn substitute_line(
    env: &dyn Env,
    line: &str,
    placeholder: &str,
    stats: &mut SubstStats,
) -> String {
    let table = env;
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
            if k <= b.len() {
                let inner = &line[start..k.min(b.len())];
                stats.exprs_total += 1;
                match parse(inner).and_then(|e| eval(&e, table)) {
                    Ok(v) => out.push_str(&fmt_num(v)),
                    Err(_) => {
                        // Keep the original delimited expression so ngspice can
                        // resolve it (e.g. behavioral `v()`/`i()` refs, or
                        // subckt-local params resolved during expansion). Zeroing
                        // these would corrupt behavioral sources. `placeholder` is
                        // reserved for a future strict/numeric-only mode.
                        let _ = placeholder;
                        stats.exprs_failed += 1;
                        out.push(c as char);
                        out.push_str(inner);
                        out.push(close as char);
                    }
                }
                i = (k + 1).min(b.len());
                continue;
            }
        }
        out.push(c as char);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn func_def_parses_all_body_forms() {
        // .func name(args) 'body'  |  {body}  |  = body  |  bare
        assert_eq!(parse_func_def("bar2(p) 'v(p)+p'"),
                   Some(("bar2".into(), vec!["p".into()], "v(p)+p".into())));
        assert_eq!(parse_func_def("foo0()  '1013.0'"),
                   Some(("foo0".into(), vec![], "1013.0".into())));
        assert_eq!(parse_func_def("baz1(n,vp) 'n+i(vp)+vp'"),
                   Some(("baz1".into(), vec!["n".into(),"vp".into()], "n+i(vp)+vp".into())));
        assert_eq!(parse_func_def("g(x) = {x*2}"),
                   Some(("g".into(), vec!["x".into()], "x*2".into())));
    }

    use std::sync::Arc;

    fn lines(src: &str) -> Vec<LogicalLine> {
        crate::reader::logical_lines(src, Arc::from("t"))
    }

    #[test]
    fn unquoted_expression_with_spaces_is_one_value() {
        // a behavioural source written without quotes
        let a = parse_assignments("i = -2m*v(a) + 1m*v(a)*v(a)*v(a)");
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].name, "i");
        assert_eq!(a[0].rhs, "-2m*v(a) + 1m*v(a)*v(a)*v(a)");
        // operator on the left of the gap, and a following assignment
        let a = parse_assignments("v = 2 * v(x) -1 tc1=0.5 m=2");
        assert_eq!(a.len(), 3);
        assert_eq!(a[0].rhs, "2 * v(x) -1");
        assert_eq!((a[1].name.as_str(), a[1].rhs.as_str()), ("tc1", "0.5"));
        assert_eq!((a[2].name.as_str(), a[2].rhs.as_str()), ("m", "2"));
        // plain lists are untouched, negative values included
        let a = parse_assignments("a=500.0, b=-500.0 c=1e-3");
        assert_eq!(a.len(), 3);
        assert_eq!(a[1].rhs, "-500.0");
        // comparison operators
        let a = parse_assignments("r = v(a) >= 1 ? 1k : 2k");
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].rhs, "v(a) >= 1 ? 1k : 2k");
    }

    #[test]
    fn multi_value_param_line() {
        let a = parse_assignments(" a=0 b = 1 c=nonfet_mm d = 'a+b' ");
        assert_eq!(a.len(), 4);
        assert_eq!(a[0].name, "a");
        assert_eq!(a[2].rhs, "nonfet_mm");
        assert_eq!(a[3].rhs, "a+b");
    }

    #[test]
    fn resolves_dependencies_and_forward_refs() {
        let mut t = ParamTable::new();
        t.collect(&lines(".param x={2*y} y=10\n.param z=nonfet_mm nonfet_mm=1\n"));
        assert_eq!(t.resolve("x").unwrap(), 20.0);
        assert_eq!(t.resolve("z").unwrap(), 1.0);
    }

    #[test]
    fn user_function_definition_and_call() {
        let mut t = ParamTable::new();
        t.collect(&lines(
            ".param sel(m)='ka*(m==1)+kb*(m==2)'\n.param ka=10 kb=20\n",
        ));
        assert_eq!(t.eval_str("sel(1)").unwrap(), 10.0);
        assert_eq!(t.eval_str("sel(2)").unwrap(), 20.0);
    }

    #[test]
    fn substitutes_inline_exprs() {
        let mut t = ParamTable::new();
        t.collect(&lines(".param fclk=150e6\n"));
        let mut s = SubstStats::default();
        let out = substitute_line(&t, "L1 a b {(2.7e-9)*(150e6/fclk)}", "0", &mut s);
        assert!(out.starts_with("L1 a b 2.7"), "got {out}");
        assert_eq!(s.exprs_failed, 0);
    }

    #[test]
    fn cycle_is_reported_not_hung() {
        let mut t = ParamTable::new();
        t.collect(&lines(".param p=q q=p\n"));
        assert!(t.resolve("p").is_err());
    }
}
