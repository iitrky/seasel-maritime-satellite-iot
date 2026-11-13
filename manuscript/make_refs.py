"""Format the Crossref metadata (crossref_metadata.json) as Nature / Scientific Reports references with DOIs.
Every DOI in this file was resolved against api.crossref.org."""
import html, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(HERE, "crossref_metadata.json")))
LATEX = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}


def tex(s):
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", "", s)
    return "".join(LATEX.get(c, c) for c in s)


def initials(given):
    out = []
    for part in given.replace(".", ". ").split():
        sub = [x for x in part.split("-") if x]
        out.append("-".join(x[0].upper() + "." for x in sub))
    return " ".join(out)


def authors(m):
    a = [(tex(x["family"]), initials(x.get("given", ""))) for x in m.get("author", []) if x.get("family")]
    names = [f"{f}, {g}" if g else f for f, g in a]
    if len(names) > 5:
        return names[0] + r" \emph{et al.}"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + r" \& " + names[-1]


def year(m):
    for k in ("published-print", "published-online", "issued"):
        if m.get(k):
            return m[k]["date-parts"][0][0]


def pages(m):
    p = m.get("page")
    if not p or "pp." in p:
        return m.get("article-number") or (p.split()[0] if p else None)
    return p.replace("-", "--")


def fmt(m):
    a, y, t = authors(m), year(m), tex(m["title"][0]).rstrip(".")
    doi = "https://doi.org/" + m["DOI"].lower()
    if m["type"] == "book":
        return f"{a} \\emph{{{t}}} 2nd edn ({tex(m.get('publisher', ''))}, {y}). {doi}"
    if m["type"] == "proceedings-article":
        venue = tex(m["container-title"][0])
        pg = pages(m)
        pub = tex(m.get("publisher", "")).replace("Institute of Electrical and Electronics Engineers (IEEE)", "IEEE")
        pub = {"IEEE": "IEEE", "ACM": "ACM"}.get(pub, pub)
        return f"{a} {t}. In \\emph{{{venue}}} {pg + ' ' if pg else ''}({pub}, {y}). {doi}"
    j = tex((m.get("short-container-title") or [""])[0] or m["container-title"][0])
    vol = m.get("volume")
    pg = pages(m)
    return f"{a} {t}. \\emph{{{j}}} \\textbf{{{vol}}}, {pg} ({y}). {doi}"


ABBR = {"IEEE Commun. Surv. Tutorials": "IEEE Commun. Surv. Tutor.", "Sci Rep": "Sci. Rep.", "Mach Learn": "Mach. Learn.",
        "Information Fusion": "Inf. Fusion", "Computer Communications": "Comput. Commun.",
        "Regional Studies in Marine Science": "Reg. Stud. Mar. Sci.",
        "Journal of the American Statistical Association": "J. Am. Stat. Assoc.",
        "Trans. Mach. Learn. Comm. Netw.": "IEEE Trans. Mach. Learn. Commun. Netw.",
        "Foundations and Trends\u00ae in Machine Learning": "Found. Trends Mach. Learn."}


def fmt_abbr(m):
    s = fmt(m)
    for a, b in ABBR.items():
        s = s.replace("\\emph{%s}" % a, "\\emph{%s}" % b)
    return s


R = {k: fmt_abbr(m) for k, m in M.items()}

if __name__ == "__main__":
    for k, v in R.items():
        print(k, "->", v)
