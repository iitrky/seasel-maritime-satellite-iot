"""Build the manuscript (PDF via pdflatex, Word via pandoc) from main.tex and the generated results.

    python manuscript/build_manuscript.py
"""
import os, re, subprocess, sys
from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
B = os.path.join(HERE, "build")
os.makedirs(B, exist_ok=True)
sys.path.insert(0, HERE)
from make_refs import R  # noqa: E402

OUT = "SeaSel_Manuscript_Bamila_Virgin_Louis"
FONT = "Times New Roman"

# ------------------------------------------------------------------ sources and citation order
parts = ["main.tex", "part1.tex", "part2.tex", "part3.tex"]
src = {p: open(os.path.join(HERE, p)).read() for p in parts}
full = src["main.tex"]
for p in parts[1:]:
    full = full.replace("\\input{%s}" % p[:-4], src[p])
order = []
for grp in re.findall(r"\\cite\{([^}]*)\}", full):
    for k in grp.split(","):
        k = k.strip()
        if k not in order:
            order.append(k)
missing = [k for k in order if k not in R]
assert not missing, missing
num = {k: i + 1 for i, k in enumerate(order)}
print("references cited:", len(order), "uncited:", [k for k in R if k not in num])
with open(os.path.join(HERE, "refs.tex"), "w") as f:
    f.write("\\begin{thebibliography}{99}\n\\small\n")
    for k in order:
        f.write("\\bibitem{%s} %s\n" % (k, R[k]))
    f.write("\\end{thebibliography}\n")

# ------------------------------------------------------------------ PDF
for _ in range(2):
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=HERE, stdout=subprocess.DEVNULL)
log = open(os.path.join(HERE, "main.log"), errors="ignore").read()
print("latex errors:", [l for l in log.splitlines() if l.startswith("!")][:5],
      "undefined:", re.findall(r"(?:Reference|Citation) `([^']*)' .*undefined", log)[:5])
subprocess.run(["cp", os.path.join(HERE, "main.pdf"), os.path.join(HERE, OUT + ".pdf")])
aux = open(os.path.join(HERE, "main.aux")).read()
lab = dict(re.findall(r"\\newlabel\{([^}@]*)\}\{\{([^}]*)\}", aux))


# ------------------------------------------------------------------ LaTeX -> pandoc-friendly LaTeX
def cite_numbers(keys):
    ns = sorted(num[k.strip()] for k in keys.split(","))
    out, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(str(ns[i]) if j == i else (f"{ns[i]},{ns[j]}" if j == i + 1 else f"{ns[i]}--{ns[j]}"))
        i = j + 1
    return ",".join(out)


macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(os.path.join(HERE, "generated", "numbers.tex")).read()))
s = full
for k in sorted(macros, key=len, reverse=True):
    s = s.replace("\\" + k + "{}", macros[k]).replace("\\" + k, macros[k])
s = s.replace("\\input{generated/numbers}\n", "")
for t in ("tab_main", "tab_disc", "tab_ablation", "tab_robust", "tab_ood", "tab_transfer"):
    s = s.replace("\\input{generated/%s}" % t, open(os.path.join(HERE, "generated", t + ".tex")).read())
s = s.replace("\\resizebox{\\textwidth}{!}{\\begin{tabular}", "\\begin{tabular}").replace("\\end{tabular}}", "\\end{tabular}")
s = re.sub(r"\\cmidrule\([^)]*\)\{[^}]*\}", "", s)


def number_floats(s, env, word):
    out, pos = [], 0
    for m in re.finditer(r"\\begin\{%s\}.*?\\end\{%s\}" % (env, env), s, re.S):
        body = m.group(0)
        lm = re.search(r"\\label\{([^}]*)\}", body)
        if lm and lm.group(1) in lab:
            body = body.replace("\\caption{", "\\caption{\\textbf{%s %s.} " % (word, lab[lm.group(1)]), 1)
        out.append(s[pos:m.start()] + body)
        pos = m.end()
    out.append(s[pos:])
    return "".join(out)


s = number_floats(s, "figure", "Figure")
s = number_floats(s, "table", "Table")
s = re.sub(r"\\ref\{([^}]*)\}", lambda m: lab.get(m.group(1), "??"), s)
s = re.sub(r"\\cite\{([^}]*)\}", lambda m: "\\textsuperscript{%s}" % cite_numbers(m.group(1)), s)
cnt = [0]


def eqn(m):
    cnt[0] += 1
    body = re.sub(r"\\label\{[^}]*\}", "", m.group(1)).rstrip()
    return "\\begin{equation}%s\\qquad\\qquad(%d)\n\\end{equation}" % (body, cnt[0])


s = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", eqn, s, flags=re.S)
sec = [0, 0]


def secnum(m):
    if m.group(1) == "section":
        sec[0] += 1; sec[1] = 0
        return "\\section*{%d~ %s}" % (sec[0], m.group(2))
    sec[1] += 1
    return "\\subsection*{%d.%d~ %s}" % (sec[0], sec[1], m.group(2))


body0 = s.index("\\begin{document}")
s = s[:body0] + re.sub(r"\\(section|subsection)\{((?:[^{}]|\{[^{}]*\})*)\}", secnum, s[body0:])
FIX = [("\\Big(", "("), ("\\Big)", ")"), ("\\big(", "("), ("\\big)", ")"), ("\\Big[", "["), ("\\Big]", "]"),
       ("$\\pm$", "\u00b1"), ("\\newcommand{\\vect}[1]{\\bm{#1}}", ""), ("\\bm{", "\\mathbf{"), ("\\bm\\", "\\mathbf\\"),
       ("A. Bamila Virgin Louis$^{1,*}$", "A. Bamila Virgin Louis\u00b9*"), ("$^{1}$Department", "\u00b9Department"),
       ("$^{*}$Correspondence", "*Correspondence"), ("0000-0003-0297-4004", "https://orcid.org/0000-0003-0297-4004"),
       ("\\!\\!", ""), ("\\!", "")]
for a, b in FIX:
    s = s.replace(a, b)
s = re.sub(r"\{\\rm\s+([^{}]*)\}", r"\\mathrm{\1}", s)
s = s.replace("$^\\circ$", "\u00b0").replace("$^{-1}$", "\u207b\u00b9").replace("$^{-2}$", "\u207b\u00b2")
s = re.sub(r"\\includegraphics(\[[^\]]*\])?\{\.\./figures/(diagrams/)?", r"\\includegraphics\1{", s)
refs = "\\section*{References}\n\n" + "\n\n".join("%d.~%s" % (num[k], R[k]) for k in order) + "\n"
s = s.replace("\\input{refs}", refs)
open(os.path.join(B, "main_word.tex"), "w").write(s)
print("equations:", cnt[0], "sections:", sec[0])
for d in (os.path.join(ROOT, "figures"), os.path.join(ROOT, "figures", "diagrams")):
    for f in os.listdir(d):
        if f.endswith(".png"):
            subprocess.run(["cp", os.path.join(d, f), B])

# ------------------------------------------------------------------ reference.docx styles
ref = os.path.join(B, "reference.docx")
subprocess.run(["pandoc", "-o", ref, "--print-default-data-file", "reference.docx"], check=True)
d = Document(ref)


def style(name, size, bold=False, italic=False, align=None, before=0, after=6, line=1.5, keep=False):
    st = None
    for cand in d.styles:
        n = cand.element.find(qn("w:name"))
        if n is not None and n.get(qn("w:val")) in (name, name.lower()):
            st = cand
            break
    if st is None:
        return
    rpr = st.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), FONT)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    st.font.size = Pt(size); st.font.bold = bold; st.font.italic = italic
    st.font.color.rgb = RGBColor(0, 0, 0)
    col = rpr.find(qn("w:color"))
    for a in ("w:themeColor", "w:themeShade", "w:themeTint"):
        if col is not None and col.get(qn(a)) is not None:
            del col.attrib[qn(a)]
    pf = st.paragraph_format
    pf.space_before, pf.space_after, pf.line_spacing = Pt(before), Pt(after), line
    if align is not None:
        pf.alignment = align
    if keep:
        pf.keep_with_next = True


J, C, L = WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT
style("Normal", 12, line=1.5)
for n in ("Body Text", "First Paragraph", "Compact"):
    style(n, 12, align=J, after=6, line=1.5)
style("Title", 16, bold=True, align=C, after=12, line=1.15)
style("Author", 12, align=C, after=4, line=1.15)
style("Date", 10, align=C, after=8, line=1.15)
style("Abstract", 12, align=J, line=1.5, after=6)
style("Abstract Title", 12, bold=True, align=L, before=12, after=4, line=1.15)
style("Heading 1", 14, bold=True, before=18, after=8, line=1.15, keep=True)
style("Heading 2", 12, bold=True, before=12, after=6, line=1.15, keep=True)
style("Image Caption", 11, align=J, after=12, line=1.15)
style("Table Caption", 11, align=J, before=6, after=4, line=1.15, keep=True)
style("Captioned Figure", 12, align=C, line=1.0, keep=True)
for s_ in d.sections:
    s_.page_height, s_.page_width = Cm(29.7), Cm(21.0)
    s_.left_margin = s_.right_margin = s_.top_margin = s_.bottom_margin = Cm(2.5)
d.save(ref)
raw = os.path.join(B, "raw.docx")
subprocess.run(["pandoc", "main_word.tex", "-f", "latex", "-t", "docx", "--reference-doc", ref, "--resource-path", B,
                "-o", raw], check=True, cwd=B)


# ------------------------------------------------------------------ post-processing
def set_cell_border(cell, **edges):
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.find(qn("w:tcBorders"))
    if b is None:
        b = OxmlElement("w:tcBorders"); tcPr.append(b)
    for edge, sz in edges.items():
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single"); el.set(qn("w:sz"), str(sz)); el.set(qn("w:color"), "000000")
        b.append(el)


d = Document(raw)
TEXT_W = 9071
for t in d.tables:
    rows = t.rows
    ncol = len(t.columns)
    first = 0.26 if ncol >= 7 else (0.3 if ncol >= 4 else 0.5)
    widths = [int(TEXT_W * first)] + [int(TEXT_W * (1 - first) / (ncol - 1))] * (ncol - 1)
    tblPr = t._tbl.tblPr
    for tag in ("w:tblW", "w:tblLayout", "w:jc"):
        for el in tblPr.findall(qn(tag)):
            tblPr.remove(el)
    tw = OxmlElement("w:tblW"); tw.set(qn("w:w"), str(sum(widths))); tw.set(qn("w:type"), "dxa"); tblPr.append(tw)
    jc = OxmlElement("w:jc"); jc.set(qn("w:val"), "center"); tblPr.append(jc)
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    for gc, w in zip(t._tbl.tblGrid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(w))
    nhead = 1
    while nhead < len(rows) and rows[nhead].cells[0].text.strip() in ("", "Corruption", "Shift", "Method", "Variant", "Predictor"):
        nhead += 1
    symtab = rows[0].cells[0].text.strip() == "Symbol"
    for ri, r in enumerate(rows):
        for ci, c in enumerate(r.cells):
            tcPr = c._tc.get_or_add_tcPr()
            for el in tcPr.findall(qn("w:tcW")):
                tcPr.remove(el)
            w = OxmlElement("w:tcW"); w.set(qn("w:w"), str(widths[min(ci, ncol - 1)])); w.set(qn("w:type"), "dxa")
            tcPr.insert(0, w)
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(1); p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.line_spacing = 1.0
                p.alignment = L if (ci == 0 or symtab) else C
                for run in p.runs:
                    run.font.size = Pt(9)
                    if ri < nhead:
                        run.font.bold = True
            if ri == 0:
                set_cell_border(c, top=12)
            if ri == nhead - 1:
                set_cell_border(c, bottom=6)
            if ri == len(rows) - 1:
                set_cell_border(c, bottom=12)
for p in d.paragraphs:
    if p._p.xpath(".//pic:pic") and not p.text.strip():
        p.alignment = C
        p.paragraph_format.keep_with_next = True
in_refs = False
for p in d.paragraphs:
    if p.style.name.startswith("Heading"):
        in_refs = p.text.strip() == "References"
        continue
    if in_refs and re.match(r"\d+\.", p.text.strip()):
        p.paragraph_format.left_indent = Cm(0.8); p.paragraph_format.first_line_indent = Cm(-0.8)
        p.paragraph_format.line_spacing = 1.15; p.paragraph_format.space_after = Pt(3); p.alignment = L
        for r_ in p.runs:
            r_.font.size = Pt(10.5)
sect = d.sections[0]._sectPr
for pm in sect.iter(qn("w:pgMar")):
    for a, v in (("w:header", "709"), ("w:footer", "709"), ("w:gutter", "0")):
        if pm.get(qn(a)) is None:
            pm.set(qn(a), v)
for s_ in d.sections:
    fp = s_.footer.paragraphs[0] if s_.footer.paragraphs else s_.footer.add_paragraph()
    fp.alignment = C
    run = fp.add_run()
    for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = txt
        run._r.append(el)
d.core_properties.author = "A. Bamila Virgin Louis"
d.core_properties.title = "Certified selective link prediction for maritime Internet of Things over low Earth orbit satellites"
out = os.path.join(HERE, OUT + ".docx")
d.save(out)
subprocess.run(["zip", "-q", "-d", out, "*/"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("wrote", OUT + ".docx and .pdf")
