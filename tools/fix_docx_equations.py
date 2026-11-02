"""Thay các đoạn mã LaTeX (style Code) trong file .docx xuất từ Claude Docs bằng phương trình Word (OMML).

Cách dùng:  .venv/bin/python tools/fix_docx_equations.py bao_cao_xuat.docx bao_cao.docx
Cần: python-docx, pypandoc_binary.
"""
import copy
import sys
import tempfile
from pathlib import Path

import docx
import pypandoc
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
doc = docx.Document(src)
targets = [p for p in doc.paragraphs if p.style.name == "Code" and p.text.lstrip().startswith("\\")]
print("formulas:", len(targets))

with tempfile.TemporaryDirectory() as tmp:
    # |V| dạng ký tự trần bị LibreOffice hiển thị sai, nên đổi sang cặp dấu \left| \right|
    md = "\n\n".join(f"$${p.text.strip().replace('|V|', '\\left| V \\right|')}$$" for p in targets)
    out = Path(tmp) / "eq.docx"
    pypandoc.convert_text(md, "docx", format="markdown", outputfile=str(out))
    eq_doc = docx.Document(out)
    maths = [el for p in eq_doc.paragraphs for el in p._p.iter(qn("m:oMathPara"))]

assert len(maths) == len(targets), (len(maths), len(targets))
for p, math in zip(targets, maths):
    for child in list(p._p):
        if child.tag != qn("w:pPr"):
            p._p.remove(child)
    p.style = doc.styles["Normal"]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p._p.append(copy.deepcopy(math))

doc.save(dst)
print("saved", dst)
