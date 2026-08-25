import re
from pathlib import Path

text = Path(__file__).resolve().parents[1].joinpath("manuscript", "manuscript.tex").read_text(encoding="utf-8")
start = text.index(r"\begin{abstract}")
end = text.index(r"\bibliographystyle")
body = text[start:end]
body = re.sub(r"%.*", "", body)
body = re.sub(r"\\cite[pt]?\{[^}]*\}", "CITE", body)
body = re.sub(r"\\citeauthor\{[^}]*\}", "CITE", body)
body = re.sub(r"\\label\{[^}]*\}", "", body)
body = re.sub(r"\\ref\{[^}]*\}", "REF", body)
body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", body)
body = re.sub(r"[{}\\]", " ", body)
words = body.split()
print("word count (abstract through conclusion):", len(words))
