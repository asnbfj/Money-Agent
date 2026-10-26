"""Spectrum 2 tokens.json → tokens.css（light / dark / 跟随系统）。"""
import json, re, sys
src, out = sys.argv[1], sys.argv[2]
t = json.load(open(src, encoding="utf-8"))
themes = [th["id"] for th in t["color"]["themes"]]
colors = {c["name"]: c["value"] for c in t["color"]["tokens"]}

def raw(name, theme):
    v = colors[name]
    if isinstance(v, dict):
        return v.get(theme, v.get(themes[0]))
    return v

def resolve(name, theme, depth=0):
    v = raw(name, theme)
    m = re.fullmatch(r"\{([^}]+)\}", str(v).strip())
    if m and depth < 20:
        return resolve(m.group(1), theme, depth + 1)
    return v

def theme_vars(theme):
    lines = [f"  --{n}: {resolve(n, theme)};" for n in colors]
    for fam in ("shadow",):
        for tok in t.get(fam, {}).get("tokens", []):
            v = tok["value"]; v = v.get(theme, v.get(themes[0])) if isinstance(v, dict) else v
            lines.append(f"  --{tok['name']}: {v};")
    return "\n".join(lines)

static = []
for fam in ("spacing", "radius", "size", "borderWidth", "opacity"):
    for tok in t.get(fam, {}).get("tokens", []):
        static.append(f"  --{tok['name']}: {tok['value']};")
for key, val in t["type"]["families"].items():
    static.append(f"  --font-{key}: {val};")
for group in t["type"]["groups"]:
    fam = group.get("family", "sans")
    for st in group["styles"]:
        n = st["name"]
        static.append(f"  --{n}-font-size: {st['fontSize']};")
        static.append(f"  --{n}-line-height: {st['lineHeight']};")
        static.append(f"  --{n}-font-weight: {st['fontWeight']};")
        static.append(f"  --{n}-font-family: var(--font-{st.get('family', fam)});")

css = f"""/* {t.get('name','Spectrum 2')} — generated from tokens.json */
:root {{
{chr(10).join(static)}
}}
:root, [data-theme="light"] {{
  color-scheme: light;
{theme_vars('light')}
}}
[data-theme="dark"] {{
  color-scheme: dark;
{theme_vars('dark')}
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
{theme_vars('dark')}
  }}
}}
"""
open(out, "w", encoding="utf-8").write(css)
print("written", out, len(css))
