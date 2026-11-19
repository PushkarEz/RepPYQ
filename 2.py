from pathlib import Path

Path("frontend/src/vite-env.d.ts").write_text(
    '/// <reference types="vite/client" />\n', encoding="utf-8"
)

main_py = Path("backend/app/main.py")
txt = main_py.read_text(encoding="utf-8")
if '@app.get("/")' not in txt:
    txt = txt.replace(
        '@app.get("/api/health")',
        '@app.get("/")\ndef root():\n    return {"app": "RepPYQ API", "status": "running"}\n\n\n@app.get("/api/health")',
    )
    main_py.write_text(txt, encoding="utf-8")

gi = Path(".gitignore")
if gi.exists():
    g = gi.read_text(encoding="utf-8")
    if "fix_vercel.py" not in g:
        gi.write_text(g.rstrip() + "\nfix_vercel.py\n", encoding="utf-8")

print("Fixed vite-env.d.ts and added root / route!")