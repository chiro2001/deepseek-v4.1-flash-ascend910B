"""Render standalone HTML and SHA256 sidecars; requires Python-Markdown."""
import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path

import markdown


STYLE = """
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#18283c;font:16px/1.85 system-ui,-apple-system,'Segoe UI','Microsoft YaHei',sans-serif}
header{background:#142b49;color:white;padding:24px max(24px,calc((100vw - 1280px)/2));font-size:14px}header a{color:#c2dcff;margin-right:22px}
.layout{max-width:1280px;margin:30px auto;padding:0 24px;display:grid;grid-template-columns:245px minmax(0,1fr);gap:26px}
nav{position:sticky;top:20px;align-self:start;max-height:calc(100vh - 40px);overflow:auto;background:#fff;border:1px solid #dce4ed;border-radius:10px;padding:18px;font-size:13px}nav ul{list-style:none;padding-left:0}nav li{margin:9px 0}nav li li{padding-left:12px}nav a{text-decoration:none}
main{min-width:0;background:white;border:1px solid #dce4ed;border-radius:10px;padding:34px 42px}h1{font-size:29px;line-height:1.45;margin-top:0}h2{font-size:23px;border-top:1px solid #dce4ed;padding-top:26px;margin-top:36px;scroll-margin-top:20px}h3{font-size:18px;scroll-margin-top:20px}a{color:#1c5bac;overflow-wrap:anywhere}p,li{overflow-wrap:anywhere}strong{color:#16497f}
code{font:0.87em ui-monospace,SFMono-Regular,Consolas,monospace;background:#edf2f8;padding:2px 5px;border-radius:4px}pre{background:#172b43;color:#edf5ff;border-radius:8px;padding:18px;overflow:auto;line-height:1.65}pre code{background:none;padding:0;white-space:pre}
.table-wrap{overflow:auto;margin:20px 0}table{border-collapse:collapse;width:100%;min-width:580px;font-size:13px;line-height:1.7}td,th{border:1px solid #dce4ed;padding:10px 12px;vertical-align:top}th{background:#e9f1fa;text-align:left}tr:nth-child(even){background:#f8fafc}footer{font-size:12px;color:#5b6d83;margin-top:30px}
@media(max-width:950px){.layout{grid-template-columns:1fr}nav{position:static;max-height:230px}main{padding:24px}h1{font-size:25px}}
@media print{body{background:white;font-size:11pt}header,nav{display:none}.layout{display:block;margin:0;padding:0}main{border:0;padding:0}table{min-width:0;font-size:9pt}h2,h3{break-after:avoid}tr{break-inside:avoid}pre,pre code{white-space:pre-wrap}a{color:inherit}}
"""


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--date',default='2026-10-09')
    args=parser.parse_args()
    source=args.source.read_text(encoding='utf-8')
    converter=markdown.Markdown(extensions=['tables','fenced_code','toc','sane_lists'],
                                extension_configs={'toc':{'toc_depth':'2-3'}})
    body=converter.convert(source)
    body=body.replace('<table>','<div class="table-wrap"><table>').replace('</table>','</table></div>')
    title=html.escape(source.splitlines()[0].removeprefix('# '))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    md=args.output_dir/args.source.name
    if md.resolve()!=args.source.resolve():shutil.copyfile(args.source,md)
    page=md.with_suffix('.html')
    document=f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{STYLE}</style></head><body>
<header>{html.escape(args.date)} · a3-21 chip4 · 本轮工作与优化原理<br>
<a href="{html.escape(md.name)}">下载 Markdown 源文件</a><a href="http://117.72.247.67:18080/">links-server</a></header>
<div class="layout"><nav aria-label="报告目录"><strong>报告目录</strong>{converter.toc}</nav>
<main>{body}<footer>静态阅读版，无外部脚本或字体依赖；时序示意与硬件实测已在正文中分别说明。</footer></main></div>
</body></html>
'''
    page.write_text(document,encoding='utf-8')
    files=[]
    for path in [md,page]:
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_name(path.name+'.sha256').write_text(f'{digest}  {path.name}\n',encoding='utf-8')
        files.append({'file':path.name,'bytes':path.stat().st_size,'sha256':digest})
    print(json.dumps({'files':files,'sections':len(converter.toc_tokens)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
