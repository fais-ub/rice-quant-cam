"""Fig. 3 and Fig. 4 of the paper at print size (one column)."""
import csv
import math
import os
from pathlib import Path

import pdfplumber
from reportlab.lib.colors import HexColor, black
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

DATA = Path(os.environ.get("JOIV_DATA", Path.home() / "joiv-data")).expanduser()
OUT = DATA / "results/paper_final/print"
OUT.mkdir(parents=True, exist_ok=True)
FONT_DIR = Path(os.environ.get("FONT_DIR", "C:/Windows/Fonts"))
for name, fn in [('Times', 'times.ttf'), ('TimesBold', 'timesbd.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / fn)))
W = 86 / 25.4 * 72


def read(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def label(c, x, y, value, size=8, align='left', bold=False):
    c.setFillColor(black)
    c.setFont('TimesBold' if bold else 'Times', size)
    {'left': c.drawString, 'center': c.drawCentredString, 'right': c.drawRightString}[align](x, y, value)


def lineplot(c, pts, color, lw=.65):
    c.setStrokeColor(HexColor(color))
    c.setLineWidth(lw)
    p = c.beginPath()
    for i, (x, y) in enumerate(pts):
        (p.moveTo if i == 0 else p.lineTo)(x, y)
    c.drawPath(p)


def finish(c, path):
    c.save()
    with pdfplumber.open(path) as pdf:
        pdf.pages[0].to_image(resolution=600).save(path.with_suffix('.png'))


def layer_error():
    h = 350
    path = OUT / 'Fig3_layer_error.pdf'
    c = canvas.Canvas(str(path), pagesize=(W, h))
    rows = read(DATA / 'results/paper_pc/mixed_precision/layer_error/layer_error_all.csv')
    for x, tag, color in [(24, 'MinMax (FR)', '#222222'), (109, 'Percentile', '#d39000')]:
        c.setStrokeColor(HexColor(color)); c.setLineWidth(.8); c.line(x, h-10, x+12, h-10)
        label(c, x+16, h-12, tag, 8)
    c.setFillColor(HexColor('#dfdfdf')); c.rect(190, h-13, 7, 7, fill=1, stroke=0)
    label(c, 201, h-12, 'SE', 8)
    label(c, 69, h-30, 'INT8 steps (central 99.9%)', 8, 'center')
    label(c, 187, h-30, 'Cumulative SQNR (dB)', 8, 'center')
    models = [('effb0', 'EN-B0'), ('mnv3l', 'MNv3-L'), ('mnv2', 'MNv2'), ('effl0', 'EN-Lite0')]
    for mi, (model, short) in enumerate(models):
        y0 = 28 + (3-mi)*72
        ph = 53
        model_rows = [r for r in rows if r['model'] == model and r['qmodel'] in ('minmax', 'pct')]
        ordered = {tag: sorted([r for r in model_rows if r['qmodel'] == tag], key=lambda r: int(r['order']))
                   for tag in ('minmax', 'pct')}
        n = len(ordered['minmax'])
        assert n == len(ordered['pct'])
        for pi in range(2):
            x0 = 30 + pi*118
            pw = 88
            xx = lambda i: x0 + pw*i/(n-1)
            if pi == 0:
                yy = lambda v: y0+ph*(math.log10(max(.5, min(300, v)))-math.log10(.5))/(math.log10(300)-math.log10(.5))
                ticks = [(1, '1'), (10, '10'), (100, '100')]
            else:
                yy = lambda v: y0+ph*(max(-5, min(60, v))+5)/65
                ticks = [(0, '0'), (20, '20'), (40, '40'), (60, '60')]
            for i, r in enumerate(ordered['minmax']):
                if r['role'].startswith('se'):
                    c.setFillColor(HexColor('#e1e1e1'))
                    c.rect(xx(i)-.45, y0, .9, ph, stroke=0, fill=1)
            for val, text in ticks:
                c.setStrokeColor(HexColor('#e5e5e5')); c.setLineWidth(.3)
                c.line(x0, yy(val), x0+pw, yy(val))
                label(c, x0-3, yy(val)-2.5, text, 7.5, 'right')
            for tag, color in [('minmax', '#222222'), ('pct', '#d39000')]:
                key = 'steps_p999' if pi == 0 else 'cum_sqnr_db'
                pts = [(xx(i), yy(float(r[key]))) for i, r in enumerate(ordered[tag])]
                lineplot(c, pts, color, .6)
            c.setStrokeColor(black); c.setLineWidth(.4)
            c.line(x0, y0, x0+pw, y0); c.line(x0, y0, x0, y0+ph)
            xticks = [0, 100, 200] if n > 200 else ([0, 50, 100] if n > 100 else [0, 25, 50])
            for tick in xticks:
                label(c, xx(tick), y0-10, str(tick), 7.5, 'center')
        c.saveState(); c.translate(8, y0+ph/2); c.rotate(90)
        label(c, 0, 0, short, 8.5, 'center', True); c.restoreState()
    label(c, W/2, 3, 'Activation tensor (graph order)', 8, 'center')
    finish(c, path)


def threads():
    h = 300
    path = OUT / 'Fig4_threads.pdf'
    c = canvas.Canvas(str(path), pagesize=(W, h))
    rows = read(DATA / 'results/paper_final/data/phone_latency_long_final.csv')
    phones = ['Redmi Note 11 Pro 5G', 'Moto g45 5G', 'OPPO Reno12 F 4G', 'Xiaomi 13T']
    colors = ['#0072b2', '#d39000', '#888888', '#000000']
    for i, text in enumerate(['Redmi', 'Moto', 'OPPO Reno12 F 4G', 'Xiaomi 13T']):
        x = 12+(i%2)*117; y = h-10-(i//2)*12
        c.setStrokeColor(HexColor(colors[i])); c.setLineWidth(.8); c.line(x, y, x+11, y)
        label(c, x+15, y-2, text, 8.5)
    label(c, W/2, h-38, 'Solid: CPU    Dashed: XNNPACK', 8, 'center')
    models = [('mnv2', 'MNv2'), ('mnv3l', 'MNv3-L'), ('effl0', 'EN-Lite0'), ('effb0', 'EN-B0')]
    for mi, (model, short) in enumerate(models):
        for pi, precision in enumerate(['int8_qdq_pct', 'fp32']):
            x0 = 31+pi*112; y0 = 24+(3-mi)*60; pw=91; ph=39
            sub=[r for r in rows if r['model'] == model+'_s0' and r['precision'] == precision]
            dmax=max(float(r['median_ms']) for r in sub)
            step=next(s*10**k for k in range(4) for s in (1,2,2.5,5) if math.ceil(dmax/(s*10**k)) <= 3)
            nt=math.ceil(dmax/step); ymax=nt*step
            xx=lambda t:x0+pw*(t-1)/3
            yy=lambda v:y0+ph*v/ymax
            if mi==0: label(c,x0+pw/2,y0+ph+8,'QDQ-Pct' if pi==0 else 'FP32',8.5,'center')
            for j in range(nt+1):
                val=j*step
                c.setStrokeColor(HexColor('#e1e1e1')); c.setLineWidth(.3); c.line(x0,yy(val),x0+pw,yy(val))
                label(c,x0-3,yy(val)-2.5,f'{val:g}',7.5,'right')
            for phone,color in zip(phones,colors):
                for ep in ['CPU','XNNPACK']:
                    points=sorted([r for r in sub if r['phone']==phone and r['ep']==ep],key=lambda r:int(r['threads']))
                    c.setDash(2,2) if ep=='XNNPACK' else c.setDash()
                    lineplot(c,[(xx(int(r['threads'])),yy(float(r['median_ms']))) for r in points],color,.75)
                    c.setDash()
                    for r in points:
                        c.setFillColor(HexColor(color)); c.circle(xx(int(r['threads'])),yy(float(r['median_ms'])),1.3,stroke=0,fill=1)
            c.setStrokeColor(black); c.setLineWidth(.4); c.line(x0,y0,x0+pw,y0)
            for t in [1,2,4]: label(c,xx(t),y0-10,str(t),7.5,'center')
            if mi==3: label(c,x0+pw/2,y0-21,'Threads',8,'center')
        c.saveState(); c.translate(8,y0+ph/2); c.rotate(90)
        label(c,0,0,short+' (ms)',8,'center'); c.restoreState()
    finish(c,path)


if __name__ == '__main__':
    layer_error()
    threads()
    print('Generated Fig. 3 and Fig. 4, vector PDF and 600 dpi PNG.')
