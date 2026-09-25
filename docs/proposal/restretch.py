"""Content was laid out to a 5.6in canvas but the slide is 7.5in tall.
Stretch everything below the title band so the fullest slide ends at 6.95in,
keeping relative positions (and therefore text inside its card) intact."""
from pptx import Presentation
EMU = 914400
SRC_TOP, DST_TOP, K = 1.72, 1.95, 1.185   # solved so max bottom lands at 6.95

pr = Presentation('fpgAI-proposal.pptx')
moved = 0
for s in pr.slides:
    for sh in s.shapes:
        if sh.top is None or sh.height is None:
            continue
        t = sh.top / EMU
        if t < 1.6:                      # title band and the chip marker stay put
            continue
        sh.top = int(round((DST_TOP + (t - SRC_TOP) * K) * EMU))
        sh.height = int(round(sh.height * K))
        moved += 1
pr.save('fpgAI-proposal.pptx')
print('rescaled %d shapes' % moved)

pr = Presentation('fpgAI-proposal.pptx')
H = pr.slide_height / EMU; W = pr.slide_width / EMU
bad = 0
for i, s in enumerate(pr.slides, 1):
    mx = 0
    for sh in s.shapes:
        if sh.top is None or sh.height is None: continue
        b = (sh.top + sh.height) / EMU; r = (sh.left + sh.width) / EMU
        mx = max(mx, b)
        if b > H - 0.3 or r > W - 0.3:
            print('  slide %d TOO CLOSE TO EDGE: bottom=%.2f right=%.2f' % (i, b, r)); bad += 1
    print('slide %2d ends y=%.2f  (%.0f%%)' % (i, mx, 100*mx/H))
print('\nedge problems:', bad)
