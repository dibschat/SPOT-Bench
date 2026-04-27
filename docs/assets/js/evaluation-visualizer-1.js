(function () {
    var mode = 'streaming';
    var animationStart = null;
    var animationFrame = null;

    window.switchEvalMode = function (newMode) {
        mode = newMode;
        animationStart = null;
    };

    function initializeEvaluationCanvas() {
        const canvas = document.getElementById('spot-eval-canvas');
        if (!canvas) return;

        mode = canvas.dataset.evalMode || 'streaming';
        const ctx = canvas.getContext('2d');
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const W = 960;
        const H = 360;

        canvas.width  = W * dpr;
        canvas.height = H * dpr;
        ctx.scale(dpr, dpr);

        // ── Layout constants ────────────────────────────────────────────────
        const ML = 56, MR = 56;
        const TL = ML;                  // timeline left x
        const TW = W - ML - MR;        // timeline width = 848
        const stripY  = 32;
        const stripH  = 26;             // strip: 32–58
        const tlY     = 98;             // timeline centre y
        const tlH     = 7;             // timeline bar height: 94–101
        const topY    = tlY + tlH / 2 + 5;   // top of marker stems ≈ 108
        const markerDrop = 50;          // max drop distance for markers

        const streamDuration = 11000;
        const retroDuration  =  9000;

        // ── Muted, website-consistent colour palette ────────────────────────
        // Site accents: streaming #17805d, paused #cc445c
        // Body text: #374151 / #6b7280 / #111827
        const C = {
            white:        '#ffffff',
            bg:           '#ffffff',

            // Evaluation markers – muted, not vivid
            tp:           '#3a8c78',          // muted teal-green
            fp:           '#b85c5c',          // muted rose-red
            fn:           '#8a80b4',          // muted lavender

            // Gold slot bands – muted amber
            goldFill:     'rgba(165, 118, 45, 0.18)',
            goldFeatherL: 'rgba(165, 118, 45, 0)',
            goldFeatherLp:'rgba(165, 118, 45, 0.26)',
            goldFeatherR0:'rgba(165, 118, 45, 0.18)',
            goldFeatherR1:'rgba(165, 118, 45, 0)',
            goldEdge:     'rgba(145, 100, 32, 0.26)',
            goldLabel:    'rgba(130, 90, 28, 0.82)',

            // Playheads
            playhead:     '#2c3a4a',
            glowPlay:     'rgba(44, 58, 74, 0.14)',
            pauseAccent:  '#cc445c',          // site paused colour
            glowPause:    'rgba(180, 60, 80, 0.20)',

            // Timeline track
            track:        '#d4dce8',
            trackBorder:  'rgba(15, 23, 42, 0.06)',
            progStream:   'rgba(46, 130, 110, 0.11)',
            progRetro:    'rgba(180, 65, 88, 0.11)',

            // Frame strip
            stripBorder:  'rgba(15, 23, 42, 0.07)',
            pauseSliver:  'rgba(180, 60, 80, 0.13)',

            // Body text (matching website)
            text:         '#374151',
            textSoft:     '#6b7280',
            textXSoft:    '#9ca3af',

            // Retrospective
            retroOverlay: 'rgba(175, 58, 78, 0.04)',
            retroDot:     'rgba(148, 160, 174, 0.45)',
            retroDotAct:  '#cc445c',
            retroLabel:   'rgba(180, 60, 80, 0.72)',
            retroNotScored:'rgba(175, 58, 78, 0.48)',

            // Chat bubbles – muted
            bubbleQ:      'rgba(78, 110, 182, 0.72)',
            bubbleA:      'rgba(112, 84, 165, 0.72)',

            // F1 bar
            barBg:        '#e8eef4',
            barGrad0:     '#3a8c78',
            barGrad1:     '#70b8a4',

            // Info box
            infoBox:      '#f4f6f9',
            infoAccent:   'rgba(180, 60, 80, 0.70)',
        };

        const F = (size, weight) =>
            `${weight || 500} ${size}px 'Google Sans','Noto Sans',sans-serif`;

        // ── Helpers ─────────────────────────────────────────────────────────
        function xf(frac) { return TL + frac * TW; }
        function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }
        function easeOut(v, p)  { return 1 - Math.pow(1 - v, p || 2); }
        function easeIn(v, p)   { return Math.pow(v, p || 2); }
        function easeInOut(v)   { return v < 0.5 ? 2*v*v : -1+(4-2*v)*v; }
        function easeOutBack(v) {
            const c1 = 1.70158, c3 = c1 + 1;
            return 1 + c3*Math.pow(v-1, 3) + c1*Math.pow(v-1, 2);
        }
        function lerp(a, b, t)  { return a + (b-a)*t; }
        function prog(v, s, e)  { return clamp((v-s)/(e-s), 0, 1); }
        function smoothstep(e0, e1, x) {
            const t = clamp((x-e0)/(e1-e0), 0, 1);
            return t*t*(3-2*t);
        }
        function rrect(x, y, w, h, r) {
            const rx = (typeof r === 'number') ? r : 5;
            ctx.moveTo(x+rx, y);
            ctx.arcTo(x+w, y, x+w, y+h, rx);
            ctx.arcTo(x+w, y+h, x, y+h, rx);
            ctx.arcTo(x, y+h, x, y, rx);
            ctx.arcTo(x, y, x+w, y, rx);
            ctx.closePath();
        }

        // ── Frame strip ─────────────────────────────────────────────────────
        // Softer, lighter pastels than before
        const frameColors = Array.from({ length: 80 }, (_, i) => {
            const hue = (205 + i*4 + Math.sin(i*0.35)*12) % 360;
            const s   = 20 + Math.abs(Math.sin(i*0.45))*12;
            const l   = 84 + Math.abs(Math.sin(i*0.28+1))*7;
            return `hsl(${hue|0},${s|0}%,${l|0}%)`;
        });

        function drawStrip(playFrac, paused) {
            const count = frameColors.length;
            const fw = TW / count;
            for (let i = 0; i < count; i++) {
                ctx.globalAlpha = (i/count <= playFrac) ? 1 : 0.20;
                ctx.fillStyle = frameColors[i];
                ctx.fillRect(TL + i*fw, stripY, fw - 0.5, stripH);
            }
            ctx.globalAlpha = 1;
            ctx.strokeStyle = C.stripBorder;
            ctx.lineWidth = 1;
            ctx.beginPath();
            rrect(TL, stripY, TW, stripH, 8);
            ctx.stroke();
            if (paused) {
                const px = xf(playFrac);
                ctx.fillStyle = C.pauseSliver;
                ctx.fillRect(px - 5, stripY, 10, stripH);
            }
        }

        function drawTrack() {
            ctx.fillStyle = C.track;
            ctx.beginPath();
            rrect(TL, tlY - tlH/2, TW, tlH, 4);
            ctx.fill();
            ctx.strokeStyle = C.trackBorder;
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        function drawProgress(frac, color) {
            if (frac <= 0) return;
            ctx.fillStyle = color;
            ctx.beginPath();
            rrect(TL, tlY - tlH/2, TW*frac, tlH, 4);
            ctx.fill();
        }

        function drawPlayhead(frac, paused) {
            const px = xf(frac);
            const lineCol = paused ? C.pauseAccent : C.playhead;
            const glowCol = paused ? C.glowPause   : C.glowPlay;
            ctx.save();
            ctx.shadowColor = glowCol;
            ctx.shadowBlur  = 8;
            ctx.strokeStyle = lineCol;
            ctx.lineWidth   = 2;
            ctx.beginPath();
            ctx.moveTo(px, stripY - 2);
            ctx.lineTo(px, tlY + tlH/2 + 22);
            ctx.stroke();
            ctx.shadowBlur = 0;
            // Arrow triangle
            ctx.fillStyle = lineCol;
            ctx.beginPath();
            ctx.moveTo(px - 6, stripY - 12);
            ctx.lineTo(px + 6, stripY - 12);
            ctx.lineTo(px,     stripY - 2);
            ctx.closePath();
            ctx.fill();
            if (paused) {
                ctx.fillStyle = C.pauseAccent;
                ctx.font = F(11, 600);
                ctx.textAlign = 'center';
                ctx.textBaseline = 'bottom';
                ctx.fillText('PAUSE', px, stripY - 14);
            }
            ctx.restore();
        }

        function drawGoldBand(s, e, alpha) {
            const sx = xf(s), ex = xf(e), bw = ex - sx;
            const by = tlY - tlH/2 - 4, bh = tlH + 8;
            ctx.save();
            ctx.globalAlpha = alpha;
            const lg = ctx.createLinearGradient(sx-30, 0, sx+6, 0);
            lg.addColorStop(0, C.goldFeatherL);
            lg.addColorStop(1, C.goldFeatherLp);
            ctx.fillStyle = lg;
            ctx.fillRect(sx-30, by-4, 36, bh+8);
            ctx.fillStyle = C.goldFill;
            ctx.fillRect(sx, by-4, bw, bh+8);
            const rg = ctx.createLinearGradient(ex-6, 0, ex+18, 0);
            rg.addColorStop(0, C.goldFeatherR0);
            rg.addColorStop(1, C.goldFeatherR1);
            ctx.fillStyle = rg;
            ctx.fillRect(ex-6, by-4, 24, bh+8);
            ctx.strokeStyle = C.goldEdge;
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(sx, tlY-16); ctx.lineTo(sx, tlY+20);
            ctx.moveTo(ex, tlY-16); ctx.lineTo(ex, tlY+20);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.restore();
        }

        function drawMarker(xFrac, type, alpha, dropP) {
            const mx = xf(xFrac);
            const pal = {
                TP: { line: C.tp, bg: 'rgba(58,140,120,0.09)' },
                FP: { line: C.fp, bg: 'rgba(184,92,92,0.09)'  },
                FN: { line: C.fn, bg: 'rgba(138,128,180,0.09)'},
            }[type];
            const dy = topY + markerDrop * easeOutBack(clamp(dropP, 0, 1));
            ctx.save();
            ctx.globalAlpha = alpha;
            ctx.strokeStyle = pal.line;
            ctx.lineWidth = 1.5;
            if (type === 'FN') ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(mx, topY);
            ctx.lineTo(mx, dy - 11);
            ctx.stroke();
            ctx.setLineDash([]);
            // Label box
            ctx.fillStyle = pal.bg;
            ctx.strokeStyle = pal.line;
            ctx.lineWidth = 1;
            ctx.beginPath();
            rrect(mx - 20, dy - 10, 40, 22, 6);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = pal.line;
            ctx.font = F(11, 700);
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(type, mx, dy + 1);
            // Dot on timeline
            ctx.beginPath();
            ctx.arc(mx, tlY, 4, 0, Math.PI*2);
            ctx.fill();
            ctx.restore();
        }

        function drawBubble(text, bx, by, bw, bh, bg, alpha, scaleP) {
            const scale = easeOutBack(clamp(scaleP, 0, 1));
            const cx = bx + bw/2, cy = by + bh/2;
            const words = text.split(' ');
            const maxW  = bw - 20;
            let line = '', lines = [];
            ctx.save();
            ctx.globalAlpha = alpha;
            ctx.translate(cx, cy);
            ctx.scale(scale, scale);
            ctx.translate(-cx, -cy);
            ctx.fillStyle = bg;
            ctx.beginPath();
            rrect(bx, by, bw, bh, 10);
            ctx.fill();
            ctx.fillStyle = C.white;
            ctx.font = F(12, 600);
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            words.forEach(w => {
                const test = line ? line + ' ' + w : w;
                if (ctx.measureText(test).width > maxW && line) {
                    lines.push(line); line = w;
                } else {
                    line = test;
                }
            });
            if (line) lines.push(line);
            const lh = 16;
            const startY = by + bh/2 - ((lines.length-1)*lh)/2;
            lines.forEach((l, i) => ctx.fillText(l, bx+bw/2, startY+i*lh));
            ctx.restore();
        }

        function drawF1Bar(value, alpha) {
            const bx = TL, by = 216, bw = TW, bh = 15;
            ctx.save();
            ctx.globalAlpha = alpha;
            ctx.fillStyle = C.barBg;
            ctx.beginPath();
            rrect(bx, by, bw, bh, 6);
            ctx.fill();
            if (value > 0) {
                const g = ctx.createLinearGradient(bx, 0, bx+bw*value, 0);
                g.addColorStop(0, C.barGrad0);
                g.addColorStop(1, C.barGrad1);
                ctx.fillStyle = g;
                ctx.beginPath();
                rrect(bx, by, bw*value, bh, 6);
                ctx.fill();
            }
            ctx.fillStyle = C.textSoft;
            ctx.font = F(12, 500);
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText('Timeliness-F1', bx, by+bh+8);
            ctx.fillStyle = C.text;
            ctx.textAlign = 'right';
            ctx.fillText((value*100).toFixed(1)+'%', bx+bw, by+bh+8);
            ctx.restore();
        }

        // ── Streaming mode ───────────────────────────────────────────────────

        const sBands = [
            { start: 0.19, end: 0.29, label: 'slot 1' },
            { start: 0.48, end: 0.58, label: 'slot 2' },
            { start: 0.72, end: 0.82, label: 'slot 3' },
        ];
        const sResponses = [
            { xf: 0.09,  type: 'FP', at: 0.09  },
            { xf: 0.235, type: 'TP', at: 0.235 },
            { xf: 0.38,  type: 'FP', at: 0.38  },
            { xf: 0.53,  type: 'TP', at: 0.53  },
            { xf: 0.65,  type: 'FP', at: 0.65  },
        ];

        function drawStreaming(t) {
            ctx.clearRect(0, 0, W, H);
            ctx.fillStyle = C.bg;
            ctx.fillRect(0, 0, W, H);

            const pf = t < 0.93
                ? t * (1/0.93) * 0.97
                : lerp(0.97, 1.0, easeInOut(prog(t, 0.93, 1.0)));

            drawStrip(pf, false);
            drawTrack();
            drawProgress(pf, C.progStream);

            // Gold bands + slot labels
            sBands.forEach(b => {
                const center = (b.start + b.end) / 2;
                const alpha  = pf < b.start-0.08 ? 0.22
                             : pf < b.end+0.05   ? 1
                             :                     0.34;
                drawGoldBand(b.start, b.end, alpha);
                ctx.globalAlpha = alpha;
                ctx.fillStyle   = C.goldLabel;
                ctx.font        = F(12, 600);
                ctx.textAlign   = 'center';
                ctx.textBaseline = 'bottom';
                ctx.fillText(b.label, xf(center), tlY - 16);
                ctx.globalAlpha = 1;
            });

            // Response markers
            let TP = 0, FP = 0;
            sResponses.forEach(r => {
                if (pf < r.at - 0.005) return;
                const age   = pf - r.at;
                const alpha = age < 0.02 ? easeOut(age/0.02) : 1;
                const drop  = clamp(age/0.035, 0, 1);
                drawMarker(r.xf, r.type, alpha, drop);
                if (r.type === 'TP') TP++;
                if (r.type === 'FP') FP++;
            });

            // FN for missed third slot
            const b3 = sBands[2];
            if (pf > b3.end + 0.025) {
                const age   = pf - (b3.end + 0.025);
                const alpha = clamp(age/0.025, 0, 1);
                const fnX   = (b3.start + b3.end) / 2;
                drawMarker(fnX, 'FN', alpha, 1);
                ctx.save();
                ctx.globalAlpha = alpha * 0.40;
                ctx.strokeStyle = C.fn;
                ctx.lineWidth   = 1.5;
                ctx.setLineDash([4, 3]);
                ctx.beginPath();
                rrect(xf(b3.start)-10, tlY-14, xf(b3.end)-xf(b3.start)+20, 28, 5);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.restore();
            }

            drawPlayhead(pf, false);

            // F1 bar + counts
            const slotsPassed = sBands.filter(b => pf > b.end+0.01).length;
            const FN   = slotsPassed - TP;
            const prec = TP / Math.max(1, TP+FP);
            const rec  = slotsPassed > 0 ? TP / Math.max(1, TP+FN) : 0;
            const f1   = prec+rec > 0 ? 2*prec*rec/(prec+rec) : 0;

            if (pf > 0.06) {
                const a = smoothstep(0.06, 0.14, pf);
                drawF1Bar(f1, a);
                ctx.globalAlpha = a;
                ctx.fillStyle   = C.textSoft;
                ctx.font        = F(12, 500);
                ctx.textAlign   = 'left';
                ctx.textBaseline = 'top';
                ctx.fillText(`TP = ${TP}   ·   FP = ${FP}   ·   FN = ${FN}`, TL, 252);
                ctx.globalAlpha = 1;
            }

            // Legend – single row, 4 items
            const legend = [
                { color: 'rgba(155,110,38,0.82)', label: 'gold interval',         dashed: false },
                { color: C.tp,                    label: 'TP: timely & correct',  dashed: false },
                { color: C.fp,                    label: 'FP: spurious response', dashed: false },
                { color: C.fn,                    label: 'FN: missed slot',        dashed: true  },
            ];
            const ly = 298;
            legend.forEach((item, i) => {
                const lx = TL + i*(TW/4);
                ctx.save();
                ctx.strokeStyle = item.color;
                ctx.lineWidth   = 2;
                if (item.dashed) ctx.setLineDash([3, 2]);
                ctx.beginPath();
                ctx.moveTo(lx, ly);
                ctx.lineTo(lx + 18, ly);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.restore();
                ctx.fillStyle   = C.textSoft;
                ctx.font        = F(12, 500);
                ctx.textAlign   = 'left';
                ctx.textBaseline = 'middle';
                ctx.fillText(item.label, lx + 24, ly);
            });
        }

        // ── Retrospective mode ───────────────────────────────────────────────

        const rEvents = [
            { frac: 0.28, question: 'When does the action start?', response: 'start', label: 't1' },
            { frac: 0.65, question: 'Did the object state change?', response: 'now',   label: 't2' },
        ];

        function getRetroState(t) {
            let pf, paused=false, evIdx=-1, bAlpha=0, bScale=0;
            if (t < 0.17) {
                pf = lerp(0, rEvents[0].frac, easeInOut(prog(t, 0, 0.17)));
            } else if (t < 0.48) {
                pf = rEvents[0].frac; paused=true; evIdx=0;
                const lc = prog(t, 0.17, 0.48);
                if      (lc < 0.15) { bAlpha=easeOut(prog(lc,0,0.15)); bScale=easeOutBack(prog(lc,0,0.18)); }
                else if (lc > 0.8)  { bAlpha=1-easeIn(prog(lc,0.8,1)); bScale=1-0.15*easeIn(prog(lc,0.85,1)); }
                else                { bAlpha=1; bScale=1; }
            } else if (t < 0.65) {
                pf = lerp(rEvents[0].frac, rEvents[1].frac, easeInOut(prog(t, 0.48, 0.65)));
            } else if (t < 0.89) {
                pf = rEvents[1].frac; paused=true; evIdx=1;
                const lc = prog(t, 0.65, 0.89);
                if      (lc < 0.15) { bAlpha=easeOut(prog(lc,0,0.15)); bScale=easeOutBack(prog(lc,0,0.18)); }
                else if (lc > 0.8)  { bAlpha=1-easeIn(prog(lc,0.8,1)); bScale=1-0.15*easeIn(prog(lc,0.85,1)); }
                else                { bAlpha=1; bScale=1; }
            } else {
                pf = lerp(rEvents[1].frac, 1, easeInOut(prog(t, 0.89, 1)));
            }
            return { pf, paused, evIdx, bAlpha, bScale };
        }

        function drawRetrospective(t) {
            ctx.clearRect(0, 0, W, H);
            ctx.fillStyle = C.bg;
            ctx.fillRect(0, 0, W, H);

            const { pf, paused, evIdx, bAlpha, bScale } = getRetroState(t);
            const activeEv = evIdx >= 0 ? rEvents[evIdx] : null;

            // Subtle rose tint on strip
            ctx.save();
            ctx.globalAlpha = 0.04;
            ctx.fillStyle = C.pauseAccent;
            ctx.fillRect(TL, stripY, TW, stripH);
            ctx.restore();

            drawStrip(pf, paused);
            drawTrack();
            drawProgress(pf, C.progRetro);

            // "Not scored" annotation between strip and timeline (when playing)
            if (!paused) {
                const midX = (xf(rEvents[0].frac) + xf(rEvents[1].frac)) / 2;
                if (pf > rEvents[0].frac+0.04 && pf < rEvents[1].frac-0.04) {
                    ctx.fillStyle    = C.retroNotScored;
                    ctx.font         = F(11, 500);
                    ctx.textAlign    = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText('responses here are not scored', midX, stripY + stripH + 10);
                }
            }

            // Pause dots + t-labels
            rEvents.forEach((ev, i) => {
                const ex     = xf(ev.frac);
                const active = (i === evIdx && paused);
                ctx.fillStyle = active ? C.retroDotAct : C.retroDot;
                ctx.beginPath();
                ctx.arc(ex, tlY, 5, 0, Math.PI*2);
                ctx.fill();
                ctx.fillStyle    = active ? C.retroLabel : C.textXSoft;
                ctx.font         = F(12, 600);
                ctx.textAlign    = 'center';
                ctx.textBaseline = 'top';
                ctx.fillText(ev.label, ex, tlY + 10);
            });

            drawPlayhead(pf, paused);

            // Chat bubbles during pause
            if (activeEv && bAlpha > 0.01) {
                const px  = xf(activeEv.frac);
                const qx  = clamp(px - 162, TL, TL + TW - 330);
                const rx  = qx + 155;

                // Dashed connector from playhead to bubble
                ctx.save();
                ctx.globalAlpha  = bAlpha * 0.28;
                ctx.strokeStyle  = C.pauseAccent;
                ctx.lineWidth    = 1;
                ctx.setLineDash([3, 4]);
                ctx.beginPath();
                ctx.moveTo(px, tlY + 8);
                ctx.lineTo(px, 132);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.restore();

                drawBubble(activeEv.question, qx,     132, 148, 60, C.bubbleQ, bAlpha, bScale);
                const rScale = clamp(bScale > 0.6 ? (bScale-0.4)/0.6 : 0, 0, 1);
                drawBubble(activeEv.response, rx,     142,  88, 40, C.bubbleA, bAlpha*rScale, rScale);

                // OK badge on timeline dot
                if (bScale > 0.7) {
                    const sa = bAlpha * smoothstep(0.7, 0.9, bScale);
                    ctx.save();
                    ctx.globalAlpha = sa;
                    ctx.fillStyle   = C.tp;
                    ctx.beginPath();
                    ctx.arc(px, tlY, 7, 0, Math.PI*2);
                    ctx.fill();
                    ctx.fillStyle    = C.white;
                    ctx.font         = F(9, 700);
                    ctx.textAlign    = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText('OK', px, tlY + 0.5);
                    ctx.restore();
                }
            }

            // Protocol info box
            const boxY = 222, boxH = 60;
            ctx.fillStyle = C.infoBox;
            ctx.beginPath();
            rrect(TL, boxY, TW, boxH, 8);
            ctx.fill();

            ctx.fillStyle    = C.infoAccent;
            ctx.font         = F(12, 700);
            ctx.textAlign    = 'left';
            ctx.textBaseline = 'middle';
            ctx.fillText('Fixed-point protocol', TL + 16, boxY + 18);

            const evalCount = rEvents.filter(ev => ev.frac <= pf).length;
            ctx.fillStyle    = C.textSoft;
            ctx.font         = F(12, 500);
            ctx.fillText(
                `Pause ${evalCount} / 2  ·  only paused timestamps are scored  ·  responses between pauses are not evaluated`,
                TL + 16, boxY + 40
            );
        }

        // ── Animation loop ───────────────────────────────────────────────────
        function tick(ts) {
            if (!animationStart) animationStart = ts;
            const dur = mode === 'streaming' ? streamDuration : retroDuration;
            const t   = ((ts - animationStart) % dur) / dur;
            if (mode === 'streaming') {
                drawStreaming(t);
            } else {
                drawRetrospective(t);
            }
            animationFrame = requestAnimationFrame(tick);
        }

        window.addEventListener('beforeunload', () => {
            if (animationFrame) cancelAnimationFrame(animationFrame);
        });

        animationFrame = requestAnimationFrame(tick);
    }

    document.addEventListener('DOMContentLoaded', initializeEvaluationCanvas);
})();