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

        canvas.width = W * dpr;
        canvas.height = H * dpr;
        ctx.scale(dpr, dpr);

        const ML = 56, MR = 56;
        const TL = ML;
        const TW = W - ML - MR;
        const stripY = 32;
        const stripH = 26;
        const tlY = 98;
        const tlH = 7;
        const topY = tlY + tlH / 2 + 5;
        const markerDrop = 50;

        const streamDuration = 13500;
        const retroDuration = 18000;


        const C = {
            white: '#ffffff',
            bg: '#ffffff',
            tp: '#3a8c78',
            fp: '#b85c5c',
            fn: '#8a80b4',
            goldFill: 'rgba(165, 118, 45, 0.18)',
            goldFeatherL: 'rgba(165, 118, 45, 0)',
            goldFeatherLp: 'rgba(165, 118, 45, 0.26)',
            goldFeatherR0: 'rgba(165, 118, 45, 0.18)',
            goldFeatherR1: 'rgba(165, 118, 45, 0)',
            goldEdge: 'rgba(145, 100, 32, 0.26)',
            goldLabel: 'rgba(130, 90, 28, 0.82)',
            playhead: '#2c3a4a',
            glowPlay: 'rgba(44, 58, 74, 0.14)',
            pauseAccent: '#cc445c',
            glowPause: 'rgba(180, 60, 80, 0.20)',
            track: '#d4dce8',
            trackBorder: 'rgba(15, 23, 42, 0.06)',
            progStream: 'rgba(46, 130, 110, 0.11)',
            progRetro: 'rgba(180, 65, 88, 0.11)',
            text: '#374151',
            textSoft: '#6b7280',
            textXSoft: '#9ca3af',
            retroOverlay: 'rgba(175, 58, 78, 0.04)',
            retroDot: 'rgba(148, 160, 174, 0.45)',
            retroDotAct: '#cc445c',
            retroLabel: 'rgba(180, 60, 80, 0.72)',
            retroNotScored: 'rgba(175, 58, 78, 0.48)',
            bubbleQ: 'rgba(78, 110, 182, 0.72)',
            bubbleA: 'rgba(112, 84, 165, 0.72)',
            barBg: '#e8eef4',
            barGrad0: '#3a8c78',
            barGrad1: '#70b8a4',
            infoBox: '#f4f6f9',
            infoAccent: 'rgba(180, 60, 80, 0.70)',
        };

        const F = (size, weight) =>
            `${weight || 500} ${size}px 'Google Sans','Noto Sans',sans-serif`;


        function xf(frac) { return TL + frac * TW; }
        function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }
        function easeOut(v, p) { return 1 - Math.pow(1 - v, p || 2); }
        function easeIn(v, p) { return Math.pow(v, p || 2); }
        function easeInOut(v) { return v < 0.5 ? 2 * v * v : -1 + (4 - 2 * v) * v; }
        function easeOutBack(v) { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(v - 1, 3) + c1 * Math.pow(v - 1, 2); }
        function lerp(a, b, t) { return a + (b - a) * t; }
        function prog(v, s, e) { return clamp((v - s) / (e - s), 0, 1); }
        function smoothstep(e0, e1, x) { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); }
        function rrect(x, y, w, h, r) {
            const rx = typeof r === 'number' ? r : 5;
            ctx.moveTo(x + rx, y);
            ctx.arcTo(x + w, y, x + w, y + h, rx); ctx.arcTo(x + w, y + h, x, y + h, rx);
            ctx.arcTo(x, y + h, x, y, rx); ctx.arcTo(x, y, x + w, y, rx);
            ctx.closePath();
        }


        const frameColors = Array.from({ length: 80 }, (_, i) => {
            const hue = (205 + i * 4 + Math.sin(i * 0.35) * 12) % 360;
            const s = 20 + Math.abs(Math.sin(i * 0.45)) * 12;
            const l = 84 + Math.abs(Math.sin(i * 0.28 + 1)) * 7;
            return `hsl(${hue | 0},${s | 0}%,${l | 0}%)`;
        });

        function drawStrip(playFrac, paused) {
            const count = frameColors.length, fw = TW / count;
            for (let i = 0; i < count; i++) {
                ctx.globalAlpha = (i / count <= playFrac) ? 1 : 0.20;
                ctx.fillStyle = frameColors[i];
                ctx.fillRect(TL + i * fw, stripY, fw - 0.5, stripH);
            }
            ctx.globalAlpha = 1;
            ctx.strokeStyle = C.trackBorder; ctx.lineWidth = 1;
            ctx.strokeRect(TL, stripY, TW, stripH);
            if (paused) { const px = xf(playFrac); ctx.fillStyle = 'rgba(180,60,80,0.13)'; ctx.fillRect(px - 5, stripY, 10, stripH); }
        }
        function drawTrack() {
            ctx.fillStyle = C.track; ctx.beginPath(); rrect(TL, tlY - tlH / 2, TW, tlH, 4); ctx.fill();
            ctx.strokeStyle = C.trackBorder; ctx.lineWidth = 1; ctx.stroke();
        }
        function drawProgress(frac, color) {
            if (frac <= 0) return;
            ctx.fillStyle = color; ctx.beginPath(); rrect(TL, tlY - tlH / 2, TW * frac, tlH, 4); ctx.fill();
        }
        function drawPlayhead(frac, paused) {
            const px = xf(frac);
            const lineCol = paused ? C.pauseAccent : C.playhead;
            const glowCol = paused ? C.glowPause : C.glowPlay;
            ctx.save(); ctx.shadowColor = glowCol; ctx.shadowBlur = 8;
            ctx.strokeStyle = lineCol; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(px, stripY - 2); ctx.lineTo(px, tlY + tlH / 2 + 22); ctx.stroke();
            ctx.shadowBlur = 0;
            ctx.fillStyle = lineCol;
            ctx.beginPath(); ctx.moveTo(px - 6, stripY - 12); ctx.lineTo(px + 6, stripY - 12); ctx.lineTo(px, stripY - 2); ctx.closePath(); ctx.fill();
            if (paused) { ctx.fillStyle = C.pauseAccent; ctx.font = F(11, 600); ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText('PAUSE', px, stripY - 14); }
            ctx.restore();
        }
        function drawGoldBand(s, e, alpha) {
            const sx = xf(s), ex = xf(e), bw = ex - sx, by = tlY - tlH / 2 - 4, bh = tlH + 8;
            ctx.save(); ctx.globalAlpha = alpha;
            const lg = ctx.createLinearGradient(sx - 30, 0, sx + 6, 0);
            lg.addColorStop(0, C.goldFeatherL); lg.addColorStop(1, C.goldFeatherLp);
            ctx.fillStyle = lg; ctx.fillRect(sx - 30, by - 4, 36, bh + 8);
            ctx.fillStyle = C.goldFill; ctx.fillRect(sx, by - 4, bw, bh + 8);
            const rg = ctx.createLinearGradient(ex - 6, 0, ex + 18, 0);
            rg.addColorStop(0, C.goldFeatherR0); rg.addColorStop(1, C.goldFeatherR1);
            ctx.fillStyle = rg; ctx.fillRect(ex - 6, by - 4, 24, bh + 8);
            ctx.strokeStyle = C.goldEdge; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(sx, tlY - 16); ctx.lineTo(sx, tlY + 20); ctx.moveTo(ex, tlY - 16); ctx.lineTo(ex, tlY + 20); ctx.stroke();
            ctx.setLineDash([]); ctx.restore();
        }
        function drawMarker(xFrac, type, alpha, dropP) {
            const mx = xf(xFrac);
            const pal = { TP: { line: C.tp, bg: 'rgba(58,140,120,0.09)' }, FP: { line: C.fp, bg: 'rgba(184,92,92,0.09)' }, FN: { line: C.fn, bg: 'rgba(138,128,180,0.09)' } }[type];
            const dy = topY + markerDrop * easeOutBack(clamp(dropP, 0, 1));
            ctx.save(); ctx.globalAlpha = alpha;
            ctx.strokeStyle = pal.line; ctx.lineWidth = 1.5;
            if (type === 'FN') ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(mx, topY); ctx.lineTo(mx, dy - 11); ctx.stroke(); ctx.setLineDash([]);
            ctx.fillStyle = pal.bg; ctx.strokeStyle = pal.line; ctx.lineWidth = 1;
            ctx.beginPath(); rrect(mx - 20, dy - 10, 40, 22, 6); ctx.fill(); ctx.stroke();
            ctx.fillStyle = pal.line; ctx.font = F(11, 700); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(type, mx, dy + 1);
            ctx.beginPath(); ctx.arc(mx, tlY, 4, 0, Math.PI * 2); ctx.fill();
            ctx.restore();
        }
        function drawBubble(text, bx, by, bw, bh, bg, alpha, scaleP) {
            const scale = easeOutBack(clamp(scaleP, 0, 1));
            const cx = bx + bw / 2, cy = by + bh / 2;
            const words = text.split(' '); const maxW = bw - 20; let line = '', lines = [];
            ctx.save(); ctx.globalAlpha = alpha; ctx.translate(cx, cy); ctx.scale(scale, scale); ctx.translate(-cx, -cy);
            ctx.fillStyle = bg; ctx.beginPath(); rrect(bx, by, bw, bh, 10); ctx.fill();
            ctx.fillStyle = C.white; ctx.font = F(12, 600); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            words.forEach(w => { const test = line ? line + ' ' + w : w; if (ctx.measureText(test).width > maxW && line) { lines.push(line); line = w; } else { line = test; } });
            if (line) lines.push(line);
            const lh = 16, startY = by + bh / 2 - ((lines.length - 1) * lh) / 2;
            lines.forEach((l, i) => ctx.fillText(l, bx + bw / 2, startY + i * lh));
            ctx.restore();
        }
        function drawF1Bar(value, alpha) {
            const bx = TL, by = 278, bw = TW, bh = 17;
            ctx.save(); ctx.globalAlpha = alpha;
            ctx.fillStyle = C.barBg; ctx.beginPath(); rrect(bx, by, bw, bh, 7); ctx.fill();
            if (value > 0) {
                const g = ctx.createLinearGradient(bx, 0, bx + bw * value, 0);
                g.addColorStop(0, C.barGrad0); g.addColorStop(1, C.barGrad1);
                ctx.fillStyle = g; ctx.beginPath(); rrect(bx, by, bw * value, bh, 7); ctx.fill();
            }
            ctx.fillStyle = C.text; ctx.font = F(13, 700); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('Timeliness-F1', bx, by + bh + 7);
            ctx.fillStyle = C.text; ctx.font = F(13, 700); ctx.textAlign = 'right';
            ctx.fillText((value * 100).toFixed(1) + '%', bx + bw, by + bh + 7);
            ctx.restore();
        }

        const SLOT_HUE = [34, 218, 145];

        function sCol(idx, alpha) { return `hsla(${SLOT_HUE[idx]},52%,40%,${alpha})`; }
        function sFill(idx, alpha) { return `hsla(${SLOT_HUE[idx]},50%,46%,${alpha})`; }
        function sBub(idx) { return `hsla(${SLOT_HUE[idx]},55%,44%,0.88)`; }


        const QY = 94;
        const RY = 198;
        const BH = 32;
        const SIG_E = 28;
        const SIG_L = 14;


        const S_SLOTS = [
            { ts: 0.24, te: 0.31, qx: 0.12 },
            { ts: 0.49, te: 0.56, qx: 0.39 },
            { ts: 0.73, te: 0.80, qx: 0.63 },
        ];

        const S_RESP = [
            { xf: 0.055, label: 'FP', type: 'FP', at: 0.055 },
            { xf: 0.165, label: 'FP', type: 'FP', at: 0.165 },
            { xf: 0.195, label: 'xTP', type: 'xTP', at: 0.195, rLabel: 'r₁' },
            { xf: 0.385, label: 'FP', type: 'FP', at: 0.385 },
            { xf: 0.520, label: 'TP', type: 'TP', at: 0.520, rLabel: 'r₂' },
            { xf: 0.870, label: 'FP', type: 'FP', at: 0.870 },
        ];


        function rStyle(type) {
            switch (type) {
                case 'TP': return { bg: 'rgba(58,140,120,0.10)', bd: C.tp, tx: C.tp, dash: false };
                case 'FP': return { bg: 'rgba(184,92,92,0.08)', bd: C.fp, tx: C.fp, dash: true };
                case 'xTP': return { bg: 'rgba(80,130,100,0.10)', bd: 'hsla(155,35%,40%,0.9)', tx: 'hsla(155,38%,38%,1.0)', dash: false };
                default: return { bg: 'rgba(138,128,180,0.09)', bd: C.fn, tx: C.fn, dash: true };
            }
        }


        function drawBell(idx, pf) {
            const sl = S_SLOTS[idx];
            if (pf < sl.qx - 0.04) return;
            const alpha = smoothstep(sl.qx - 0.04, sl.qx + 0.08, pf);
            if (alpha < 0.01) return;

            const tsx = xf(sl.ts), tex = xf(sl.te);
            const rangeL = SIG_E * 2.8, rangeR = SIG_L * 2.8;
            const N = 64;

            ctx.save();
            ctx.globalAlpha = alpha;

            ctx.beginPath();
            ctx.moveTo(tsx - rangeL, QY);


            for (let i = 0; i <= N; i++) {
                const x = tsx - rangeL + i * (rangeL / N);
                const dx = x - tsx;
                const g = Math.exp(-(dx * dx) / (2 * SIG_E * SIG_E));
                ctx.lineTo(x, QY - BH * g);
            }

            ctx.lineTo(tex, QY - BH);


            for (let i = 1; i <= N; i++) {
                const x = tex + i * (rangeR / N);
                const dx = x - tex;
                const g = Math.exp(-(dx * dx) / (2 * SIG_L * SIG_L));
                ctx.lineTo(x, QY - BH * g);
            }
            ctx.lineTo(tex + rangeR, QY);
            ctx.closePath();

            const midX = (tsx + tex) / 2;
            const grd = ctx.createLinearGradient(midX, QY - BH, midX, QY);
            grd.addColorStop(0, sFill(idx, 0.22));
            grd.addColorStop(1, sFill(idx, 0.03));
            ctx.fillStyle = grd;
            ctx.fill();

            ctx.strokeStyle = sCol(idx, 0.30);
            ctx.lineWidth = 1.2;
            ctx.stroke();

            ctx.restore();
        }


        function drawFNAboveBell(idx, alpha) {
            if (alpha < 0.01) return;
            const sl = S_SLOTS[idx];
            const cx = xf((sl.ts + sl.te) / 2);
            const peakY = QY - BH;

            ctx.save();
            ctx.globalAlpha = alpha;


            ctx.strokeStyle = C.fn;
            ctx.lineWidth = 1.2;
            ctx.setLineDash([3, 2]);
            ctx.beginPath();
            ctx.moveTo(cx, peakY - 3);
            ctx.lineTo(cx, peakY - 20);
            ctx.stroke();
            ctx.setLineDash([]);


            ctx.fillStyle = C.fn;
            ctx.beginPath();
            ctx.moveTo(cx - 4, peakY - 3);
            ctx.lineTo(cx + 4, peakY - 3);
            ctx.lineTo(cx, peakY + 2);
            ctx.closePath();
            ctx.fill();


            ctx.font = F(12, 700);
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            ctx.fillText('FN', cx, peakY - 21);

            ctx.restore();
        }


        function drawQueryBubble(idx, pf) {
            const sl = S_SLOTS[idx];
            if (pf < sl.qx - 0.01) return;
            const alpha = smoothstep(sl.qx - 0.01, sl.qx + 0.04, pf);
            if (alpha < 0.01) return;
            const sc = easeOutBack(clamp(alpha, 0, 1));
            const qxPx = xf(sl.qx);
            const bw = 36, bh = 26;
            const bx = qxPx - bw / 2;
            const by = QY - bh - 7;

            ctx.save();
            ctx.globalAlpha = alpha;
            const cy = QY - bh / 2 - 7;
            ctx.translate(qxPx, cy); ctx.scale(sc, sc); ctx.translate(-qxPx, -cy);


            ctx.fillStyle = sBub(idx);
            ctx.beginPath(); rrect(bx, by, bw, bh, 6); ctx.fill();


            ctx.beginPath();
            ctx.moveTo(qxPx - 5, QY - 7);
            ctx.lineTo(qxPx, QY - 1);
            ctx.lineTo(qxPx + 5, QY - 7);
            ctx.closePath();
            ctx.fill();


            ctx.fillStyle = 'rgba(255,255,255,0.90)';
            [-7, 0, 7].forEach(dx => {
                ctx.beginPath();
                ctx.arc(qxPx + dx, by + bh / 2, 2.2, 0, Math.PI * 2);
                ctx.fill();
            });


            ctx.fillStyle = sCol(idx, 0.72);
            ctx.font = F(16, 700);
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText(`q${['₁', '₂', '₃'][idx]}`, qxPx, QY + 5);

            ctx.restore();
        }


        function drawRespBox(resp, pf) {
            if (pf < resp.at - 0.005) return;
            const age = pf - resp.at;
            const alpha = age < 0.022 ? easeOut(age / 0.022) : 1.0;
            const sc = easeOutBack(clamp(age / 0.032, 0, 1));
            if (alpha < 0.01) return;

            const rx = xf(resp.xf);
            const bw = resp.type === 'TPFP' ? 38 : 26;
            const bh = 14;
            const by = RY - bh / 2;
            const sty = rStyle(resp.type);

            ctx.save();
            ctx.globalAlpha = alpha;
            ctx.translate(rx, RY); ctx.scale(sc, sc); ctx.translate(-rx, -RY);


            ctx.fillStyle = sty.bg;
            ctx.strokeStyle = sty.bd;
            ctx.lineWidth = 1.2;
            if (sty.dash) ctx.setLineDash([3, 2]);
            ctx.beginPath(); rrect(rx - bw / 2, by, bw, bh, 5); ctx.fill(); ctx.stroke();
            ctx.setLineDash([]);


            ctx.fillStyle = sty.bd;
            ctx.font = F(9, 400);
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('· · ·', rx, RY);


            if (resp.rLabel) {
                ctx.fillStyle = sty.tx;
                ctx.font = F(16, 700);
                ctx.textBaseline = 'bottom';
                ctx.fillText(resp.rLabel, rx, by - 5);
            }


            ctx.fillStyle = sty.tx;
            ctx.font = F(11, 700);
            ctx.textBaseline = 'top';
            ctx.fillText(resp.label, rx, by + bh + 4);


            if (resp.type === 'xTP' || resp.type === 'TP') {
                const tScore = resp.type === 'TP' ? 1.0 : (() => {
                    const sl = S_SLOTS[0];
                    const dPx = (sl.ts - resp.xf) * TW;
                    return Math.exp(-(dPx * dPx) / (2 * SIG_E * SIG_E));
                })();
                const gw = bw, gh = 4;
                const gx = rx - gw / 2;
                const gy = by + bh + 20;
                ctx.fillStyle = 'rgba(80,140,105,0.14)';
                ctx.beginPath(); rrect(gx, gy, gw, gh, 2); ctx.fill();
                ctx.fillStyle = 'hsla(155,38%,42%,0.9)';
                ctx.beginPath(); rrect(gx, gy, gw * tScore, gh, 2); ctx.fill();
                ctx.fillStyle = C.textXSoft;
                ctx.font = F(9, 500);
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                const tLabel = resp.type === 'TP' ? 'T = 100%' : `T ≈ ${(tScore * 100) | 0}%`;
                ctx.fillText(tLabel, rx, gy + gh + 2);
            }

            ctx.restore();
        }


        function drawRespBell(resp, pf) {
            if (pf < resp.at - 0.005) return;
            const age = pf - resp.at;
            const alpha = age < 0.025 ? easeOut(age / 0.025) : 1.0;
            if (alpha < 0.01) return;

            const cx = xf(resp.xf);
            const maxH = 28 * resp.bellH;
            const sigma = 18;
            const range = sigma * 2.6;
            const N = 60;
            const isTP = resp.type === 'TP';
            const fillA = isTP ? 0.18 : 0.13;
            const fillCol = isTP ? `rgba(58,140,120,${fillA})` : `rgba(80,130,100,${fillA})`;
            const fillColLo = isTP ? 'rgba(58,140,120,0.02)' : 'rgba(80,130,100,0.02)';
            const strokeCol = isTP ? C.tp : 'hsla(155,35%,40%,0.80)';
            const labelCol = isTP ? C.tp : 'hsla(155,35%,38%,1.0)';

            ctx.save();
            ctx.globalAlpha = alpha;

            ctx.beginPath();
            ctx.moveTo(cx - range, RY);
            for (let i = 0; i <= N; i++) {
                const x = cx - range + i * (2 * range / N);
                const dx = x - cx;
                const g = Math.exp(-(dx * dx) / (2 * sigma * sigma));
                ctx.lineTo(x, RY - maxH * g);
            }
            ctx.lineTo(cx + range, RY);
            ctx.closePath();

            const grd = ctx.createLinearGradient(cx, RY - maxH, cx, RY);
            grd.addColorStop(0, fillCol);
            grd.addColorStop(1, fillColLo);
            ctx.fillStyle = grd;
            ctx.fill();

            ctx.strokeStyle = strokeCol;
            ctx.lineWidth = 1.2;
            if (!isTP) ctx.setLineDash([3, 2]);
            ctx.stroke();
            ctx.setLineDash([]);


            ctx.fillStyle = labelCol;
            ctx.font = F(11, 700);
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            ctx.fillText(resp.label, cx, RY - maxH - 4);

            ctx.restore();
        }


        function drawSlotSpan(idx, pf) {
            const sl = S_SLOTS[idx];
            const a = smoothstep(sl.qx, sl.qx + 0.06, pf);
            if (a < 0.01) return;
            const tsx = xf(sl.ts), tex = xf(sl.te), bw = tex - tsx;

            ctx.save();


            ctx.globalAlpha = a * 0.13;
            const grd = ctx.createLinearGradient(0, QY, 0, RY);
            grd.addColorStop(0, sFill(idx, 0.35));
            grd.addColorStop(1, sFill(idx, 0.10));
            ctx.fillStyle = grd;
            ctx.fillRect(tsx, QY, bw, RY - QY);


            ctx.globalAlpha = a * 0.25;
            ctx.strokeStyle = sCol(idx, 1.0);
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            ctx.moveTo(tsx, QY); ctx.lineTo(tsx, RY + 16);
            ctx.moveTo(tex, QY); ctx.lineTo(tex, RY + 16);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.restore();
        }

        function drawStreaming(t) {
            ctx.clearRect(0, 0, W, H);
            ctx.fillStyle = C.bg;
            ctx.fillRect(0, 0, W, H);

            const STREAM_END = 0.43;
            const SCORE_START = 0.47;
            const SCORE_END = 0.67;
            const SCORES_APPEAR = 0.78;

            const pf = t < 0.39
                ? prog(t, 0, 0.39) * 0.97
                : t < STREAM_END
                    ? lerp(0.97, 1.0, easeInOut(prog(t, 0.39, STREAM_END)))
                    : 1.0;

            const sf = t < SCORE_START ? 0
                : t >= SCORE_END ? 1.0
                    : prog(t, SCORE_START, SCORE_END);
            const scoringDone = t >= SCORE_END;


            function revealFrac(resp) {
                if (sf <= resp.xf) return 0;
                return smoothstep(resp.xf, Math.min(resp.xf + 0.05, 1.0), sf);
            }


            S_SLOTS.forEach((sl, i) => drawBell(i, pf));


            S_SLOTS.forEach((sl, i) => drawSlotSpan(i, pf));


            ctx.fillStyle = C.textXSoft; ctx.font = F(11, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText('query stream', TL, QY - 5);
            ctx.fillStyle = C.track;
            ctx.fillRect(TL, QY - 1.5, TW, 3);
            if (pf > 0) { ctx.fillStyle = C.progStream; ctx.fillRect(TL, QY - 1.5, TW * pf, 3); }


            S_SLOTS.forEach((sl, i) => drawQueryBubble(i, pf));


            ctx.fillStyle = C.textXSoft; ctx.font = F(11, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText('response stream', TL, RY - 5);
            ctx.fillStyle = C.track;
            ctx.fillRect(TL, RY - 1.5, TW, 3);


            S_RESP.forEach(r => {
                if (pf < r.at - 0.005) return;
                const age = pf - r.at;
                const popAlpha = age < 0.022 ? easeOut(age / 0.022) : 1.0;
                const popSc = easeOutBack(clamp(age / 0.032, 0, 1));
                if (popAlpha < 0.01) return;

                const rx = xf(r.xf);
                const bw = 26, bh = 14, by = RY - bh / 2;
                const rv = revealFrac(r);

                ctx.save();
                ctx.globalAlpha = popAlpha;
                ctx.translate(rx, RY); ctx.scale(popSc, popSc); ctx.translate(-rx, -RY);


                if (rv < 1.0) {
                    ctx.save();
                    ctx.globalAlpha = 1.0 - rv * 0.9;
                    ctx.strokeStyle = '#606060'; ctx.lineWidth = 1.2; ctx.setLineDash([3, 2]);
                    ctx.beginPath(); rrect(rx - bw / 2, by, bw, bh, 5); ctx.stroke();
                    ctx.setLineDash([]);
                    ctx.fillStyle = '#777'; ctx.font = F(9, 400); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                    ctx.fillText('· · ·', rx, RY);
                    ctx.restore();
                }


                if (rv > 0) {
                    const sty = rStyle(r.type);
                    ctx.save();
                    ctx.globalAlpha = rv;
                    ctx.fillStyle = sty.bg; ctx.strokeStyle = sty.bd; ctx.lineWidth = 1.2;
                    if (sty.dash) ctx.setLineDash([3, 2]);
                    ctx.beginPath(); rrect(rx - bw / 2, by, bw, bh, 5); ctx.fill(); ctx.stroke();
                    ctx.setLineDash([]);
                    ctx.fillStyle = sty.bd; ctx.font = F(9, 400); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                    ctx.fillText('· · ·', rx, RY);
                    if (r.rLabel) {
                        ctx.fillStyle = sty.tx; ctx.font = F(16, 700); ctx.textBaseline = 'bottom';
                        ctx.fillText(r.rLabel, rx, by - 5);
                    }
                    ctx.fillStyle = sty.tx; ctx.font = F(11, 700); ctx.textBaseline = 'top';
                    ctx.fillText(r.label, rx, by + bh + 4);
                    if (r.type === 'xTP' || r.type === 'TP') {
                        const tScore = r.type === 'TP' ? 1.0 : (() => {
                            const sl = S_SLOTS[0];
                            const dPx = (sl.ts - r.xf) * TW;
                            return Math.exp(-(dPx * dPx) / (2 * SIG_E * SIG_E));
                        })();
                        const gw = bw, gh = 4, gx = rx - gw / 2, gy = by + bh + 20;
                        ctx.fillStyle = 'rgba(80,140,105,0.14)';
                        ctx.beginPath(); rrect(gx, gy, gw, gh, 2); ctx.fill();
                        ctx.fillStyle = 'hsla(155,38%,42%,0.9)';
                        ctx.beginPath(); rrect(gx, gy, gw * tScore, gh, 2); ctx.fill();
                        ctx.fillStyle = C.textXSoft; ctx.font = F(9, 500); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
                        ctx.fillText(r.type === 'TP' ? 'T = 100%' : `T ≈ ${(tScore * 100) | 0}%`, rx, gy + gh + 2);
                    }
                    ctx.restore();
                }

                ctx.restore();
            });


            if (scoringDone) {
                const b3 = S_SLOTS[2];
                const fnX = xf((b3.ts + b3.te) / 2);
                const fa = smoothstep(SCORE_END, SCORE_END + 0.03, t);
                ctx.save();
                ctx.globalAlpha = fa;
                ctx.strokeStyle = C.fn; ctx.lineWidth = 1.2; ctx.setLineDash([3, 2]);
                ctx.beginPath(); rrect(fnX - 13, RY - 7, 26, 14, 4); ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = C.fn;
                ctx.font = F(9, 400); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillText('· · ·', fnX, RY);
                ctx.font = F(11, 700); ctx.textBaseline = 'top';
                ctx.fillText('FN', fnX, RY + 10);
                ctx.restore();
            }


            {
                const streamA = t < STREAM_END ? 1.0
                    : t < SCORE_START ? lerp(1.0, 0, prog(t, STREAM_END, SCORE_START))
                        : 0;
                if (streamA > 0.01) {
                    const px = xf(pf);
                    ctx.save();
                    ctx.globalAlpha = streamA;
                    ctx.shadowColor = C.glowPlay; ctx.shadowBlur = 8;
                    ctx.strokeStyle = C.playhead; ctx.lineWidth = 1.8;
                    ctx.beginPath(); ctx.moveTo(px, QY - 14); ctx.lineTo(px, RY + 22); ctx.stroke();
                    ctx.shadowBlur = 0;
                    ctx.fillStyle = C.playhead;
                    ctx.beginPath(); ctx.moveTo(px - 5, QY - 22); ctx.lineTo(px + 5, QY - 22); ctx.lineTo(px, QY - 14); ctx.closePath(); ctx.fill();
                    ctx.restore();
                }
            }


            if (t >= STREAM_END && !scoringDone) {
                const amber = '#d97706';
                const beamA = smoothstep(STREAM_END, SCORE_START, t);
                const spx = xf(sf);
                ctx.save();
                ctx.globalAlpha = beamA;
                ctx.shadowColor = 'rgba(217,119,6,0.22)'; ctx.shadowBlur = 8;
                ctx.strokeStyle = amber; ctx.lineWidth = 1.8;
                ctx.beginPath(); ctx.moveTo(spx, QY - 14); ctx.lineTo(spx, RY + 22); ctx.stroke();
                ctx.shadowBlur = 0;
                ctx.fillStyle = amber;
                ctx.beginPath(); ctx.moveTo(spx - 5, QY - 22); ctx.lineTo(spx + 5, QY - 22); ctx.lineTo(spx, QY - 14); ctx.closePath(); ctx.fill();
                ctx.font = F(10, 700); ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
                ctx.fillText('SCORING', spx, QY - 24);
                ctx.restore();
            }


            {
                const iconX = 28, iconY = (QY + RY) / 2;
                const isScoring = t >= SCORE_START;
                const iconColor = isScoring ? '#d97706' : C.tp;
                ctx.save();
                ctx.fillStyle = isScoring ? 'rgba(217,119,6,0.10)' : 'rgba(46,130,110,0.10)';
                ctx.strokeStyle = iconColor; ctx.lineWidth = 1.5;
                ctx.beginPath(); ctx.arc(iconX, iconY, 14, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
                ctx.fillStyle = iconColor;
                if (isScoring) {
                    ctx.beginPath(); ctx.moveTo(iconX - 4, iconY - 7); ctx.lineTo(iconX + 8, iconY); ctx.lineTo(iconX - 4, iconY + 7); ctx.closePath(); ctx.fill();
                    ctx.font = F(9, 700); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
                    ctx.fillText('SCORE', iconX, iconY + 18);
                } else {
                    ctx.fillRect(iconX - 6, iconY - 7, 4, 14);
                    ctx.fillRect(iconX + 2, iconY - 7, 4, 14);
                    ctx.font = F(9, 700); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
                    ctx.fillText('PLAY', iconX, iconY + 18);
                }
                ctx.restore();
            }


            if (t >= SCORE_START) {
                let TP_full = 0, TP_frac = 0, xTP_count = 0, FP_count = 0;
                S_RESP.forEach(r => {
                    if (sf < r.xf) return;
                    if (r.type === 'TP') { TP_full++; }
                    else if (r.type === 'xTP') {
                        xTP_count++;
                        const sl = S_SLOTS[0];
                        const dPx = (sl.ts - r.xf) * TW;
                        TP_frac += Math.exp(-(dPx * dPx) / (2 * SIG_E * SIG_E));
                    }
                    else if (r.type === 'FP') FP_count++;
                });
                const TP_count = TP_full + xTP_count;
                const xTP_val = TP_full + TP_frac;
                const FN_count = Math.max(0, S_SLOTS.length - TP_count);
                const prec = xTP_val / Math.max(1, TP_count + FP_count);
                const rec = xTP_val / Math.max(1, TP_count + FN_count);
                const f1 = prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0;


                if (sf > S_RESP[0].xf || scoringDone) {
                    const statA = smoothstep(SCORE_START + 0.01, SCORE_START + 0.06, t);
                    ctx.save();
                    ctx.globalAlpha = statA;
                    ctx.fillStyle = C.textSoft; ctx.font = F(12, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
                    ctx.fillText(
                        `xTP = ${xTP_val.toFixed(2)}   ·   FP = ${FP_count}   ·   FN = ${scoringDone ? FN_count : 0}`,
                        TL, 262
                    );
                    ctx.restore();
                }

                if (t >= SCORES_APPEAR) {
                    const f1Alpha = smoothstep(SCORES_APPEAR, SCORES_APPEAR + 0.03, t);
                    const tScore = xTP_val / S_SLOTS.length;
                    ctx.save();
                    ctx.globalAlpha = f1Alpha;
                    ctx.fillStyle = C.text; ctx.font = F(13, 700); ctx.textAlign = 'right'; ctx.textBaseline = 'top';
                    ctx.fillText(`Timeliness-F1 = ${(f1 * 100).toFixed(1)}%`, TL + TW, 255);
                    ctx.fillText(`Timeliness-score = ${(tScore * 100).toFixed(1)}%`, TL + TW, 272);
                    ctx.restore();
                }
            }


            const ly = 308;


            {
                const lx = TL;
                const miniW = 32, miniH = 10;
                const leftW = miniW * 0.50, flatW = miniW * 0.22, rightW = miniW * 0.28;
                const peakL = lx + leftW, peakR = peakL + flatW;
                const sigL = leftW / 2.8, sigR = rightW / 2.5;
                const N = 24;
                const baseline = ly + miniH / 2;

                ctx.save();
                ctx.beginPath();
                ctx.moveTo(lx, baseline);
                for (let j = 0; j <= N; j++) {
                    const x = lx + j * (leftW / N);
                    const g = Math.exp(-Math.pow(x - peakL, 2) / (2 * sigL * sigL));
                    ctx.lineTo(x, baseline - miniH * g);
                }
                ctx.lineTo(peakR, baseline - miniH);
                for (let j = 1; j <= N; j++) {
                    const x = peakR + j * (rightW / N);
                    const g = Math.exp(-Math.pow(x - peakR, 2) / (2 * sigR * sigR));
                    ctx.lineTo(x, baseline - miniH * g);
                }
                ctx.lineTo(lx + miniW, baseline);


                const grd = ctx.createLinearGradient(lx + miniW / 2, baseline - miniH, lx + miniW / 2, baseline);
                grd.addColorStop(0, 'rgba(100,110,125,0.42)');
                grd.addColorStop(1, 'rgba(100,110,125,0.06)');
                ctx.fillStyle = grd;
                ctx.fill();
                ctx.strokeStyle = 'rgba(65,75,92,0.62)';
                ctx.lineWidth = 1.2;
                ctx.stroke();
                ctx.restore();

                ctx.fillStyle = C.textSoft; ctx.font = F(12, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
                ctx.fillText('Timeliness-score > 0', lx + miniW + 6, ly);
            }


            const lineItems = [
                { col: C.tp, lbl: 'TP: timely & correct', dash: false },
                { col: C.fp, lbl: 'FP: spurious / mistimed', dash: true },
                { col: C.fn, lbl: 'FN: missed slot', dash: true },
            ];
            lineItems.forEach((item, i) => {
                const lx = TL + (i + 1) * (TW / 4);
                ctx.save();
                ctx.strokeStyle = item.col; ctx.lineWidth = 2;
                if (item.dash) ctx.setLineDash([3, 2]);
                ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 18, ly); ctx.stroke();
                ctx.setLineDash([]); ctx.restore();
                ctx.fillStyle = C.textSoft; ctx.font = F(12, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
                ctx.fillText(item.lbl, lx + 24, ly);
            });
        }

        const R_PAUSES = [
            { pf: 0.28, t0: 0.12, t1: 0.26, qLabel: 'q₁', rLabel: 'r₁', correct: true },
            { pf: 0.52, t0: 0.36, t1: 0.50, qLabel: 'q₂', rLabel: 'r₂', correct: false },
            { pf: 0.76, t0: 0.60, t1: 0.74, qLabel: 'q₃', rLabel: 'r₃', correct: true },
        ];
        const R_PAUSE_ENDS = [0.26, 0.50, 0.74];

        function getRetroState(t) {
            let pf = 0, paused = false, evIdx = -1;
            const qA = [0, 0, 0], qSc = [0, 0, 0], rA = [0, 0, 0];

            function setPause(i) {
                paused = true; evIdx = i; pf = R_PAUSES[i].pf;
                const lc = prog(t, R_PAUSES[i].t0, R_PAUSES[i].t1);

                qA[i] = lc < 0.15 ? easeOut(prog(lc, 0, 0.15)) : 1.0;
                qSc[i] = lc < 0.18 ? easeOutBack(prog(lc, 0, 0.18)) : 1.0;

                rA[i] = lc < 0.40 ? 0 : smoothstep(0.40, 0.52, lc);
            }

            if (t < 0.12) { pf = lerp(0, 0.28, prog(t, 0, 0.12)); }
            else if (t < 0.26) { setPause(0); }
            else if (t < 0.36) { pf = lerp(0.28, 0.52, prog(t, 0.26, 0.36)); }
            else if (t < 0.50) { setPause(1); }
            else if (t < 0.60) { pf = lerp(0.52, 0.76, prog(t, 0.50, 0.60)); }
            else if (t < 0.74) { setPause(2); }
            else if (t < 0.84) { pf = lerp(0.76, 1.0, prog(t, 0.74, 0.84)); }
            else { pf = 1.0; }


            R_PAUSES.forEach((_, i) => {
                if (i === evIdx) return;
                if (t > R_PAUSE_ENDS[i]) { qA[i] = 1.0; qSc[i] = 1.0; rA[i] = 1.0; }
            });

            return { pf, paused, evIdx, qA, qSc, rA };
        }

        function drawRetrospective(t) {
            ctx.clearRect(0, 0, W, H);
            ctx.fillStyle = C.bg;
            ctx.fillRect(0, 0, W, H);

            const { pf, paused, evIdx, qA, qSc, rA } = getRetroState(t);


            {
                const count = 80, fw = TW / count;
                for (let i = 0; i < count; i++) {
                    const l = 78 + Math.abs(Math.sin(i * 0.28 + 1)) * 12;
                    ctx.fillStyle = `hsl(0,0%,${l | 0}%)`;
                    ctx.globalAlpha = (i / count <= pf ? 1.0 : 0.22);
                    ctx.fillRect(TL + i * fw, stripY, fw - 0.5, stripH);
                }
                ctx.globalAlpha = 1;
                ctx.strokeStyle = C.trackBorder; ctx.lineWidth = 1;
                ctx.strokeRect(TL, stripY, TW, stripH);
                if (paused) {
                    const px = xf(pf);
                    ctx.fillStyle = 'rgba(180,60,80,0.13)';
                    ctx.fillRect(px - 5, stripY, 10, stripH);
                }
            }


            ctx.fillStyle = C.textXSoft; ctx.font = F(11, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText('query stream', TL, QY - 5);
            ctx.fillStyle = C.track;
            ctx.fillRect(TL, QY - 1.5, TW, 3);
            if (pf > 0) { ctx.fillStyle = C.progRetro; ctx.fillRect(TL, QY - 1.5, TW * pf, 3); }


            R_PAUSES.forEach((ev, i) => {
                if (pf < ev.pf - 0.04) return;
                const a = smoothstep(ev.pf - 0.04, ev.pf + 0.01, pf);
                ctx.save(); ctx.globalAlpha = a;
                ctx.fillStyle = (paused && evIdx === i) ? C.pauseAccent : sCol(i, 0.55);
                const tx = xf(ev.pf);
                ctx.beginPath(); ctx.moveTo(tx, QY - 6); ctx.lineTo(tx + 4, QY); ctx.lineTo(tx, QY + 6); ctx.lineTo(tx - 4, QY); ctx.closePath(); ctx.fill();
                ctx.restore();
            });


            ctx.fillStyle = C.textXSoft; ctx.font = F(11, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText('response stream', TL, RY - 5);
            ctx.fillStyle = C.track;
            ctx.fillRect(TL, RY - 1.5, TW, 3);


            R_PAUSES.forEach((ev, i) => {
                const epx = xf(ev.pf);


                if (qA[i] > 0.01) {
                    const qbw = 36, qbh = 26, qby = QY - qbh - 7;
                    ctx.save();
                    ctx.globalAlpha = qA[i];
                    const qcy = QY - qbh / 2 - 7;
                    ctx.translate(epx, qcy); ctx.scale(qSc[i], qSc[i]); ctx.translate(-epx, -qcy);
                    ctx.fillStyle = sBub(i);
                    ctx.beginPath(); rrect(epx - qbw / 2, qby, qbw, qbh, 6); ctx.fill();
                    ctx.beginPath(); ctx.moveTo(epx - 5, QY - 7); ctx.lineTo(epx, QY - 1); ctx.lineTo(epx + 5, QY - 7); ctx.closePath(); ctx.fill();
                    ctx.fillStyle = 'rgba(255,255,255,0.90)';
                    [-7, 0, 7].forEach(dx => { ctx.beginPath(); ctx.arc(epx + dx, qby + qbh / 2, 2.2, 0, Math.PI * 2); ctx.fill(); });
                    ctx.fillStyle = sCol(i, 0.85); ctx.font = F(16, 700); ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
                    ctx.fillText(ev.qLabel, epx, qby - 5);
                    ctx.restore();
                }


                if (rA[i] > 0.01) {
                    const col = ev.correct ? C.tp : C.fp;
                    const rSc = rA[i] < 1.0 ? easeOutBack(clamp(prog(rA[i], 0, 0.8), 0, 1)) : 1.0;
                    const bw = 26, bh = 14, by = RY - bh / 2;
                    ctx.save();
                    ctx.globalAlpha = rA[i];
                    ctx.translate(epx, RY); ctx.scale(rSc, rSc); ctx.translate(-epx, -RY);


                    ctx.fillStyle = ev.correct ? 'rgba(58,140,120,0.10)' : 'rgba(184,92,92,0.08)';
                    ctx.strokeStyle = col;
                    ctx.lineWidth = 1.2;
                    if (!ev.correct) ctx.setLineDash([3, 2]);
                    ctx.beginPath(); rrect(epx - bw / 2, by, bw, bh, 5); ctx.fill(); ctx.stroke();
                    ctx.setLineDash([]);


                    ctx.fillStyle = col; ctx.font = F(9, 400); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                    ctx.fillText('· · ·', epx, RY);


                    ctx.font = F(16, 700); ctx.textBaseline = 'top';
                    ctx.fillText(ev.rLabel, epx, by + bh + 5);

                    ctx.font = F(13, 700);
                    ctx.fillText(ev.correct ? '✓' : '✗', epx, by + bh + 32);

                    ctx.font = F(10, 600);
                    ctx.fillText(ev.correct ? 'Correct' : 'Incorrect', epx, by + bh + 47);
                    ctx.restore();
                }
            });


            const iconX = 28, iconY = (QY + RY) / 2;
            ctx.save();
            ctx.fillStyle = paused ? 'rgba(204,68,92,0.10)' : 'rgba(46,130,110,0.10)';
            ctx.strokeStyle = paused ? C.pauseAccent : C.tp;
            ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.arc(iconX, iconY, 14, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
            ctx.fillStyle = paused ? C.pauseAccent : C.tp;
            if (paused) {

                ctx.beginPath(); ctx.moveTo(iconX - 4, iconY - 7); ctx.lineTo(iconX + 8, iconY); ctx.lineTo(iconX - 4, iconY + 7); ctx.closePath(); ctx.fill();
            } else {

                ctx.fillRect(iconX - 6, iconY - 7, 4, 14);
                ctx.fillRect(iconX + 2, iconY - 7, 4, 14);
            }
            ctx.font = F(9, 700); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            ctx.fillText(paused ? 'PAUSE' : 'PLAY', iconX, iconY + 18);
            ctx.restore();


            const px = xf(pf);
            ctx.save();
            ctx.shadowColor = paused ? C.glowPause : C.glowPlay; ctx.shadowBlur = 8;
            ctx.strokeStyle = paused ? C.pauseAccent : C.playhead; ctx.lineWidth = 1.8;
            if (paused) {

                ctx.beginPath(); ctx.moveTo(px, QY + 2); ctx.lineTo(px, RY - 2); ctx.stroke();
            } else {

                ctx.beginPath(); ctx.moveTo(px, QY - 14); ctx.lineTo(px, RY + 22); ctx.stroke();
                ctx.shadowBlur = 0;
                ctx.fillStyle = C.playhead;
                ctx.beginPath(); ctx.moveTo(px - 5, QY - 22); ctx.lineTo(px + 5, QY - 22); ctx.lineTo(px, QY - 14); ctx.closePath(); ctx.fill();
            }
            ctx.restore();


            const correctCount = R_PAUSES.filter((ev, i) => rA[i] > 0.5 && ev.correct).length;
            const incorrectCount = R_PAUSES.filter((ev, i) => rA[i] > 0.5 && !ev.correct).length;

            if (pf > 0.20) {
                const a = smoothstep(0.20, 0.28, pf);
                ctx.save(); ctx.globalAlpha = a;
                ctx.fillStyle = C.textSoft; ctx.font = F(12, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
                ctx.fillText(`Correct = ${correctCount}   ·   Incorrect = ${incorrectCount}`, TL, 262);
                ctx.restore();

                if (t >= 0.84) {
                    const total = correctCount + incorrectCount;
                    const acc = total > 0 ? correctCount / total : 0;
                    ctx.save();
                    ctx.globalAlpha = smoothstep(0.84, 0.87, t) * a;
                    ctx.fillStyle = C.text; ctx.font = F(13, 700); ctx.textAlign = 'right'; ctx.textBaseline = 'top';
                    ctx.fillText(`Accuracy = ${(acc * 100).toFixed(1)}%`, TL + TW, 262);
                    ctx.restore();
                }
            }


            const rLegend = [
                { col: C.tp, sym: '✓', lbl: 'Correct' },
                { col: C.fp, sym: '✗', lbl: 'Incorrect' },
            ];
            const ly = 308;
            rLegend.forEach((item, i) => {
                const lx = TL + i * (TW / 4);
                ctx.fillStyle = item.col; ctx.font = F(14, 700); ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
                ctx.fillText(item.sym, lx, ly);
                ctx.fillStyle = C.textSoft; ctx.font = F(12, 500);
                ctx.fillText(item.lbl, lx + 20, ly);
            });
        }

        function tick(ts) {
            if (!animationStart) animationStart = ts;
            const dur = mode === 'streaming' ? streamDuration : retroDuration;
            const t = ((ts - animationStart) % dur) / dur;
            mode === 'streaming' ? drawStreaming(t) : drawRetrospective(t);
            animationFrame = requestAnimationFrame(tick);
        }
        window.addEventListener('beforeunload', () => { if (animationFrame) cancelAnimationFrame(animationFrame); });
        animationFrame = requestAnimationFrame(tick);
    }

    document.addEventListener('DOMContentLoaded', initializeEvaluationCanvas);
})();