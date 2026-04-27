(function () {
    'use strict';

    const W = 960, H = 290;

    const N = 5;
    const FH = 100;
    const FW = Math.round(FH * 140 / 132);
    const FGAP = 8;
    const BAND = 13;

    const FRAMES_W = N * FW + (N - 1) * FGAP;
    const FRAME_X0 = Math.round((W - FRAMES_W) / 2);
    const FRAME_X_END = FRAME_X0 + FRAMES_W;

    const STRIP_Y_RETRO = 36;
    const STRIP_Y_PROACT = 54;

    const HOLE_W = 7, HOLE_H = 11, HOLE_SPACING = 27;
    const SCROLL_PX_PER_MS = 0.04;

    const C = {
        bg: '#ffffff',
        white: '#ffffff',
        textSoft: '#6b7280',
        textXSoft: '#9ca3af',
        track: '#d4dce8',
        glowPlay: 'rgba(44,58,74,0.14)',
        retroAcct: '#cc445c',
        proactAcct: '#3a8c78',
        userCol: '#4a6cb3',
        modelCol: '#2c3a4a',
        filmBand: '#1c1c1e',
        filmHole: '#dde3ec',
    };

    const F = (size, weight) =>
        `${weight || 500} ${size}px 'Google Sans','Noto Sans',sans-serif`;


    function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }
    function easeOutBack(v) { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(v - 1, 3) + c1 * Math.pow(v - 1, 2); }
    function prog(v, s, e) { return clamp((v - s) / (e - s), 0, 1); }
    function smoothstep(e0, e1, x) { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); }

    function rrect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + w, y, x + w, y + h, r);
        ctx.arcTo(x + w, y + h, x, y + h, r);
        ctx.arcTo(x, y + h, x, y, r);
        ctx.arcTo(x, y, x + w, y, r);
        ctx.closePath();
    }

    function wrapText(ctx, text, cx, cy, maxW, lh) {
        const words = text.split(' ');
        let line = '', lines = [];
        words.forEach(w => {
            const test = line ? line + ' ' + w : w;
            if (ctx.measureText(test).width > maxW && line) { lines.push(line); line = w; }
            else line = test;
        });
        if (line) lines.push(line);
        const sy = cy - ((lines.length - 1) * lh) / 2;
        lines.forEach((l, i) => ctx.fillText(l, cx, sy + i * lh));
    }


    const frameImgs = Array.from({ length: N }, (_, i) => {
        const img = new Image();
        img.src = `assets/images/t${i + 1}.png`;
        return img;
    });


    function drawFrame(ctx, fi, x, y, w, h, alpha) {
        ctx.save();
        ctx.globalAlpha = alpha;
        rrect(ctx, x, y, w, h, 3);
        ctx.clip();
        const img = frameImgs[fi];
        if (img && img.complete && img.naturalWidth > 0) {
            const sc = Math.max(w / img.naturalWidth, h / img.naturalHeight);
            const sw = img.naturalWidth * sc, sh = img.naturalHeight * sc;
            ctx.drawImage(img, x + (w - sw) / 2, y + (h - sh) / 2, sw, sh);
        } else {
            const hue = (200 + fi * 28) % 360;
            ctx.fillStyle = `hsl(${hue},16%,86%)`;
            ctx.fillRect(x, y, w, h);
            ctx.fillStyle = `hsl(${hue},20%,58%)`;
            ctx.font = F(9, 600); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(`t${fi + 1}`, x + w / 2, y + h / 2);
        }
        ctx.restore();
    }




    function drawFilmStrip(ctx, stripY, scrollOffset) {
        const frameY = stripY + BAND;
        const botBandY = frameY + FH;


        ctx.fillStyle = C.filmBand;
        ctx.fillRect(0, stripY, W, BAND);
        ctx.fillRect(0, botBandY, W, BAND);


        const holeY_top = stripY + (BAND - HOLE_H) / 2;
        const holeY_bot = botBandY + (BAND - HOLE_H) / 2;


        const startX = -(scrollOffset % HOLE_SPACING) - HOLE_SPACING;

        ctx.fillStyle = C.filmHole;
        for (let x = startX; x < W + HOLE_SPACING; x += HOLE_SPACING) {
            const hx = Math.round(x);
            if (hx + HOLE_W < 0 || hx > W) continue;


            ctx.save();
            ctx.beginPath(); ctx.rect(0, stripY, W, BAND); ctx.clip();
            rrect(ctx, hx, holeY_top, HOLE_W, HOLE_H, 2); ctx.fill();
            ctx.restore();


            ctx.save();
            ctx.beginPath(); ctx.rect(0, botBandY, W, BAND); ctx.clip();
            rrect(ctx, hx, holeY_bot, HOLE_W, HOLE_H, 2); ctx.fill();
            ctx.restore();
        }
    }


    function drawFrames(ctx, stripY, alphas) {
        const frameY = stripY + BAND;
        for (let i = 0; i < N; i++) {
            const fx = FRAME_X0 + i * (FW + FGAP);
            drawFrame(ctx, i, fx, frameY, FW, FH, alphas[i]);
        }
    }


    function drawGlasses(ctx, cx, cy, scale, color, alpha) {
        const r = 9 * scale, gap = 6 * scale, tLen = 13 * scale;
        const lx = cx - gap - r, rx = cx + gap + r;
        ctx.save(); ctx.globalAlpha = alpha;
        ctx.strokeStyle = color; ctx.lineWidth = 1.8 * scale; ctx.lineCap = 'round';
        [lx, rx].forEach(lx2 => {
            ctx.beginPath(); ctx.arc(lx2, cy, r, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(44,58,74,0.07)'; ctx.fill(); ctx.stroke();
        });
        ctx.beginPath(); ctx.moveTo(lx + r, cy - r * 0.30); ctx.lineTo(rx - r, cy - r * 0.30); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(lx - r, cy); ctx.lineTo(lx - r - tLen, cy + tLen * 0.25); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(rx + r, cy); ctx.lineTo(rx + r + tLen, cy + tLen * 0.25); ctx.stroke();
        ctx.restore();
    }


    function drawUser(ctx, cx, cy, scale, color, alpha) {
        ctx.save(); ctx.globalAlpha = alpha; ctx.fillStyle = color;
        const hR = 9 * scale;
        ctx.beginPath(); ctx.arc(cx, cy - hR * 1.1, hR, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath();
        ctx.moveTo(cx - hR * 1.1, cy + hR * 1.8);
        ctx.quadraticCurveTo(cx - hR * 1.5, cy, cx, cy + hR * 0.2);
        ctx.quadraticCurveTo(cx + hR * 1.5, cy, cx + hR * 1.1, cy + hR * 1.8);
        ctx.closePath(); ctx.fill();
        ctx.restore();
    }

    const RETRO_DUR = 14000;
    const RIGHT_X = 930;

    function drawRetro(t, ctx, scroll) {
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);

        const STRIP_Y = STRIP_Y_RETRO;
        const USER_CY = 214;
        const Q_BY = 188, Q_BH = 52;


        drawFilmStrip(ctx, STRIP_Y, scroll);
        drawFrames(ctx, STRIP_Y, [1, 1, 1, 1, 1]);


        ctx.save(); ctx.globalAlpha = 0.55;
        ctx.fillStyle = C.textXSoft; ctx.font = F(11, 600);
        ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
        ctx.fillText('t = 0', FRAME_X0 + FW / 2, STRIP_Y - 4);
        ctx.restore();


        const tTA = smoothstep(0.54, 0.64, t);
        if (tTA > 0.01) {
            ctx.save(); ctx.globalAlpha = tTA;
            ctx.fillStyle = C.textSoft; ctx.font = F(11, 600);
            ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
            ctx.fillText('(t = T)', FRAME_X_END - FW / 2, STRIP_Y - 4);

            ctx.strokeStyle = 'rgba(44,58,74,0.10)'; ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(RIGHT_X, STRIP_Y + BAND + FH + BAND + 6);
            ctx.lineTo(RIGHT_X, Q_BY);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.restore();
        }


        const modelA = smoothstep(0.60, 0.70, t);
        if (modelA > 0.01) {
            drawGlasses(ctx, RIGHT_X, 18, 1.1, C.modelCol, modelA);
        }


        const noA = smoothstep(0.78, 0.86, t);
        const noSc = easeOutBack(clamp(prog(t, 0.78, 0.84), 0, 1));
        if (noA > 0.01) {
            const bw = 64, bh = 28, bx = RIGHT_X - bw - 10, by = 4;
            const bcx = bx + bw / 2, bcy = by + bh / 2;
            ctx.save(); ctx.globalAlpha = noA;
            ctx.translate(bcx, bcy); ctx.scale(noSc, noSc); ctx.translate(-bcx, -bcy);
            ctx.fillStyle = 'rgba(44,58,74,0.88)';
            rrect(ctx, bx, by, bw, bh, 7); ctx.fill();

            ctx.beginPath();
            ctx.moveTo(bx + bw, bcy - 6); ctx.lineTo(bx + bw + 12, bcy); ctx.lineTo(bx + bw, bcy + 6);
            ctx.closePath(); ctx.fill();
            ctx.fillStyle = C.white; ctx.font = F(14, 700);
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText('No.', bcx, bcy);
            ctx.restore();
        }


        const userA = smoothstep(0.60, 0.70, t);
        if (userA > 0.01) {
            drawUser(ctx, RIGHT_X, USER_CY, 1.1, C.userCol, userA);
            ctx.save(); ctx.globalAlpha = userA;
            ctx.fillStyle = C.userCol; ctx.font = F(10, 600);
            ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            ctx.fillText('User', RIGHT_X, USER_CY + 20);
            ctx.restore();
        }


        const qA = smoothstep(0.68, 0.78, t);
        const qSc = easeOutBack(clamp(prog(t, 0.68, 0.76), 0, 1));
        if (qA > 0.01) {
            const bw = 196, bh = Q_BH, bx = RIGHT_X - bw - 12, by = Q_BY;
            const bcx = bx + bw / 2, bcy = by + bh / 2;
            ctx.save(); ctx.globalAlpha = qA;
            ctx.translate(bcx, bcy); ctx.scale(qSc, qSc); ctx.translate(-bcx, -bcy);
            ctx.fillStyle = 'rgba(74,108,179,0.88)';
            rrect(ctx, bx, by, bw, bh, 8); ctx.fill();

            ctx.beginPath();
            ctx.moveTo(bx + bw, bcy - 7); ctx.lineTo(bx + bw + 14, bcy); ctx.lineTo(bx + bw, bcy + 7);
            ctx.closePath(); ctx.fill();

            ctx.fillStyle = 'rgba(74,108,179,0.50)'; ctx.font = F(10, 600);
            ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
            ctx.fillText('Retrospective QA', bx + bw, by - 5);

            ctx.fillStyle = C.white; ctx.font = F(11, 600);
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            wrapText(ctx, 'Did I switch off the air-conditioner when I left the room?', bcx, bcy, bw - 18, 15);
            ctx.restore();
        }


        const capA = smoothstep(0.88, 0.96, t);
        if (capA > 0.01) {
            ctx.save(); ctx.globalAlpha = capA;
            ctx.fillStyle = C.retroAcct; ctx.font = F(12, 600);
            ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
            ctx.fillText('Query at t = T  →  immediate answer', W / 2, H - 10);
            ctx.restore();
        }
    }

    const PROACT_DUR = 15000;
    const CORRECT = 4;


    const FRAME_T = [0.07, 0.21, 0.35, 0.48, 0.62];

    function drawProactive(t, ctx, scroll) {
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);

        const STRIP_Y = STRIP_Y_PROACT;
        const GLASS_Y = STRIP_Y + BAND + FH + BAND + 32;
        const BUB_Y = GLASS_Y + 22;


        const userA = smoothstep(0, 0.06, t);
        if (userA > 0.01) {
            const UCX = 28, UCY = 28;
            drawUser(ctx, UCX, UCY, 1.0, C.userCol, userA);
            ctx.save(); ctx.globalAlpha = userA;
            ctx.fillStyle = C.userCol; ctx.font = F(10, 600);
            ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            ctx.fillText('User  (t = 0)', UCX, UCY + 18);
            ctx.restore();

            const bw = 226, bh = 48, bx = 56, by = 2;
            const bcx = bx + bw / 2, bcy = by + bh / 2;
            const sc = easeOutBack(clamp(prog(t, 0, 0.07), 0, 1));
            ctx.save(); ctx.globalAlpha = userA;
            ctx.translate(bcx, bcy); ctx.scale(sc, sc); ctx.translate(-bcx, -bcy);
            ctx.fillStyle = 'rgba(74,108,179,0.88)';
            rrect(ctx, bx, by, bw, bh, 8); ctx.fill();

            ctx.beginPath();
            ctx.moveTo(bx, bcy - 6); ctx.lineTo(bx - 12, bcy); ctx.lineTo(bx, bcy + 6);
            ctx.closePath(); ctx.fill();
            ctx.fillStyle = 'rgba(74,108,179,0.50)'; ctx.font = F(10, 600);
            ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText('Proactive QA', bx, by - 4);
            ctx.fillStyle = C.white; ctx.font = F(11, 600);
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            wrapText(ctx, 'Remind me to switch off the air-conditioner when I leave the room.', bcx, bcy, bw - 18, 15);
            ctx.restore();
        }


        const frameA = FRAME_T.map(ft => {
            if (t < ft) return 0;
            return smoothstep(ft, ft + 0.05, t);
        });


        drawFilmStrip(ctx, STRIP_Y, scroll);
        drawFrames(ctx, STRIP_Y, frameA);


        const TL = 56, TW = W - 56 - 56;
        const ARROW_Y = STRIP_Y + BAND + FH + BAND + 12;
        ctx.strokeStyle = C.track; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(TL, ARROW_Y); ctx.lineTo(TL + TW, ARROW_Y); ctx.stroke();
        ctx.fillStyle = C.track;
        ctx.beginPath();
        ctx.moveTo(TL + TW, ARROW_Y - 5); ctx.lineTo(TL + TW + 14, ARROW_Y); ctx.lineTo(TL + TW, ARROW_Y + 5);
        ctx.closePath(); ctx.fill();
        ctx.fillStyle = C.textXSoft; ctx.font = F(11, 500);
        ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
        ctx.fillText('streaming video', TL + TW + 18, ARROW_Y);


        for (let i = 0; i < N; i++) {
            if (frameA[i] < 0.01) continue;

            const cx = FRAME_X0 + i * (FW + FGAP) + FW / 2;
            const isCorrect = (i === CORRECT);
            const responded = isCorrect && t >= FRAME_T[CORRECT] + 0.04;

            drawGlasses(ctx, cx, GLASS_Y, 1.0,
                responded ? C.proactAcct : C.modelCol, frameA[i]);

            if (responded) {

                const rA = smoothstep(FRAME_T[CORRECT] + 0.04, FRAME_T[CORRECT] + 0.09, t);
                const rSc = easeOutBack(clamp(prog(t, FRAME_T[CORRECT] + 0.04, FRAME_T[CORRECT] + 0.10), 0, 1));
                const bw = 140, bh = 44, bx = cx - bw / 2, by = BUB_Y;
                const bcx = cx, bcy = by + bh / 2;
                ctx.save(); ctx.globalAlpha = rA;
                ctx.translate(bcx, bcy); ctx.scale(rSc, rSc); ctx.translate(-bcx, -bcy);
                ctx.fillStyle = 'rgba(58,140,120,0.90)';
                rrect(ctx, bx, by, bw, bh, 7); ctx.fill();
                ctx.fillStyle = C.white; ctx.font = F(10, 600);
                ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                wrapText(ctx, 'You forgot to switch the air-conditioner!', bcx, bcy, bw - 14, 14);
                ctx.restore();
            } else {

                const bw = 36, bh = 20, bx = cx - bw / 2, by = BUB_Y;
                ctx.save(); ctx.globalAlpha = frameA[i] * 0.85;
                ctx.fillStyle = 'rgba(44,58,74,0.72)';
                rrect(ctx, bx, by, bw, bh, 4); ctx.fill();
                ctx.fillStyle = C.white; ctx.font = F(11, 500);
                ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillText('...', cx, by + bh / 2);
                ctx.restore();
            }
        }


        const capA = smoothstep(0.84, 0.92, t);
        if (capA > 0.01) {
            ctx.save(); ctx.globalAlpha = capA;
            ctx.fillStyle = C.proactAcct; ctx.font = F(12, 600);
            ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
            ctx.fillText('Query at t = 0  →  stays silent (···) and responds at the correct moment', W / 2, H - 10);
            ctx.restore();
        }
    }

    let mode = 'retro';
    let animStart = null;
    let animFrame = null;

    window.switchQueryMode = function (newMode) {
        mode = newMode;
        animStart = null;
    };

    function initQueryCanvas() {
        const canvas = document.getElementById('spot-query-canvas');
        if (!canvas) return;

        mode = canvas.dataset.queryMode || 'retro';
        const ctx = canvas.getContext('2d');
        const dpr = Math.min(window.devicePixelRatio || 1, 2);

        canvas.width = W * dpr;
        canvas.height = H * dpr;
        ctx.scale(dpr, dpr);

        function tick(ts) {
            if (!animStart) animStart = ts;
            const dur = mode === 'retro' ? RETRO_DUR : PROACT_DUR;
            const t = ((ts - animStart) % dur) / dur;
            const scroll = (ts * SCROLL_PX_PER_MS) % HOLE_SPACING;

            if (mode === 'retro') drawRetro(t, ctx, scroll);
            else drawProactive(t, ctx, scroll);

            animFrame = requestAnimationFrame(tick);
        }

        window.addEventListener('beforeunload', () => {
            if (animFrame) cancelAnimationFrame(animFrame);
        });

        animFrame = requestAnimationFrame(tick);
    }

    document.addEventListener('DOMContentLoaded', initQueryCanvas);
})();