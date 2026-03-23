/**
 * Facial Proportion Analysis: Refactored with correct pixel-space math,
 * motion-stabilized auto-capture, profile head-pose enforcement, and camera controls.
 * MediaPipe returns normalized [0,1] coords; ALL distances/angles use absolute pixels.
 */

(function () {
  'use strict';

  // ---------- MediaPipe Face Mesh: exact node indices ----------
  const LANDMARKS = {
    // Eyes: Inner 133/362, Outer 33/263
    LEFT_EYE_INNER: 133,
    LEFT_EYE_OUTER: 33,
    RIGHT_EYE_INNER: 362,
    RIGHT_EYE_OUTER: 263,
    LEFT_EYE_TOP: 159,
    LEFT_EYE_BOTTOM: 145,
    RIGHT_EYE_TOP: 386,
    RIGHT_EYE_BOTTOM: 374,
    LEFT_IRIS_CENTER: 468,
    RIGHT_IRIS_CENTER: 473,

    // Nose: Tip 4, Base (Subnasale) 2
    NOSE_TIP: 4,
    SUBNASALE: 2,
    // Philtrum: Subnasale 2 → Upper lip top 0. Chin: Lower lip bottom 17 → Menton 152.
    UPPER_LIP_TOP: 0,
    LOWER_LIP_BOTTOM: 17,
    PHILTRUM_TOP: 164, // kept for FWHR / midface overlay
    NOSE_LEFT_ALAR: 129,
    NOSE_RIGHT_ALAR: 358,

    // Lips & chin (philtrum-to-chin uses UPPER_LIP_TOP 0, LOWER_LIP_BOTTOM 17, MENTON 152)
    LOWER_LIP: 14,
    MENTON: 152,

    // Brow
    GLABELLA: 10,

    // Bizygomatic (cheekbones): 234, 454
    LEFT_CHEEK: 234,
    RIGHT_CHEEK: 454,

    // Bigonial (jaw corners), frontal view: 132, 361
    LEFT_GONION: 132,
    RIGHT_GONION: 361,

    // Tragion (ear/jaw hinge): Left 127, Right 356
    LEFT_TRAGION: 127,
    RIGHT_TRAGION: 356,

    // Profile: Menton (chin) 152. Dynamic Gonion = sharpest corner on face oval (see JAWLINE_SILHOUETTE_*).
    // Strict outer jawline silhouette nodes (face oval), used for "smallest interior angle" gonion search.
    JAWLINE_SILHOUETTE_LEFT: [132, 58, 172, 136, 150, 149, 176, 148],
    JAWLINE_SILHOUETTE_RIGHT: [361, 288, 397, 365, 379, 378, 400, 377],

    // Legacy jawline contour (reference)
    JAW_CONTOUR_LEFT: [132, 135, 136, 138, 149, 148, 152],
    JAW_CONTOUR_RIGHT: [361, 364, 365, 366, 377, 378, 152],
  };

  // ---------- Page detection: camera vs image ----------
  const isImagePage = document.body.dataset?.page === 'image';

  // ---------- Pixel conversion (MUST do before any distance/angle) ----------
  const video = document.getElementById('video');
  const sourceImage = document.getElementById('sourceImage');
  const videoWidth = () => {
    if (isImagePage && sourceImage && sourceImage.complete && sourceImage.naturalWidth)
      return sourceImage.naturalWidth;
    if (video) return video.videoWidth || 0;
    return 0;
  };
  const videoHeight = () => {
    if (isImagePage && sourceImage && sourceImage.complete && sourceImage.naturalHeight)
      return sourceImage.naturalHeight;
    if (video) return video.videoHeight || 0;
    return 0;
  };

  function toPixel(normalized, width, height) {
    if (!normalized) return null;
    return {
      x: normalized.x * width,
      y: normalized.y * height,
      z: normalized.z,
    };
  }

  function getLandmark(landmarks, index, width, height) {
    const lm = landmarks[index];
    if (!lm) return null;
    return toPixel(lm, width, height);
  }

  // ---------- Math: all inputs must be absolute pixel coordinates ----------
  function euclidean2D(a, b) {
    if (!a || !b) return 0;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  /** Scalar distance from `origin` along the ray toward `end` at which `p` projects (2D, pixels). */
  function projDistAlongSegment(origin, end, p) {
    if (!origin || !end || !p) return NaN;
    const dx = end.x - origin.x;
    const dy = end.y - origin.y;
    const L = Math.hypot(dx, dy);
    if (L === 0) return NaN;
    return ((p.x - origin.x) * dx + (p.y - origin.y) * dy) / L;
  }

  /** Point at distance `distAlong` from `origin` toward `end` (same direction as segment, may extend past `end`). */
  function pointAtDistAlongSegment(origin, end, distAlong) {
    if (!origin || !end || !Number.isFinite(distAlong)) return null;
    const dx = end.x - origin.x;
    const dy = end.y - origin.y;
    const L = Math.hypot(dx, dy);
    if (L === 0) return null;
    const t = distAlong / L;
    return { x: origin.x + t * dx, y: origin.y + t * dy };
  }

  /**
   * Facial thirds along the Trichion→Menton axis: segment lengths are projections onto that line so
   * top + middle + lower equals facial height (when landmarks are ordered along the axis).
   */
  function facialThirdsProjectedOntoTM(trichion, glabella, subnasale, menton) {
    const faceHeight = euclidean2D(trichion, menton);
    if (!trichion || !glabella || !subnasale || !menton || faceHeight === 0) {
      return { faceHeight: faceHeight || 0, topLen: NaN, midLen: NaN, lowerLen: NaN };
    }
    const h = faceHeight;
    const tG = projDistAlongSegment(trichion, menton, glabella);
    const tSn = projDistAlongSegment(trichion, menton, subnasale);
    if (!Number.isFinite(tG) || !Number.isFinite(tSn)) {
      return { faceHeight: h, topLen: NaN, midLen: NaN, lowerLen: NaN };
    }
    const topLen = Math.max(0, tG);
    const midLen = Math.max(0, tSn - tG);
    const lowerLen = Math.max(0, h - tSn);
    return { faceHeight: h, topLen, midLen, lowerLen };
  }

  /**
   * Canthal tilt angle for one eye in degrees.
   * Definition: angle between (1) a horizontal line through the inner corner (medial canthus) and
   * (2) the line from inner corner to outer corner (lateral canthus). Canvas Y increases downward,
   * so "outer higher" means outer.y < inner.y. Returns positive when lateral corner is higher.
   */
  function canthalTiltEyeAngleDeg(innerPixel, outerPixel) {
    if (!innerPixel || !outerPixel) return 0;
    const dx = outerPixel.x - innerPixel.x;
    const dy = innerPixel.y - outerPixel.y; // canvas Y-down: outer higher => dy > 0
    if (dx === 0 && dy === 0) return 0;
    let angleDeg = Math.atan2(dy, dx) * (180 / Math.PI);
    // Normalize so both eyes report the same convention: magnitude = acute angle from horizontal, sign = outer higher
    if (angleDeg > 90) angleDeg = 180 - angleDeg;
    else if (angleDeg < -90) angleDeg = -180 - angleDeg;
    return angleDeg;
  }

  /**
   * Signed tilt of segment inner→outer vs facial horizon (intercanthal line medial L→medial R).
   * Same convention as canthalTiltEyeAngleDeg: lateral point higher than medial → positive.
   * Result in [-90°, 90°] so left/right sides are comparable (raw atan2 would show ~±180° on one side).
   */
  function segmentTiltVsIntercanthalHorizonDeg(segmentInner, segmentOuter, intercanthalLeft, intercanthalRight) {
    if (!segmentInner || !segmentOuter || !intercanthalLeft || !intercanthalRight) return 0;
    const toDeg = 180 / Math.PI;
    const horizonDx = intercanthalRight.x - intercanthalLeft.x;
    const horizonDy = intercanthalRight.y - intercanthalLeft.y;
    const headRollAngleRad = Math.atan2(horizonDy, horizonDx);
    const rawDeg = canthalTiltEyeAngleDeg(segmentInner, segmentOuter);
    const normalize90 = (deg) => {
      let d = deg;
      while (d > 90) d -= 180;
      while (d < -90) d += 180;
      return d;
    };
    const relativeRad = (rawDeg * Math.PI) / 180 - headRollAngleRad;
    return normalize90(relativeRad * toDeg);
  }

  /**
   * Head-roll-compensated canthal tilt. Measures tilt relative to the facial horizon (intercanthal line 133–362).
   * Returns { leftDeg, rightDeg } with "outer corner higher than inner" = POSITIVE for both eyes.
   */
  function canthalTiltWithHeadRollCompensation(leftInner, leftOuter, rightInner, rightOuter) {
    if (!leftInner || !leftOuter || !rightInner || !rightOuter) return { leftDeg: 0, rightDeg: 0 };
    return {
      leftDeg: segmentTiltVsIntercanthalHorizonDeg(leftInner, leftOuter, leftInner, rightInner),
      rightDeg: segmentTiltVsIntercanthalHorizonDeg(rightInner, rightOuter, leftInner, rightInner),
    };
  }

  /** Angle at vertex b (a–b–c) in degrees [0, 180]. All pixel coords. */
  function angleAtVertex(a, b, c) {
    if (!a || !b || !c) return 0;
    const ba = { x: a.x - b.x, y: a.y - b.y };
    const bc = { x: c.x - b.x, y: c.y - b.y };
    const dot = ba.x * bc.x + ba.y * bc.y;
    const magBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y);
    const magBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y);
    if (magBA === 0 || magBC === 0) return 0;
    const cos = Math.max(-1, Math.min(1, dot / (magBA * magBC)));
    return (Math.acos(cos) * 180) / Math.PI;
  }

  /**
   * Gonial angle at jaw corner (vertex B) using Law of Cosines.
   * Triangle A–B–C: a = B–C, b = A–B, c = A–C. Angle at B = acos((a² + b² − c²) / (2ab)).
   * All inputs in pixel space; returns degrees [0, 180].
   */
  function gonialAngleLawOfCosines(tragionA, gonionB, mentonC) {
    if (!tragionA || !gonionB || !mentonC) return 0;
    const a = euclidean2D(gonionB, mentonC);
    const b = euclidean2D(tragionA, gonionB);
    const c = euclidean2D(tragionA, mentonC);
    const denom = 2 * a * b;
    if (denom === 0) return 0;
    const cosB = (a * a + b * b - c * c) / denom;
    const clamped = Math.max(-1, Math.min(1, cosB));
    return Math.acos(clamped) * (180 / Math.PI);
  }

  /**
   * Extra Face Mesh geometry (468 landmarks): raw ratios & angles. Indices follow mesh spec
   * (e.g. glabella 9, trichion 10); separate from LANDMARKS used for overlays where noted.
   */
  const FACE_MESH_GEOM = {
    TRICHION: 10,
    GLABELLA: 9,
    NASION: 6,
    PRONASALE: 1,
    SUBNASALE: 2,
    UPPER_LIP_TOP: 0,
    UPPER_LIP_CENTER: 13,
    LOWER_LIP_CENTER: 14,
    LOWER_LIP_BOTTOM: 17,
    MENTON: 152,
    POGONION: 199,
    BIZYGOMATIC_L: 234,
    BIZYGOMATIC_R: 454,
    BIGONIAL_L: 132,
    BIGONIAL_R: 361,
    ALAR_L: 129,
    ALAR_R: 358,
    CHEILION_L: 61,
    CHEILION_R: 291,
    OUTER_CANTHUS_L: 33,
    OUTER_CANTHUS_R: 263,
    INNER_CANTHUS_L: 133,
    INNER_CANTHUS_R: 362,
    LEFT_EYE_TOP: 159,
    LEFT_EYE_BOTTOM: 145,
    RIGHT_EYE_TOP: 386,
    RIGHT_EYE_BOTTOM: 374,
    BITEMPORAL_L: 68,
    BITEMPORAL_R: 298,
    BROW_OUTER_L: 46,
    BROW_OUTER_R: 276,
    BROW_INNER_L: 55,
    BROW_INNER_R: 285,
  };

  function landmarkNormXYZ(landmarks, index) {
    const lm = landmarks[index];
    if (!lm || typeof lm.x !== 'number' || typeof lm.y !== 'number') return null;
    const z = typeof lm.z === 'number' && Number.isFinite(lm.z) ? lm.z : 0;
    return { x: lm.x, y: lm.y, z };
  }

  /** Interior angle at vertex p2 using normalized x,y,z (profile / 3D-consistent). */
  function angle3PointsNorm(landmarks, i1, i2, i3) {
    const p1 = landmarkNormXYZ(landmarks, i1);
    const p2 = landmarkNormXYZ(landmarks, i2);
    const p3 = landmarkNormXYZ(landmarks, i3);
    if (!p1 || !p2 || !p3) return NaN;
    const ax = p1.x - p2.x;
    const ay = p1.y - p2.y;
    const az = p1.z - p2.z;
    const bx = p3.x - p2.x;
    const by = p3.y - p2.y;
    const bz = p3.z - p2.z;
    const dot = ax * bx + ay * by + az * bz;
    const magA = Math.sqrt(ax * ax + ay * ay + az * az);
    const magB = Math.sqrt(bx * bx + by * by + bz * bz);
    if (magA === 0 || magB === 0) return NaN;
    const cos = Math.max(-1, Math.min(1, dot / (magA * magB)));
    return (Math.acos(cos) * 180) / Math.PI;
  }

  /** Raw atan2(dy,dx) in degrees (image x-right, y-down). Prefer canthalTiltEyeAngleDeg / segmentTiltVsIntercanthalHorizonDeg for comparable signed tilts across face sides. */
  function angleHorizontalDeg(a, b) {
    if (!a || !b) return NaN;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    if (dx === 0 && dy === 0) return NaN;
    return (Math.atan2(dy, dx) * 180) / Math.PI;
  }

  function safeRatioGeom(num, den) {
    if (!Number.isFinite(num) || !Number.isFinite(den) || den === 0) return NaN;
    return num / den;
  }

  function fmtRatioGeom(v) {
    return Number.isFinite(v) ? v.toFixed(2) : '—';
  }

  function fmtDegGeom(v) {
    return Number.isFinite(v) ? `${v.toFixed(1)}°` : '—';
  }

  /** Qualitative bands for mesh metrics (same idea as core dashboard labels). */
  function labelThirdRatio(r, segmentLabel) {
    if (!Number.isFinite(r)) return '';
    const t = 1 / 3;
    if (Math.abs(r - t) <= 0.055) return 'Balanced';
    return r > t ? `Longer ${segmentLabel}` : `Shorter ${segmentLabel}`;
  }
  function labelBizygOverFaceHeight(r) {
    if (!Number.isFinite(r)) return '';
    if (r < 0.7) return 'Narrower';
    if (r > 0.82) return 'Wider';
    return 'Medium';
  }
  function labelBitemporalOverBizyg(r) {
    if (!Number.isFinite(r)) return '';
    if (r < 0.82) return 'Narrow temples';
    if (r > 0.98) return 'Broad temples';
    return 'Average';
  }
  function labelMentonMidcheekOverHeight(r) {
    if (!Number.isFinite(r)) return '';
    if (r < 0.42) return 'Lower profile';
    if (r > 0.52) return 'Higher cheek mass';
    return 'Average';
  }
  function labelBrowSpanOverBizyg(r) {
    if (!Number.isFinite(r)) return '';
    if (r < 0.88) return 'Narrow brow';
    if (r > 1.02) return 'Broad brow';
    return 'Balanced';
  }
  function labelIntercanthalOverBizyg(r) {
    if (!Number.isFinite(r)) return '';
    if (r < 0.44) return 'Tighter spacing';
    if (r > 0.56) return 'Wider spacing';
    return 'Average';
  }
  function labelBrowTiltDeg(deg) {
    if (!Number.isFinite(deg)) return '';
    if (Math.abs(deg) < 4) return 'Neutral';
    return deg > 0 ? 'Ascending outer' : 'Descending outer';
  }
  function labelNoseBridgeOverAlar(r) {
    if (!Number.isFinite(r)) return '';
    if (r < 0.35) return 'Short bridge';
    if (r > 0.55) return 'Long bridge';
    return 'Balanced';
  }
  function labelLowerOverUpperLip(r) {
    if (!Number.isFinite(r)) return '';
    if (r < 1.1) return 'Upper-heavy';
    if (r > 1.45) return 'Lower-heavy';
    return 'Balanced';
  }
  function labelChinOverPhiltrum(r) {
    if (!Number.isFinite(r)) return '';
    if (r < 1.35) return 'Shorter chin segment';
    if (r > 2.2) return 'Longer chin segment';
    return 'Balanced';
  }
  function labelMouthOverAlar(r) {
    if (!Number.isFinite(r)) return '';
    if (r < 0.95) return 'Narrow mouth';
    if (r > 1.12) return 'Wide mouth';
    return 'Balanced';
  }
  function labelBigonialOverBizyg(r) {
    if (!Number.isFinite(r)) return '';
    if (r >= 0.95) return 'Square jaw';
    if (r <= 0.8) return 'Tapered';
    return 'Balanced';
  }
  function labelJawFrontalAngleDeg(deg) {
    if (!Number.isFinite(deg)) return '';
    if (deg < 145) return 'Tapered front';
    if (deg > 168) return 'Broad front';
    return 'Average';
  }
  function labelJawSlopeDeg(deg) {
    if (!Number.isFinite(deg)) return '';
    if (Math.abs(deg) < 10) return 'Neutral';
    return 'Sloped';
  }
  function labelNasofrontal3D(deg) {
    if (!Number.isFinite(deg)) return '';
    if (deg < 118) return 'Acute';
    if (deg > 148) return 'Open';
    return 'Average';
  }
  function labelConvexityGlabella3D(deg) {
    if (!Number.isFinite(deg)) return '';
    if (deg < 160) return 'Convex';
    if (deg > 172) return 'Flatter';
    return 'Neutral';
  }
  function labelTotalFacialConvexity3D(deg) {
    if (!Number.isFinite(deg)) return '';
    if (deg < 130) return 'Strong projection';
    if (deg > 155) return 'Straighter profile';
    return 'Average';
  }
  function labelNasolabial3D(deg) {
    if (!Number.isFinite(deg)) return '';
    if (deg < 105) return 'Acute';
    if (deg > 135) return 'Open';
    return 'Average';
  }
  function labelNasomental3D(deg) {
    if (!Number.isFinite(deg)) return '';
    if (deg < 30) return 'Acute';
    if (deg > 50) return 'Open';
    return 'Average';
  }
  function labelMentolabial3D(deg) {
    if (!Number.isFinite(deg)) return '';
    if (deg < 125) return 'Shallow sulcus';
    if (deg > 155) return 'Deep sulcus';
    return 'Average';
  }

  /**
   * Appends raw mesh-derived metrics. Frontal: planar ratios/angles only. Profile: adds normalized 3D angles only
   * (planar mesh metrics stay on the frontal capture so the side view dashboard is not duplicated).
   */
  function appendFaceMeshGeometryMetrics(metrics, landmarks, width, height, mode) {
    const M = FACE_MESH_GEOM;
    const p = (i) => getLandmark(landmarks, i, width, height);

    const pushR = (key, name, val, sub, label) => {
      const row = { key, name, value: fmtRatioGeom(val), sub };
      if (label) row.label = label;
      metrics.push(row);
    };
    const pushD = (key, name, val, sub, label) => {
      const row = { key, name, value: fmtDegGeom(val), sub };
      if (label) row.label = label;
      metrics.push(row);
    };

    if (mode === 'frontal') {
      const Tm = p(M.TRICHION);
      const Gla = p(M.GLABELLA);
      const Sn = p(M.SUBNASALE);
      const Me = p(M.MENTON);
      const thirds = facialThirdsProjectedOntoTM(Tm, Gla, Sn, Me);
      const faceHeight = thirds.faceHeight;
      const bizygW = euclidean2D(p(M.BIZYGOMATIC_L), p(M.BIZYGOMATIC_R));
      const leftCheek = p(M.BIZYGOMATIC_L);
      const rightCheek = p(M.BIZYGOMATIC_R);
      const midCheek =
        leftCheek && rightCheek ? { x: (leftCheek.x + rightCheek.x) / 2, y: (leftCheek.y + rightCheek.y) / 2 } : null;

      const icL = p(M.INNER_CANTHUS_L);
      const icR = p(M.INNER_CANTHUS_R);
      const intercanthal = euclidean2D(icL, icR);
      const alarW = euclidean2D(p(M.ALAR_L), p(M.ALAR_R));

      const rTop = safeRatioGeom(thirds.topLen, faceHeight);
      const rMid = safeRatioGeom(thirds.midLen, faceHeight);
      const rLow = safeRatioGeom(thirds.lowerLen, faceHeight);
      const rFw = safeRatioGeom(bizygW, faceHeight);
      const rBi = safeRatioGeom(euclidean2D(p(M.BITEMPORAL_L), p(M.BITEMPORAL_R)), bizygW);
      const rCh = midCheek && p(M.MENTON) ? safeRatioGeom(euclidean2D(p(M.MENTON), midCheek), faceHeight) : NaN;
      const rBr = safeRatioGeom(euclidean2D(p(M.BROW_OUTER_L), p(M.BROW_OUTER_R)), bizygW);
      const rIcBz = safeRatioGeom(intercanthal, bizygW);
      const browL = segmentTiltVsIntercanthalHorizonDeg(p(M.BROW_INNER_L), p(M.BROW_OUTER_L), icL, icR);
      const browR = segmentTiltVsIntercanthalHorizonDeg(p(M.BROW_INNER_R), p(M.BROW_OUTER_R), icL, icR);
      const rNb = safeRatioGeom(euclidean2D(p(M.NASION), p(M.PRONASALE)), alarW);
      const rLip = safeRatioGeom(
        euclidean2D(p(M.LOWER_LIP_CENTER), p(M.LOWER_LIP_BOTTOM)),
        euclidean2D(p(M.UPPER_LIP_TOP), p(M.UPPER_LIP_CENTER))
      );
      const rChinPh = safeRatioGeom(
        euclidean2D(p(M.LOWER_LIP_BOTTOM), p(M.MENTON)),
        euclidean2D(p(M.SUBNASALE), p(M.UPPER_LIP_TOP))
      );
      const rMouth = safeRatioGeom(euclidean2D(p(M.CHEILION_L), p(M.CHEILION_R)), alarW);
      const rJawW = safeRatioGeom(euclidean2D(p(M.BIGONIAL_L), p(M.BIGONIAL_R)), bizygW);
      const jl = p(M.BIGONIAL_L);
      const menton = p(M.MENTON);
      const jr = p(M.BIGONIAL_R);
      const jawFrontDeg = jl && menton && jr ? angleAtVertex(jl, menton, jr) : NaN;
      const jawSlopeL = angleHorizontalDeg(p(M.BIGONIAL_L), p(M.MENTON));

      pushR(
        'fmTopThird',
        'Top third ratio',
        rTop,
        'Upper segment along Trichion–Menton (Glabella projected) / Trichion–Menton',
        labelThirdRatio(rTop, 'upper third')
      );
      pushR(
        'fmMidThird',
        'Middle third ratio',
        rMid,
        'Middle segment along Trichion–Menton (Glabella→Subnasale projections) / Trichion–Menton',
        labelThirdRatio(rMid, 'middle third')
      );
      pushR(
        'fmLowerThird',
        'Lower third ratio',
        rLow,
        'Lower segment along Trichion–Menton (Subnasale projected) / Trichion–Menton',
        labelThirdRatio(rLow, 'lower third')
      );
      pushR('fmTotalFWHR', 'Bizygomatic / facial height', rFw, 'Left zygoma–Right zygoma / Trichion–Menton', labelBizygOverFaceHeight(rFw));
      pushR('fmBitemporalOverBizyg', 'Bitemporal / bizygomatic width', rBi, 'Left temporal–Right temporal / Left zygoma–Right zygoma', labelBitemporalOverBizyg(rBi));
      pushR(
        'fmCheekboneHeight',
        'Menton to mid-cheek / facial height',
        rCh,
        'Menton–Mid-cheek / Trichion–Menton',
        labelMentonMidcheekOverHeight(rCh)
      );

      pushR('fmBrowSpanOverBizyg', 'Brow span / bizygomatic width', rBr, 'Left brow outer–Right brow outer / Left zygoma–Right zygoma', labelBrowSpanOverBizyg(rBr));
      pushR('fmEyeSepOverBizyg', 'Intercanthal / bizygomatic width', rIcBz, 'Medial canthus L–Medial canthus R / Left zygoma–Right zygoma', labelIntercanthalOverBizyg(rIcBz));

      pushD(
        'fmBrowTiltL',
        'Eyebrow tilt left (°)',
        browL,
        'Medial brow–Lateral brow vs intercanthal horizon, left',
        labelBrowTiltDeg(browL)
      );
      pushD(
        'fmBrowTiltR',
        'Eyebrow tilt right (°)',
        browR,
        'Medial brow–Lateral brow vs intercanthal horizon, right',
        labelBrowTiltDeg(browR)
      );

      pushR('fmNoseBridgeOverAlar', 'Nose bridge / alar width', rNb, 'Nasion–Pronasale / Left ala–Right ala', labelNoseBridgeOverAlar(rNb));

      pushR(
        'fmLowerOverUpperLip',
        'Lower lip / upper lip height',
        rLip,
        'Lower lip center–Lower lip bottom / Upper lip top–Upper lip center',
        labelLowerOverUpperLip(rLip)
      );
      pushR(
        'fmChinOverPhiltrum',
        'Chin height / philtrum length',
        rChinPh,
        'Lower lip bottom–Menton / Subnasale–Upper lip top',
        labelChinOverPhiltrum(rChinPh)
      );
      pushR('fmMouthOverNoseWidth', 'Mouth width / alar width', rMouth, 'Left cheilion–Right cheilion / Left ala–Right ala', labelMouthOverAlar(rMouth));

      pushR('fmBigonialOverBizyg', 'Bigonial / bizygomatic width', rJawW, 'Left gonion–Right gonion / Left zygoma–Right zygoma', labelBigonialOverBizyg(rJawW));

      pushD('fmJawFrontalAngle', 'Jaw frontal angle', jawFrontDeg, 'Left gonion–Menton–Right gonion (angle at Menton)', labelJawFrontalAngleDeg(jawFrontDeg));
      pushD('fmJawSlopeL', 'Jaw slope left (°)', jawSlopeL, 'Left gonion–Menton', labelJawSlopeDeg(jawSlopeL));
    }

    if (mode === 'profile') {
      const aNf = angle3PointsNorm(landmarks, M.GLABELLA, M.NASION, M.PRONASALE);
      const aCg = angle3PointsNorm(landmarks, M.GLABELLA, M.SUBNASALE, M.POGONION);
      const aTc = angle3PointsNorm(landmarks, M.NASION, M.PRONASALE, M.POGONION);
      const aNl = angle3PointsNorm(landmarks, M.PRONASALE, M.SUBNASALE, M.UPPER_LIP_TOP);
      const aNm = angle3PointsNorm(landmarks, M.PRONASALE, M.NASION, M.POGONION);
      const aMl = angle3PointsNorm(landmarks, M.LOWER_LIP_CENTER, M.LOWER_LIP_BOTTOM, M.POGONION);
      pushD('fmNasofrontal', 'Nasofrontal angle', aNf, 'Glabella–Nasion–Pronasale', labelNasofrontal3D(aNf));
      pushD('fmConvexityGlabella', 'Facial convexity (mesh)', aCg, 'Glabella–Subnasale–Pogonion', labelConvexityGlabella3D(aCg));
      pushD('fmTotalFacialConvexity', 'Total facial convexity', aTc, 'Nasion–Pronasale–Pogonion', labelTotalFacialConvexity3D(aTc));
      pushD('fmNasolabial', 'Nasolabial angle', aNl, 'Pronasale–Subnasale–Upper lip top', labelNasolabial3D(aNl));
      pushD('fmNasomental', 'Nasomental angle', aNm, 'Pronasale–Nasion–Pogonion', labelNasomental3D(aNm));
      pushD('fmMentolabial', 'Mentolabial angle', aMl, 'Lower lip center–Lower lip bottom–Pogonion', labelMentolabial3D(aMl));
    }
  }

  // ---------- DOM & state ----------
  const overlay = document.getElementById('overlay');
  const hint = document.getElementById('hint');
  const hintSub = document.getElementById('hintSub');
  const metricsRoot = document.getElementById('metricsRoot');
  const modeSwitch = document.getElementById('modeSwitch');
  const videoWrapper = document.getElementById('videoWrapper');
  const videoContainer = document.getElementById('videoContainer');
  const overlayMessage = document.getElementById('overlayMessage');
  const retakeBtn = document.getElementById('retakeBtn');
  const clearBtn = document.getElementById('clearBtn');
  const mirrorBtn = document.getElementById('mirrorBtn');
  const fullscreenBtn = document.getElementById('fullscreenBtn');
  const frozenFrame = document.getElementById('frozenFrame');
  const fileInput = document.getElementById('fileInput');
  const uploadBtn = document.getElementById('uploadBtn');
  const imageDropZone = document.getElementById('imageDropZone');

  let faceMesh = null;
  let animationId = null;
  let isProfileMode = false;
  let isMirrored = false;

  // Per-view capture: frontal and side each have their own saved state and image. Neither view can see the other's.
  let capturedStateFrontal = null;
  let capturedStateProfile = null;
  let capturedImageFrontalURL = null;
  let capturedImageProfileURL = null;

  function getCurrentCapturedState() {
    return isProfileMode ? capturedStateProfile : capturedStateFrontal;
  }

  function getCurrentCapturedImageURL() {
    return isProfileMode ? capturedImageProfileURL : capturedImageFrontalURL;
  }

  /** On image page: URL of the image currently loaded for this mode (even if analysis failed). */
  function getCurrentImageSourceURL() {
    return isProfileMode ? imageSourceProfileURL : imageSourceFrontalURL;
  }

  const isCaptured = () => getCurrentCapturedState() !== null;

  // Paste/upload image: when set we show that image and analyze it (no camera loop)
  // On image page: separate URL per view (frontal vs side); side requires its own upload.
  let imageSourceURL = null;
  let imageSourceFrontalURL = null;
  let imageSourceProfileURL = null;
  let analyzingImage = false;
  let analyzingForProfileMode = false; // when true, analysis result goes to profile slot (image page only)
  let pendingImageAnalysisWidth = 0;
  let pendingImageAnalysisHeight = 0;

  // Motion stabilization: nose tip (node 4) pixel positions, last N frames
  const NOSE_BUFFER_SIZE = 20;
  const NOSE_BUFFER_SIZE_PROFILE = 10; // smaller = faster warm-up when turned
  const STABILITY_THRESHOLD_PX = 2;
  const STABILITY_THRESHOLD_PROFILE_PX = 8; // much looser for profile (head jitter when turned)
  const STABLE_FRAMES_FOR_CAPTURE = 30; // ~1 sec at 30fps
  const STABLE_FRAMES_FOR_CAPTURE_PROFILE = 12; // ~0.4 sec for profile so it actually triggers
  const MIN_GONIAL_NODE_SEPARATION_PX = 8; // require 3 distinct points: ear, gonion, chin
  const noseTipBuffer = [];
  let stableFrameCount = 0;

  // ---------- Single accent for every metric (canvas lines + dashboard card borders) ----------
  const METRIC_ACCENT = '#06B6D4';

  const DOT_RADIUS = 1.1;
  const LINE_WIDTH_BASE = 1;
  /** No card hovered: lines/dots barely visible (was 0.3; many stacked metrics read too loud). */
  const DEFAULT_OPACITY = 0.06;
  /** Hovered card: full contrast. */
  const HOVERED_OPACITY = 1.0;
  const HOVERED_LINE_WIDTH = 1.5;
  /** Hovered segment along Trichion–Menton (thirds); thicker than generic hovered metrics. */
  const FACIAL_THIRDS_HOVER_LINE_WIDTH = 4;
  /** Another card hovered: hide non-selected geometry almost entirely. */
  const NON_HOVERED_OPACITY = 0.03;

  let hoveredMetricKey = null;

  function clearOverlay() {
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  }

  function drawDot(ctx, p, hexColor, opacity, radius) {
    if (!p) return;
    const r = radius ?? DOT_RADIUS;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    const [rr, gg, bb] = hexToRgb(hexColor);
    ctx.fillStyle = `rgba(${rr},${gg},${bb},${opacity})`;
    ctx.fill();
  }

  function drawLine(ctx, p1, p2, hexColor, lineWidth, opacity) {
    if (!p1 || !p2) return;
    const [r, g, b] = hexToRgb(hexColor);
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.strokeStyle = `rgba(${r},${g},${b},${opacity})`;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
  }

  function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)] : [128, 128, 128];
  }

  function getOpacityAndLineWidth(metricKey) {
    const isHovered = hoveredMetricKey === metricKey;
    const isAnyHovered = hoveredMetricKey !== null;
    if (!isAnyHovered) {
      return { opacity: DEFAULT_OPACITY, lineWidth: LINE_WIDTH_BASE };
    }
    if (isHovered) {
      return { opacity: HOVERED_OPACITY, lineWidth: HOVERED_LINE_WIDTH };
    }
    return { opacity: NON_HOVERED_OPACITY, lineWidth: LINE_WIDTH_BASE };
  }

  const OVERLAY_REF_SIZE = 480;
  /** Cap so high-res frames do not inflate strokes/dots without bound. */
  const OVERLAY_SCALE_MAX = 2.25;
  function overlayScale(width, height) {
    const s = Math.min(width, height) / OVERLAY_REF_SIZE;
    return Math.max(1, Math.min(s, OVERLAY_SCALE_MAX));
  }

  /** Draw segments + nodes for one metric card (hover dims non-selected). */
  function drawMetricLineSet(ctx, metricKey, linePairs, width, height) {
    const C = METRIC_ACCENT;
    const scale = overlayScale(width, height);
    const lwPx = (w) => Math.max(1, Math.round(w * scale));
    const dr = () => Math.max(1.1, DOT_RADIUS * scale);
    const { opacity, lineWidth } = getOpacityAndLineWidth(metricKey);
    const pairs = linePairs.filter(([a, b]) => a && b);
    for (let i = 0; i < pairs.length; i++) {
      const a = pairs[i][0];
      const b = pairs[i][1];
      drawLine(ctx, a, b, C, lwPx(lineWidth), opacity);
    }
    for (let i = 0; i < pairs.length; i++) {
      const a = pairs[i][0];
      const b = pairs[i][1];
      if (a) drawDot(ctx, a, C, opacity, dr());
      if (b) drawDot(ctx, b, C, opacity, dr());
    }
  }

  /**
   * Vertical thirds along Trichion→Menton: T→PG and T→Me are collinear, so separate drawMetricLineSet calls
   * stack into one line. Draw once with perpendicular ticks at glabella/subnasale projections, connectors from
   * landmarks, larger dots, and a thick stroke on the hovered third’s segment.
   */
  function drawFacialThirdsUnifiedOverlay(ctx, T, Me, PG, PSn, Gla, Sn, width, height) {
    const C = METRIC_ACCENT;
    const scale = overlayScale(width, height);
    const lwPx = (w) => Math.max(1, Math.round(w * scale));
    const dotR = () => Math.max(1.75, 2.4 * scale);

    const structOp = Math.max(
      getOpacityAndLineWidth('fmTopThird').opacity,
      getOpacityAndLineWidth('fmMidThird').opacity,
      getOpacityAndLineWidth('fmLowerThird').opacity
    );

    if (!T || !Me || !PG || !PSn || !Gla || !Sn) {
      drawMetricLineSet(ctx, 'fmTopThird', [[T, Gla], [T, Me]], width, height);
      drawMetricLineSet(ctx, 'fmMidThird', [[Gla, Sn], [T, Me]], width, height);
      drawMetricLineSet(ctx, 'fmLowerThird', [[Sn, Me], [T, Me]], width, height);
      return;
    }

    const dx = Me.x - T.x;
    const dy = Me.y - T.y;
    const len = Math.hypot(dx, dy);
    const nx = len > 0 ? -dy / len : -1;
    const ny = len > 0 ? dx / len : 0;
    const tickHalf = Math.max(28, Math.min(width, height) * 0.068);

    drawLine(ctx, T, Me, C, lwPx(LINE_WIDTH_BASE), structOp);

    const tickEnds = (c) => [
      { x: c.x - nx * tickHalf, y: c.y - ny * tickHalf },
      { x: c.x + nx * tickHalf, y: c.y + ny * tickHalf },
    ];
    const [pgA, pgB] = tickEnds(PG);
    const [snA, snB] = tickEnds(PSn);
    drawLine(ctx, pgA, pgB, C, lwPx(LINE_WIDTH_BASE), structOp);
    drawLine(ctx, snA, snB, C, lwPx(LINE_WIDTH_BASE), structOp);

    drawLine(ctx, Gla, PG, C, lwPx(LINE_WIDTH_BASE), structOp);
    drawLine(ctx, Sn, PSn, C, lwPx(LINE_WIDTH_BASE), structOp);

    [T, PG, PSn, Me, Gla, Sn].forEach((pt) => drawDot(ctx, pt, C, structOp, dotR()));

    const segments = [
      { key: 'fmTopThird', a: T, b: PG },
      { key: 'fmMidThird', a: PG, b: PSn },
      { key: 'fmLowerThird', a: PSn, b: Me },
    ];
    for (let i = 0; i < segments.length; i++) {
      const { key, a, b } = segments[i];
      if (hoveredMetricKey === key) {
        drawLine(ctx, a, b, C, lwPx(FACIAL_THIRDS_HOVER_LINE_WIDTH), HOVERED_OPACITY);
      }
    }
  }

  /** Extra mesh metrics (fm*): same line+dot treatment as core metrics. */
  function drawFrontalFaceMeshMetricOverlays(ctx, landmarks, width, height) {
    const M = FACE_MESH_GEOM;
    const g = (i) => getLandmark(landmarks, i, width, height);
    const lc = g(M.BIZYGOMATIC_L);
    const rc = g(M.BIZYGOMATIC_R);
    const midCheek = lc && rc ? { x: (lc.x + rc.x) / 2, y: (lc.y + rc.y) / 2 } : null;

    const T = g(M.TRICHION), Gla = g(M.GLABELLA), Sn = g(M.SUBNASALE), Me = g(M.MENTON);
    const faceH = T && Me ? euclidean2D(T, Me) : 0;
    const tG = T && Me && Gla && faceH > 0 ? projDistAlongSegment(T, Me, Gla) : NaN;
    const tSn = T && Me && Sn && faceH > 0 ? projDistAlongSegment(T, Me, Sn) : NaN;
    const PG =
      Number.isFinite(tG) && faceH > 0 ? pointAtDistAlongSegment(T, Me, Math.max(0, Math.min(tG, faceH))) : null;
    const PSn =
      Number.isFinite(tSn) && faceH > 0 ? pointAtDistAlongSegment(T, Me, Math.max(0, Math.min(tSn, faceH))) : null;

    drawFacialThirdsUnifiedOverlay(ctx, T, Me, PG, PSn, Gla, Sn, width, height);
    drawMetricLineSet(ctx, 'fmTotalFWHR', [[g(M.BIZYGOMATIC_L), g(M.BIZYGOMATIC_R)], [g(M.TRICHION), g(M.MENTON)]], width, height);
    drawMetricLineSet(ctx, 'fmBitemporalOverBizyg', [[g(M.BITEMPORAL_L), g(M.BITEMPORAL_R)], [g(M.BIZYGOMATIC_L), g(M.BIZYGOMATIC_R)]], width, height);
    if (midCheek) {
      drawMetricLineSet(ctx, 'fmCheekboneHeight', [[g(M.MENTON), midCheek], [g(M.TRICHION), g(M.MENTON)]], width, height);
    }
    drawMetricLineSet(ctx, 'fmBrowSpanOverBizyg', [[g(M.BROW_OUTER_L), g(M.BROW_OUTER_R)], [g(M.BIZYGOMATIC_L), g(M.BIZYGOMATIC_R)]], width, height);
    drawMetricLineSet(ctx, 'fmEyeSepOverBizyg', [[g(M.INNER_CANTHUS_L), g(M.INNER_CANTHUS_R)], [g(M.BIZYGOMATIC_L), g(M.BIZYGOMATIC_R)]], width, height);
    drawMetricLineSet(ctx, 'fmBrowTiltL', [[g(M.BROW_INNER_L), g(M.BROW_OUTER_L)]], width, height);
    drawMetricLineSet(ctx, 'fmBrowTiltR', [[g(M.BROW_INNER_R), g(M.BROW_OUTER_R)]], width, height);
    drawMetricLineSet(ctx, 'fmNoseBridgeOverAlar', [[g(M.NASION), g(M.PRONASALE)], [g(M.ALAR_L), g(M.ALAR_R)]], width, height);
    drawMetricLineSet(ctx, 'fmLowerOverUpperLip', [[g(M.LOWER_LIP_CENTER), g(M.LOWER_LIP_BOTTOM)], [g(M.UPPER_LIP_TOP), g(M.UPPER_LIP_CENTER)]], width, height);
    drawMetricLineSet(ctx, 'fmChinOverPhiltrum', [[g(M.LOWER_LIP_BOTTOM), g(M.MENTON)], [g(M.SUBNASALE), g(M.UPPER_LIP_TOP)]], width, height);
    drawMetricLineSet(ctx, 'fmMouthOverNoseWidth', [[g(M.CHEILION_L), g(M.CHEILION_R)], [g(M.ALAR_L), g(M.ALAR_R)]], width, height);
    drawMetricLineSet(ctx, 'fmBigonialOverBizyg', [[g(M.BIGONIAL_L), g(M.BIGONIAL_R)], [g(M.BIZYGOMATIC_L), g(M.BIZYGOMATIC_R)]], width, height);
    drawMetricLineSet(ctx, 'fmJawFrontalAngle', [[g(M.BIGONIAL_L), g(M.MENTON)], [g(M.MENTON), g(M.BIGONIAL_R)], [g(M.BIGONIAL_L), g(M.BIGONIAL_R)]], width, height);
    drawMetricLineSet(ctx, 'fmJawSlopeL', [[g(M.BIGONIAL_L), g(M.MENTON)]], width, height);
  }

  function drawProfileFaceMeshMetricOverlays(ctx, landmarks, width, height) {
    const M = FACE_MESH_GEOM;
    const g = (i) => getLandmark(landmarks, i, width, height);
    drawMetricLineSet(ctx, 'fmNasofrontal', [[g(M.GLABELLA), g(M.NASION)], [g(M.NASION), g(M.PRONASALE)]], width, height);
    drawMetricLineSet(ctx, 'fmConvexityGlabella', [[g(M.GLABELLA), g(M.SUBNASALE)], [g(M.SUBNASALE), g(M.POGONION)]], width, height);
    drawMetricLineSet(ctx, 'fmTotalFacialConvexity', [[g(M.NASION), g(M.PRONASALE)], [g(M.PRONASALE), g(M.POGONION)]], width, height);
    drawMetricLineSet(ctx, 'fmNasolabial', [[g(M.PRONASALE), g(M.SUBNASALE)], [g(M.SUBNASALE), g(M.UPPER_LIP_TOP)]], width, height);
    drawMetricLineSet(ctx, 'fmNasomental', [[g(M.PRONASALE), g(M.NASION)], [g(M.NASION), g(M.POGONION)]], width, height);
    drawMetricLineSet(ctx, 'fmMentolabial', [[g(M.LOWER_LIP_CENTER), g(M.LOWER_LIP_BOTTOM)], [g(M.LOWER_LIP_BOTTOM), g(M.POGONION)]], width, height);
  }

  function drawFrontalOverlay(landmarks, width, height) {
    const ctx = overlay.getContext('2d');
    const g = (i) => getLandmark(landmarks, i, width, height);
    const scale = overlayScale(width, height);
    const lw = (w) => Math.max(1, Math.round(w * scale));
    const dr = () => Math.max(1.1, DOT_RADIUS * scale);

    const leftInner = g(LANDMARKS.LEFT_EYE_INNER), leftOuter = g(LANDMARKS.LEFT_EYE_OUTER);
    const rightInner = g(LANDMARKS.RIGHT_EYE_INNER), rightOuter = g(LANDMARKS.RIGHT_EYE_OUTER);
    let key = 'canthalTilt';
    let opacity, lineWidth, lineOpacity;
    ({ opacity, lineWidth } = getOpacityAndLineWidth(key));
    lineOpacity = opacity;
    // Canthal tilt: draw only the segment from inner to outer canthus (no extension past corners).
    drawLine(ctx, leftInner, leftOuter, METRIC_ACCENT, lw(lineWidth), lineOpacity);
    drawLine(ctx, rightInner, rightOuter, METRIC_ACCENT, lw(lineWidth), lineOpacity);
    drawDot(ctx, leftInner, METRIC_ACCENT, opacity, dr());
    drawDot(ctx, leftOuter, METRIC_ACCENT, opacity, dr());
    drawDot(ctx, rightInner, METRIC_ACCENT, opacity, dr());
    drawDot(ctx, rightOuter, METRIC_ACCENT, opacity, dr());

    const leftTop = g(LANDMARKS.LEFT_EYE_TOP), leftBottom = g(LANDMARKS.LEFT_EYE_BOTTOM);
    const rightTop = g(LANDMARKS.RIGHT_EYE_TOP), rightBottom = g(LANDMARKS.RIGHT_EYE_BOTTOM);
    key = 'palpebralFissure';
    ({ opacity, lineWidth } = getOpacityAndLineWidth(key));
    lineOpacity = opacity;
    drawLine(ctx, leftTop, leftBottom, METRIC_ACCENT, lw(lineWidth), lineOpacity);
    drawLine(ctx, rightTop, rightBottom, METRIC_ACCENT, lw(lineWidth), lineOpacity);
    drawDot(ctx, leftTop, METRIC_ACCENT, opacity, dr());
    drawDot(ctx, leftBottom, METRIC_ACCENT, opacity, dr());
    drawDot(ctx, rightTop, METRIC_ACCENT, opacity, dr());
    drawDot(ctx, rightBottom, METRIC_ACCENT, opacity, dr());

    key = 'intercanthalRatio';
    ({ opacity, lineWidth } = getOpacityAndLineWidth(key));
    lineOpacity = opacity;
    drawLine(ctx, leftInner, rightInner, METRIC_ACCENT, lw(lineWidth), lineOpacity);
    drawDot(ctx, leftInner, METRIC_ACCENT, opacity, dr());
    drawDot(ctx, rightInner, METRIC_ACCENT, opacity, dr());

    const leftCheek = g(LANDMARKS.LEFT_CHEEK), rightCheek = g(LANDMARKS.RIGHT_CHEEK);
    const glabella = g(LANDMARKS.GLABELLA), philtrumTop = g(LANDMARKS.PHILTRUM_TOP);
    key = 'fwhr';
    ({ opacity, lineWidth } = getOpacityAndLineWidth(key));
    lineOpacity = opacity;
    drawLine(ctx, leftCheek, rightCheek, METRIC_ACCENT, lw(lineWidth), lineOpacity);
    drawLine(ctx, glabella, philtrumTop, METRIC_ACCENT, lw(lineWidth), lineOpacity);
    drawDot(ctx, leftCheek, METRIC_ACCENT, opacity, dr());
    drawDot(ctx, rightCheek, METRIC_ACCENT, opacity, dr());
    drawDot(ctx, glabella, METRIC_ACCENT, opacity, dr());
    drawDot(ctx, philtrumTop, METRIC_ACCENT, opacity, dr());

    const leftIris = landmarks[LANDMARKS.LEFT_IRIS_CENTER];
    const rightIris = landmarks[LANDMARKS.RIGHT_IRIS_CENTER];
    const leftIrisP = leftIris ? toPixel(leftIris, width, height) : { x: (leftInner.x + leftOuter.x) / 2, y: (leftInner.y + leftOuter.y) / 2 };
    const rightIrisP = rightIris ? toPixel(rightIris, width, height) : { x: (rightInner.x + rightOuter.x) / 2, y: (rightInner.y + rightOuter.y) / 2 };
    key = 'midfaceRatio';
    ({ opacity, lineWidth } = getOpacityAndLineWidth(key));
    lineOpacity = opacity;
    drawDot(ctx, leftIrisP, METRIC_ACCENT, opacity, dr());
    drawDot(ctx, rightIrisP, METRIC_ACCENT, opacity, dr());
    const midY = (leftIrisP.y + rightIrisP.y) / 2;
    const midP = { x: (leftIrisP.x + rightIrisP.x) / 2, y: midY };
    drawLine(ctx, leftIrisP, rightIrisP, METRIC_ACCENT, lw(lineWidth), lineOpacity);
    drawLine(ctx, midP, philtrumTop, METRIC_ACCENT, lw(lineWidth), lineOpacity);
    drawDot(ctx, philtrumTop, METRIC_ACCENT, opacity, dr());

    const jawP1 = g(LANDMARKS.LEFT_GONION), jawP2 = g(LANDMARKS.RIGHT_GONION);
    const leftGonionDraw = jawP1 && jawP2 && jawP1.x <= jawP2.x ? jawP1 : jawP2;
    const rightGonionDraw = jawP1 && jawP2 && jawP1.x > jawP2.x ? jawP1 : jawP2;
    key = 'bigonialToBizygomatic';
    ({ opacity, lineWidth } = getOpacityAndLineWidth(key));
    lineOpacity = opacity;
    drawLine(ctx, leftGonionDraw, rightGonionDraw, METRIC_ACCENT, lw(lineWidth), lineOpacity);
    drawDot(ctx, leftGonionDraw, METRIC_ACCENT, opacity, dr());
    drawDot(ctx, rightGonionDraw, METRIC_ACCENT, opacity, dr());

    const subnasale = g(LANDMARKS.SUBNASALE), upperLipTop = g(LANDMARKS.UPPER_LIP_TOP);
    const lowerLipBottom = g(LANDMARKS.LOWER_LIP_BOTTOM), menton = g(LANDMARKS.MENTON);
    key = 'philtrumToChin';
    ({ opacity, lineWidth } = getOpacityAndLineWidth(key));
    lineOpacity = opacity;
    drawLine(ctx, subnasale, upperLipTop, METRIC_ACCENT, lw(lineWidth), lineOpacity);
    drawLine(ctx, lowerLipBottom, menton, METRIC_ACCENT, lw(lineWidth), lineOpacity);
    drawDot(ctx, subnasale, METRIC_ACCENT, opacity, dr());
    drawDot(ctx, upperLipTop, METRIC_ACCENT, opacity, dr());
    drawDot(ctx, lowerLipBottom, METRIC_ACCENT, opacity, dr());
    drawDot(ctx, menton, METRIC_ACCENT, opacity, dr());

    const noseLeft = g(LANDMARKS.NOSE_LEFT_ALAR), noseRight = g(LANDMARKS.NOSE_RIGHT_ALAR);
    key = 'alarBase';
    ({ opacity, lineWidth } = getOpacityAndLineWidth(key));
    lineOpacity = opacity;
    drawLine(ctx, noseLeft, noseRight, METRIC_ACCENT, lw(lineWidth), lineOpacity);
    drawLine(ctx, leftInner, rightInner, METRIC_ACCENT, lw(lineWidth), lineOpacity);
    drawDot(ctx, noseLeft, METRIC_ACCENT, opacity, dr());
    drawDot(ctx, noseRight, METRIC_ACCENT, opacity, dr());

    drawFrontalFaceMeshMetricOverlays(ctx, landmarks, width, height);
  }

  function drawProfileOverlay(landmarks, width, height, profileSide) {
    const ctx = overlay.getContext('2d');
    const g = (i) => getLandmark(landmarks, i, width, height);
    const scale = overlayScale(width, height);
    const lw = (w) => Math.max(1, Math.round(w * scale));
    const dr = () => Math.max(1.1, DOT_RADIUS * scale);
    const side = profileSide ?? getProfileSide(landmarks, width, height);

    // Gonial angle: only Tragion, Dynamic Gonion, Menton + two connecting lines
    const gonialPoints = getProfileGonialPoints(landmarks, width, height, side);
    if (gonialPoints) {
      const { tragion, gonion, menton } = gonialPoints;
      const key = 'gonialAngleRamus';
      const { opacity, lineWidth } = getOpacityAndLineWidth(key);
      const lineOpacity = opacity;
      drawLine(ctx, tragion, gonion, METRIC_ACCENT, lw(lineWidth), lineOpacity);
      drawLine(ctx, gonion, menton, METRIC_ACCENT, lw(lineWidth), lineOpacity);
      drawDot(ctx, tragion, METRIC_ACCENT, opacity, dr());
      drawDot(ctx, gonion, METRIC_ACCENT, opacity, dr());
      drawDot(ctx, menton, METRIC_ACCENT, opacity, dr());
    }

    const glabella = g(LANDMARKS.GLABELLA);
    const subnasale = g(LANDMARKS.SUBNASALE);
    const menton = g(LANDMARKS.MENTON);
    const key2 = 'facialConvexity';
    const { opacity: op2, lineWidth: lw2 } = getOpacityAndLineWidth(key2);
    const lineOpacity2 = op2;
    drawLine(ctx, glabella, subnasale, METRIC_ACCENT, lw(lw2), lineOpacity2);
    drawLine(ctx, subnasale, menton, METRIC_ACCENT, lw(lw2), lineOpacity2);
    drawDot(ctx, glabella, METRIC_ACCENT, op2, dr());
    drawDot(ctx, subnasale, METRIC_ACCENT, op2, dr());
    drawDot(ctx, menton, METRIC_ACCENT, op2, dr());

    drawProfileFaceMeshMetricOverlays(ctx, landmarks, width, height);
  }

  // ---------- Profile turn detection: nose-to-eye horizontal distance ratio ----------
  // Relaxed so image analyzer and camera accept more angles (e.g. 3/4 and half profile).
  // Ratio: when one eye is closer to nose than the other, min/max < threshold = valid profile.
  const PROFILE_EYE_RATIO_THRESHOLD = 0.6;

  /** Horizontal distance (pixels) from nose tip to left/right eye. */
  function getNoseToEyeDistances(landmarks, width, height) {
    const noseTip = getLandmark(landmarks, LANDMARKS.NOSE_TIP, width, height);
    const leftEye = getLandmark(landmarks, LANDMARKS.LEFT_EYE_OUTER, width, height);   // 33
    const rightEye = getLandmark(landmarks, LANDMARKS.RIGHT_EYE_OUTER, width, height); // 263
    if (!noseTip || !leftEye || !rightEye) return null;
    const distToLeft = Math.abs(noseTip.x - leftEye.x);
    const distToRight = Math.abs(noseTip.x - rightEye.x);
    return { distToLeft, distToRight };
  }

  /** True when face is validly turned for profile: one eye closer to nose than the other (ratio < threshold). */
  function isProfileTurnEnough(landmarks, width, height) {
    const d = getNoseToEyeDistances(landmarks, width, height);
    if (!d) return false;
    const minDist = Math.min(d.distToLeft, d.distToRight);
    const maxDist = Math.max(d.distToLeft, d.distToRight);
    if (maxDist === 0) return false;
    return minDist / maxDist < PROFILE_EYE_RATIO_THRESHOLD;
  }

  /** Which side of face is visible: nose tip position relative to eyes. Left side visible = 'left', right = 'right'. */
  /** Active profile side: compare Nose Tip (4) X to Ears (127 left, 356 right). Left profile = nose tip right of ear mid. */
  function getProfileSide(landmarks, width, height) {
    const noseTip = getLandmark(landmarks, LANDMARKS.NOSE_TIP, width, height);
    const leftEar = getLandmark(landmarks, LANDMARKS.LEFT_TRAGION, width, height);
    const rightEar = getLandmark(landmarks, LANDMARKS.RIGHT_TRAGION, width, height);
    if (!noseTip || !leftEar || !rightEar) return 'left';
    const earMidX = (leftEar.x + rightEar.x) / 2;
    return noseTip.x > earMidX ? 'left' : 'right';
  }

  /**
   * Profile jaw: Tragion (127/356), Dynamic Gonion (smallest interior angle on face oval), Menton (152).
   * Active side from Nose Tip vs Ears; then scan JAWLINE_SILHOUETTE_LEFT or _RIGHT for the node N that
   * minimizes the interior angle at N in triangle (Ear, N, Chin) via Law of Cosines.
   */
  function getProfileGonialPoints(landmarks, width, height, profileSide) {
    const g = (i) => getLandmark(landmarks, i, width, height);
    const side = profileSide ?? getProfileSide(landmarks, width, height);
    const useRight = side === 'right';
    const tragion = useRight ? g(LANDMARKS.RIGHT_TRAGION) : g(LANDMARKS.LEFT_TRAGION);
    const menton = g(LANDMARKS.MENTON);
    const jawlineIndices = useRight ? LANDMARKS.JAWLINE_SILHOUETTE_RIGHT : LANDMARKS.JAWLINE_SILHOUETTE_LEFT;
    if (!tragion || !menton) return null;

    let minAngleDeg = Infinity;
    let gonion = null;
    const c = euclidean2D(tragion, menton);
    for (let k = 0; k < jawlineIndices.length; k++) {
      const N = g(jawlineIndices[k]);
      if (!N) continue;
      const a = euclidean2D(tragion, N);
      const b = euclidean2D(N, menton);
      const denom = 2 * a * b;
      if (denom === 0) continue;
      const cosN = (a * a + b * b - c * c) / denom;
      const clamped = Math.max(-1, Math.min(1, cosN));
      const angleNDeg = Math.acos(clamped) * (180 / Math.PI);
      if (angleNDeg < minAngleDeg) {
        minAngleDeg = angleNDeg;
        gonion = N;
      }
    }
    if (!gonion) return null;
    // Require 3 distinct nodes: gonion must not collapse onto ear or chin
    const earToGonion = euclidean2D(tragion, gonion);
    const gonionToChin = euclidean2D(gonion, menton);
    if (earToGonion < MIN_GONIAL_NODE_SEPARATION_PX || gonionToChin < MIN_GONIAL_NODE_SEPARATION_PX)
      return null;
    return { tragion, gonion, menton };
  }

  function setOverlayMessage(text, show) {
    overlayMessage.textContent = text || '';
    overlayMessage.classList.toggle('visible', !!show);
  }

  // ---------- Motion stabilization & auto-capture ----------
  function updateNoseBuffer(nosePixel) {
    if (!nosePixel) return;
    noseTipBuffer.push({ x: nosePixel.x, y: nosePixel.y });
    if (noseTipBuffer.length > NOSE_BUFFER_SIZE) noseTipBuffer.shift();
  }

  function isNoseStable() {
    const bufSize = isProfileMode ? NOSE_BUFFER_SIZE_PROFILE : NOSE_BUFFER_SIZE;
    if (noseTipBuffer.length < bufSize) return false;
    const threshold = isProfileMode ? STABILITY_THRESHOLD_PROFILE_PX : STABILITY_THRESHOLD_PX;
    let sumX = 0, sumY = 0;
    for (let i = 0; i < noseTipBuffer.length; i++) {
      sumX += noseTipBuffer[i].x;
      sumY += noseTipBuffer[i].y;
    }
    const meanX = sumX / noseTipBuffer.length;
    const meanY = sumY / noseTipBuffer.length;
    let maxDist = 0;
    for (let i = 0; i < noseTipBuffer.length; i++) {
      const dx = noseTipBuffer[i].x - meanX;
      const dy = noseTipBuffer[i].y - meanY;
      maxDist = Math.max(maxDist, Math.sqrt(dx * dx + dy * dy));
    }
    return maxDist < threshold;
  }

  function captureVideoFrameToDataURL() {
    if (isImagePage && sourceImage && sourceImage.complete) {
      const w = sourceImage.naturalWidth;
      const h = sourceImage.naturalHeight;
      if (!w || !h) return imageSourceURL || null;
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(sourceImage, 0, 0);
      return canvas.toDataURL('image/png');
    }
    if (video) {
      const w = video.videoWidth;
      const h = video.videoHeight;
      if (!w || !h) return null;
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      return canvas.toDataURL('image/png');
    }
    return null;
  }

  function showFrozenFrame() {
    const url = getCurrentCapturedImageURL();
    if (frozenFrame && url) {
      frozenFrame.src = url;
      frozenFrame.classList.add('visible');
    }
    if (video) video.pause();
  }

  function hideFrozenFrame() {
    if (frozenFrame) {
      frozenFrame.src = '';
      frozenFrame.classList.remove('visible');
    }
    if (video) video.play();
  }

  function triggerAutoCapture(lockedMetrics, landmarks, width, height) {
    const profileSide = isProfileMode ? getProfileSide(landmarks, width, height) : null;
    const state = { landmarks, width, height, profileSide };
    const dataURL = captureVideoFrameToDataURL();

    if (isProfileMode) {
      capturedStateProfile = state;
      capturedImageProfileURL = dataURL;
    } else {
      capturedStateFrontal = state;
      capturedImageFrontalURL = dataURL;
    }

    (retakeBtn || clearBtn)?.classList.add('visible');
    videoContainer.classList.add('capture-flash');
    setTimeout(() => videoContainer.classList.remove('capture-flash'), 500);
    setOverlayMessage('', false);
    renderMetrics(lockedMetrics);
    showFrozenFrame();
  }

  function clearImagePage() {
    if (imageSourceFrontalURL && imageSourceFrontalURL.startsWith('blob:')) URL.revokeObjectURL(imageSourceFrontalURL);
    if (imageSourceProfileURL && imageSourceProfileURL.startsWith('blob:')) URL.revokeObjectURL(imageSourceProfileURL);
    if (imageSourceURL && imageSourceURL.startsWith('blob:')) URL.revokeObjectURL(imageSourceURL);
    imageSourceURL = null;
    imageSourceFrontalURL = null;
    imageSourceProfileURL = null;
    if (sourceImage) {
      sourceImage.src = '';
      sourceImage.style.display = 'none';
    }
    updateImagePageDropZoneVisibility();
    if (clearBtn) clearBtn.classList.remove('visible');
    capturedStateFrontal = null;
    capturedStateProfile = null;
    capturedImageFrontalURL = null;
    capturedImageProfileURL = null;
    stableFrameCount = 0;
    noseTipBuffer.length = 0;
    hoveredMetricKey = null;
    clearOverlay();
    renderMetrics([]);
    setImagePageHintForMode();
    if (hintSub) { hintSub.textContent = ''; hintSub.classList.add('empty'); }
    setMode(false);
  }

  function setImagePageHintForMode() {
    if (!isImagePage || !hint) return;
    if (isProfileMode) {
      hint.textContent = 'Upload a side profile image (paste or choose file).';
    } else {
      hint.textContent = 'Paste an image (Ctrl+V) or choose a file to analyze.';
    }
  }

  function updateImagePageDropZoneVisibility() {
    if (!isImagePage || !imageDropZone) return;
    const hasCurrent = getCurrentCapturedState() !== null;
    const hasImageForMode = getCurrentImageSourceURL() != null;
    const showDropZone = !hasImageForMode;
    imageDropZone.style.display = showDropZone ? 'flex' : 'none';
    if (sourceImage) {
      if (hasImageForMode) {
        const url = getCurrentImageSourceURL() || getCurrentCapturedImageURL() || '';
        sourceImage.src = url;
        sourceImage.style.display = url ? 'block' : 'none';
      } else {
        sourceImage.src = '';
        sourceImage.style.display = 'none';
      }
    }
    const dropText = imageDropZone.querySelector('.image-drop-text');
    if (dropText) {
      dropText.textContent = isProfileMode
        ? 'Paste (Ctrl+V) or choose a side profile image'
        : 'Paste (Ctrl+V) or choose an image below';
    }
    if (clearBtn) clearBtn.classList.toggle('visible', capturedStateFrontal !== null || capturedStateProfile !== null || hasImageForMode);
  }

  function retake() {
    if (isImagePage) {
      clearImagePage();
      return;
    }
    if (imageSourceURL) {
      if (imageSourceURL.startsWith('blob:')) URL.revokeObjectURL(imageSourceURL);
      imageSourceURL = null;
      if (videoWrapper) videoWrapper.classList.remove('image-source');
      if (video) video.style.display = '';
    }
    if (isProfileMode) {
      capturedStateProfile = null;
      capturedImageProfileURL = null;
    } else {
      capturedStateFrontal = null;
      capturedImageFrontalURL = null;
    }
    stableFrameCount = 0;
    noseTipBuffer.length = 0;
    hoveredMetricKey = null;
    (retakeBtn || clearBtn)?.classList.remove('visible');
    clearOverlay();
    hideFrozenFrame();
    if (!isImagePage) scheduleNextSend();
  }

  function analyzeImage(url) {
    if (!faceMesh || !url) return;
    imageSourceURL = url;
    if (isImagePage) {
      analyzingForProfileMode = isProfileMode;
      if (isProfileMode) {
        if (imageSourceProfileURL && imageSourceProfileURL.startsWith('blob:')) URL.revokeObjectURL(imageSourceProfileURL);
        imageSourceProfileURL = url;
      } else {
        if (imageSourceFrontalURL && imageSourceFrontalURL.startsWith('blob:')) URL.revokeObjectURL(imageSourceFrontalURL);
        imageSourceFrontalURL = url;
      }
    }
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const w = img.naturalWidth;
      const h = img.naturalHeight;
      if (!w || !h) {
        if (hint) hint.textContent = 'Could not read image dimensions.';
        return;
      }
      overlay.width = w;
      overlay.height = h;
      if (isImagePage) {
        if (sourceImage) {
          sourceImage.src = url;
          sourceImage.style.display = 'block';
        }
        if (imageDropZone) imageDropZone.style.display = 'none';
        if (clearBtn) clearBtn.classList.add('visible');
      } else {
        if (videoWrapper) videoWrapper.classList.add('image-source');
        if (video) video.style.display = 'none';
        if (frozenFrame) { frozenFrame.src = url; frozenFrame.classList.add('visible'); }
        if (video) video.pause();
        if (retakeBtn) retakeBtn.classList.add('visible');
      }
      clearOverlay();
      analyzingImage = true;
      pendingImageAnalysisWidth = w;
      pendingImageAnalysisHeight = h;
      if (hint) hint.textContent = 'Analyzing image…';
      if (hintSub) { hintSub.textContent = ''; hintSub.classList.add('empty'); }
      faceMesh.send({ image: img }).catch((e) => {
        console.warn('FaceMesh image error', e);
        analyzingImage = false;
        pendingImageAnalysisWidth = 0;
        pendingImageAnalysisHeight = 0;
        if (hint) hint.textContent = 'Analysis failed. Try another image.' + (isImagePage ? '' : ' Or use the camera.');
        if (isImagePage) updateImagePageDropZoneVisibility();
      });
    };
    img.onerror = () => {
      if (hint) hint.textContent = 'Failed to load image. Try another.';
      if (hintSub) hintSub.classList.add('empty');
      if (isImagePage) updateImagePageDropZoneVisibility();
    };
    img.src = url;
  }

  function applyCapturedView() {
    const capturedState = getCurrentCapturedState();
    if (!capturedState) return;
    const { landmarks, width, height, profileSide } = capturedState;
    clearOverlay();
    if (isProfileMode) {
      drawProfileOverlay(landmarks, width, height, profileSide ?? 'left');
      const metrics = computeProfileMetrics(landmarks, width, height, profileSide ?? 'left');
      renderMetrics(metrics);
    } else {
      drawFrontalOverlay(landmarks, width, height);
      const metrics = computeFrontalMetrics(landmarks, width, height);
      renderMetrics(metrics);
    }
  }

  function redrawCapturedOverlay() {
    const capturedState = getCurrentCapturedState();
    if (!capturedState) return;
    const { landmarks, width, height, profileSide } = capturedState;
    clearOverlay();
    if (isProfileMode) {
      drawProfileOverlay(landmarks, width, height, profileSide ?? 'left');
    } else {
      drawFrontalOverlay(landmarks, width, height);
    }
  }

  // ---------- Frontal metrics (all pixel-based) ----------
  function computeFrontalMetrics(landmarks, width, height) {
    const g = (i) => getLandmark(landmarks, i, width, height);
    const metrics = [];

    const leftInner = g(LANDMARKS.LEFT_EYE_INNER), leftOuter = g(LANDMARKS.LEFT_EYE_OUTER);
    const rightInner = g(LANDMARKS.RIGHT_EYE_INNER), rightOuter = g(LANDMARKS.RIGHT_EYE_OUTER);

    // Canthal tilt: angle between horizontal through medial canthus and line medial→lateral canthus.
    // Use head-roll compensation so "horizontal" is the intercanthal line (face-referenced).
    const canthalComp = canthalTiltWithHeadRollCompensation(leftInner, leftOuter, rightInner, rightOuter);
    const leftCanthalDeg = canthalComp.leftDeg;
    const rightCanthalDeg = canthalComp.rightDeg;
    const canthalAvg = (leftCanthalDeg + rightCanthalDeg) / 2;
    let canthalLabel = 'Neutral';
    if (canthalAvg > 2) canthalLabel = 'Positive';
    else if (canthalAvg < -2) canthalLabel = 'Negative';
    metrics.push({
      key: 'canthalTilt',
      name: 'Canthal Tilt',
      value: Math.round(canthalAvg) + '°',
      sub: `L ${Math.round(leftCanthalDeg)}° · R ${Math.round(rightCanthalDeg)}° · Lateral–medial canthus vs intercanthal horizon`,
      label: canthalLabel,
    });

    const leftTop = g(LANDMARKS.LEFT_EYE_TOP), leftBottom = g(LANDMARKS.LEFT_EYE_BOTTOM);
    const rightTop = g(LANDMARKS.RIGHT_EYE_TOP), rightBottom = g(LANDMARKS.RIGHT_EYE_BOTTOM);
    const leftWidth = euclidean2D(leftInner, leftOuter);
    const leftHeight = euclidean2D(leftTop, leftBottom);
    const rightWidth = euclidean2D(rightInner, rightOuter);
    const rightHeight = euclidean2D(rightTop, rightBottom);
    const leftPF = leftHeight > 0 ? leftWidth / leftHeight : 0;
    const rightPF = rightHeight > 0 ? rightWidth / rightHeight : 0;
    const pfRatio = (leftPF + rightPF) / 2;
    metrics.push({
      key: 'palpebralFissure',
      name: 'Palpebral Fissure Ratio',
      value: pfRatio.toFixed(2),
      sub: `L ${leftPF.toFixed(2)} · R ${rightPF.toFixed(2)} · Palpebral width (lateral–medial) / palpebral height`,
      label: pfRatio >= 3.5 ? 'Wide' : pfRatio <= 2.5 ? 'Narrow' : 'Medium',
    });

    const intercanthalDist = euclidean2D(leftInner, rightInner);
    const oneEyeWidth = (leftWidth + rightWidth) / 2;
    const intercanthalRatio = oneEyeWidth > 0 ? intercanthalDist / oneEyeWidth : 0;
    metrics.push({
      key: 'intercanthalRatio',
      name: 'Intercanthal Ratio',
      value: intercanthalRatio.toFixed(2),
      sub: 'Medial canthus–Medial canthus / Mean palpebral width (lateral–medial)',
      label: intercanthalRatio >= 1.1 ? 'Wide-set' : intercanthalRatio <= 0.9 ? 'Close-set' : 'Average',
    });

    const leftCheek = g(LANDMARKS.LEFT_CHEEK), rightCheek = g(LANDMARKS.RIGHT_CHEEK);
    const glabella = g(LANDMARKS.GLABELLA), philtrumTop = g(LANDMARKS.PHILTRUM_TOP);
    const bizygomaticWidth = euclidean2D(leftCheek, rightCheek);
    const upperFaceHeight = euclidean2D(glabella, philtrumTop);
    const fWHR = upperFaceHeight > 0 ? bizygomaticWidth / upperFaceHeight : 0;
    metrics.push({
      key: 'fwhr',
      name: 'Facial Width–Height Ratio (fWHR)',
      value: fWHR.toFixed(2),
      sub: 'Left zygoma–Right zygoma / Glabella–Upper lip superior (midline)',
      label: fWHR >= 1.35 ? 'Wider' : fWHR <= 1.15 ? 'Narrower' : 'Medium',
    });

    const leftIris = landmarks[LANDMARKS.LEFT_IRIS_CENTER];
    const rightIris = landmarks[LANDMARKS.RIGHT_IRIS_CENTER];
    let midfaceRatio = 0;
    if (leftIris && rightIris) {
      const leftP = toPixel(leftIris, width, height);
      const rightP = toPixel(rightIris, width, height);
      const ipd = euclidean2D(leftP, rightP);
      const midY = (leftP.y + rightP.y) / 2;
      const pupilToLip = Math.abs(midY - philtrumTop.y);
      midfaceRatio = pupilToLip > 0 ? ipd / pupilToLip : 0;
    } else {
      const leftMid = { x: (leftInner.x + leftOuter.x) / 2, y: (leftInner.y + leftOuter.y) / 2 };
      const rightMid = { x: (rightInner.x + rightOuter.x) / 2, y: (rightInner.y + rightOuter.y) / 2 };
      const ipd = euclidean2D(leftMid, rightMid);
      const midY = (leftMid.y + rightMid.y) / 2;
      const pupilToLip = Math.abs(midY - philtrumTop.y);
      midfaceRatio = pupilToLip > 0 ? ipd / pupilToLip : 0;
    }
    metrics.push({
      key: 'midfaceRatio',
      name: 'Midface Ratio',
      value: midfaceRatio.toFixed(2),
      sub: 'Interpupillary distance / Mid-pupil line to upper lip superior',
      label: midfaceRatio >= 1.0 ? 'Longer midface' : midfaceRatio <= 0.75 ? 'Shorter midface' : 'Average',
    });

    // Bigonial: use min/max x so we always get true jaw width (indices can vary by model)
    const jawP1 = g(LANDMARKS.LEFT_GONION), jawP2 = g(LANDMARKS.RIGHT_GONION);
    const leftGonion = jawP1 && jawP2 && jawP1.x <= jawP2.x ? jawP1 : jawP2;
    const rightGonion = jawP1 && jawP2 && jawP1.x > jawP2.x ? jawP1 : jawP2;
    const bigonialWidth = euclidean2D(leftGonion, rightGonion);
    const bigonialToBizy = bizygomaticWidth > 0 ? bigonialWidth / bizygomaticWidth : 0;
    metrics.push({
      key: 'bigonialToBizygomatic',
      name: 'Bigonial to Bizygomatic Ratio',
      value: bigonialToBizy.toFixed(2),
      sub: 'Left gonion–Right gonion / Left zygoma–Right zygoma',
      label: bigonialToBizy >= 0.95 ? 'Square jaw' : bigonialToBizy <= 0.8 ? 'Tapered' : 'Balanced',
    });

    const subnasale = g(LANDMARKS.SUBNASALE), upperLipTop = g(LANDMARKS.UPPER_LIP_TOP);
    const lowerLipBottom = g(LANDMARKS.LOWER_LIP_BOTTOM), menton = g(LANDMARKS.MENTON);
    // Philtrum = Subnasale (2) → Upper lip top (0). Chin = Lower lip bottom (17) → Menton (152).
    const philtrumLen = euclidean2D(subnasale, upperLipTop);
    const chinLen = euclidean2D(lowerLipBottom, menton);
    const philtrumChinRatio = chinLen > 0 ? philtrumLen / chinLen : 0;
    let philtrumLabel = 'Ideal/Balanced';
    if (philtrumChinRatio > 0.55) philtrumLabel = 'Long Philtrum';
    else if (philtrumChinRatio < 0.35) philtrumLabel = 'Short Philtrum';
    metrics.push({
      key: 'philtrumToChin',
      name: 'Philtrum-to-Chin Ratio',
      value: philtrumChinRatio.toFixed(2),
      sub: 'Subnasale–Upper lip top / Lower lip bottom–Menton',
      label: philtrumLabel,
    });

    const noseLeft = g(LANDMARKS.NOSE_LEFT_ALAR), noseRight = g(LANDMARKS.NOSE_RIGHT_ALAR);
    const alarWidth = euclidean2D(noseLeft, noseRight);
    const alarIntercanthalRatio = intercanthalDist > 0 ? alarWidth / intercanthalDist : 0;
    metrics.push({
      key: 'alarBase',
      name: 'Alar-Base to Intercanthal Ratio',
      value: alarIntercanthalRatio.toFixed(2),
      sub: 'Left ala–Right ala / Medial canthus–Medial canthus',
      label: alarIntercanthalRatio >= 1.05 ? 'Wider nose' : alarIntercanthalRatio <= 0.85 ? 'Narrower nose' : 'Proportional',
    });

    appendFaceMeshGeometryMetrics(metrics, landmarks, width, height, 'frontal');

    return metrics;
  }

  // ---------- Profile metrics (only when head is turned). profileSide: 'left' | 'right' = which face side is visible. ----------
  function computeProfileMetrics(landmarks, width, height, profileSide) {
    const g = (i) => getLandmark(landmarks, i, width, height);
    const metrics = [];
    const side = profileSide ?? getProfileSide(landmarks, width, height);
    const gonialPoints = getProfileGonialPoints(landmarks, width, height, side);

    if (gonialPoints) {
      const { tragion, gonion, menton } = gonialPoints;
      const gonialAngle = gonialAngleLawOfCosines(tragion, gonion, menton);
      const ramusLength = euclidean2D(tragion, gonion);
      metrics.push({
        key: 'gonialAngleRamus',
        name: 'Gonial Angle',
        value: gonialAngle.toFixed(1) + '°',
        sub: `Ramus: ${ramusLength.toFixed(0)} px · Ear–Dynamic Gonion–Chin (oval sharpest corner)`,
        label: gonialAngle >= 130 ? 'Obtuse' : gonialAngle <= 110 ? 'Acute' : 'Average',
      });
    }

    const glabella = g(LANDMARKS.GLABELLA);
    const subnasale = g(LANDMARKS.SUBNASALE);
    const menton = g(LANDMARKS.MENTON);
    const convexityAngle = angleAtVertex(glabella, subnasale, menton);
    metrics.push({
      key: 'facialConvexity',
      name: 'Facial Convexity',
      value: convexityAngle.toFixed(1) + '°',
      sub: 'Glabella–Subnasale–Menton (angle at Subnasale)',
      label: convexityAngle >= 175 ? 'Flat' : convexityAngle <= 165 ? 'Convex' : 'Neutral',
    });

    appendFaceMeshGeometryMetrics(metrics, landmarks, width, height, 'profile');

    return metrics;
  }

  // ---------- Dashboard ----------
  function renderMetrics(metrics) {
    if (!metrics.length) {
      const msg = isProfileMode
        ? (isImagePage ? 'No face detected. Use a 3/4 profile image.' : 'No face detected. Face the camera first, then slowly turn your head to the side (~50–70°).')
        : (isImagePage ? 'No face detected.' : 'No face detected. Look at the camera.');
      metricsRoot.innerHTML = `<p class="metric-label">${msg}</p>`;
      return;
    }
    metricsRoot.innerHTML = metrics
      .map(
        (m) => {
          const key = m.key || 'alarBase';
          const borderColor = METRIC_ACCENT;
          return `
        <div class="metric-card" data-metric="${key}" style="--metric-color: ${borderColor}">
          <h3>${m.name}</h3>
          <div class="metric-value">${m.value}</div>
          ${m.sub ? `<div class="metric-label">${m.sub}</div>` : ''}
          ${m.label ? `<div class="metric-label">${m.label}</div>` : ''}
        </div>
      `;
        }
      )
      .join('');
  }

  function setupMetricCardHover() {
    metricsRoot.addEventListener('mouseenter', (e) => {
      const card = e.target.closest('.metric-card');
      if (!card) return;
      const key = card.getAttribute('data-metric');
      if (key) {
        hoveredMetricKey = key;
        if (isCaptured()) redrawCapturedOverlay();
      }
    }, true);
    metricsRoot.addEventListener('mouseleave', (e) => {
      if (e.relatedTarget && metricsRoot.contains(e.relatedTarget)) return;
      hoveredMetricKey = null;
      if (isCaptured()) redrawCapturedOverlay();
    }, true);
  }

  // ---------- Pipeline ----------
  function onResults(results, width, height) {
    if (analyzingImage) {
      analyzingImage = false;
      pendingImageAnalysisWidth = 0;
      pendingImageAnalysisHeight = 0;
      clearOverlay();
      if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
        hint.textContent = isImagePage && analyzingForProfileMode
          ? 'No face detected. Try a clearer side image or a slightly less extreme angle.'
          : 'No face detected in image. Try another or use the camera.';
        if (hintSub) { hintSub.textContent = ''; hintSub.classList.add('empty'); }
        renderMetrics([]);
        if (isImagePage) updateImagePageDropZoneVisibility();
        return;
      }
      const landmarks = results.multiFaceLandmarks[0];
      const turnEnough = isProfileTurnEnough(landmarks, width, height);
      const profileSide = getProfileSide(landmarks, width, height);
      const state = { landmarks, width, height, profileSide };

      if (isImagePage) {
        if (analyzingForProfileMode) {
          capturedStateProfile = state;
          capturedImageProfileURL = imageSourceProfileURL || imageSourceURL;
          drawProfileOverlay(landmarks, width, height, profileSide);
          const metrics = computeProfileMetrics(landmarks, width, height, profileSide);
          renderMetrics(metrics);
        } else {
          capturedStateFrontal = state;
          capturedImageFrontalURL = imageSourceFrontalURL || imageSourceURL;
          drawFrontalOverlay(landmarks, width, height);
          const metrics = computeFrontalMetrics(landmarks, width, height);
          renderMetrics(metrics);
        }
        updateImagePageDropZoneVisibility();
        setImagePageHintForMode();
        if (hintSub) hintSub.classList.add('empty');
        return;
      }

      if (turnEnough && isProfileMode) {
        capturedStateProfile = state;
        capturedImageProfileURL = imageSourceURL;
        drawProfileOverlay(landmarks, width, height, profileSide ?? 'left');
        const metrics = computeProfileMetrics(landmarks, width, height, profileSide ?? 'left');
        renderMetrics(metrics);
      } else {
        capturedStateFrontal = state;
        capturedImageFrontalURL = imageSourceURL;
        if (!turnEnough && isProfileMode) {
          capturedStateProfile = null;
          capturedImageProfileURL = null;
        }
        drawFrontalOverlay(landmarks, width, height);
        const metrics = computeFrontalMetrics(landmarks, width, height);
        renderMetrics(metrics);
      }
      hint.textContent = 'Hold still for ~1 second to auto-capture.';
      if (hintSub) {
        hintSub.textContent = 'Paste (Ctrl+V) or upload another image. Use mirror / fullscreen below if needed.';
        hintSub.classList.remove('empty');
      }
      return;
    }

    if (isCaptured()) return;

    clearOverlay();
    const hasFace = results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0;

    if (!hasFace) {
      setOverlayMessage('', false);
      if (!isImagePage && isProfileMode && hintSub) {
        hint.textContent = 'Hold still for ~1 second to auto-capture.';
        hintSub.textContent = 'Turn head to the side (~60–75°). Start facing the camera, then turn slowly.';
        hintSub.classList.remove('empty');
      }
      renderMetrics([]);
      stableFrameCount = 0;
      scheduleNextSend();
      return;
    }

    const landmarks = results.multiFaceLandmarks[0];
    const noseTipPixel = getLandmark(landmarks, LANDMARKS.NOSE_TIP, width, height);

    if (isProfileMode) {
      const turnEnough = isProfileTurnEnough(landmarks, width, height);
      const profileSide = getProfileSide(landmarks, width, height);

      if (!isImagePage) {
        hint.textContent = 'Hold still for ~1 second to auto-capture.';
        if (hintSub) hintSub.classList.remove('empty');
      }
      setOverlayMessage('', false);

      drawProfileOverlay(landmarks, width, height, profileSide);
      const profileMetrics = computeProfileMetrics(landmarks, width, height, profileSide);
      renderMetrics(profileMetrics);

      if (turnEnough) {
        updateNoseBuffer(noseTipPixel);
        if (isNoseStable()) {
          stableFrameCount++;
          if (stableFrameCount >= STABLE_FRAMES_FOR_CAPTURE_PROFILE) {
            triggerAutoCapture(profileMetrics, landmarks, width, height);
            return;
          }
        } else {
          stableFrameCount = 0;
        }
      } else {
        stableFrameCount = 0;
      }
      scheduleNextSend();
      return;
    }

    setOverlayMessage('', false);
    drawFrontalOverlay(landmarks, width, height);
    const frontalMetrics = computeFrontalMetrics(landmarks, width, height);
    renderMetrics(frontalMetrics);

    updateNoseBuffer(noseTipPixel);
    if (isNoseStable()) {
      stableFrameCount++;
      if (stableFrameCount >= STABLE_FRAMES_FOR_CAPTURE) {
        triggerAutoCapture(frontalMetrics, landmarks, width, height);
        return;
      }
    } else {
      stableFrameCount = 0;
    }

    scheduleNextSend();
  }

  function resizeOverlay() {
    const vw = videoWidth();
    const vh = videoHeight();
    if (!vw || !vh) return;
    overlay.width = vw;
    overlay.height = vh;
  }

  function scheduleNextSend() {
    if (isImagePage || isCaptured()) return;
    if (animationId) cancelAnimationFrame(animationId);
    animationId = requestAnimationFrame(async () => {
      if (!videoWidth() || !faceMesh || isCaptured()) return;
      resizeOverlay();
      try {
        await faceMesh.send({ image: video });
      } catch (e) {
        console.warn('FaceMesh send error', e);
        scheduleNextSend();
      }
    });
  }

  // ---------- MediaPipe init ----------
  async function initFaceMesh() {
    faceMesh = new FaceMesh({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
    });
    faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.3,
      minTrackingConfidence: 0.3,
    });
    faceMesh.onResults((results) => {
      const w = pendingImageAnalysisWidth || overlay.width || videoWidth();
      const h = pendingImageAnalysisHeight || overlay.height || videoHeight();
      onResults(results, w, h);
    });
  }

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
      });
      video.srcObject = stream;
      hint.textContent = 'Hold still for ~1 second to auto-capture.';
      if (hintSub) {
        hintSub.textContent = 'Use mirror / fullscreen below if necessary.';
        hintSub.classList.remove('empty');
      }
    } catch (e) {
      hint.textContent = 'Camera access denied. Please allow camera and refresh.';
      if (hintSub) {
        hintSub.textContent = '';
        hintSub.classList.add('empty');
      }
      console.error(e);
    }
  }

  // ---------- Mode toggle ----------
  function setMode(profile) {
    isProfileMode = profile;
    modeSwitch.checked = profile;
    document.querySelector('.mode-label[data-mode="frontal"]').classList.toggle('active', !profile);
    document.querySelector('.mode-label[data-mode="side"]').classList.toggle('active', profile);
    setOverlayMessage('', false);
    if (!isImagePage) {
      hint.textContent = 'Hold still for ~1 second to auto-capture.';
      if (hintSub) {
        if (profile) {
          hintSub.textContent = 'Turn head to the side (~50–70°). Start facing the camera, then turn slowly; full 90° often loses detection.';
          hintSub.classList.remove('empty');
        } else {
          hintSub.textContent = 'Use mirror / fullscreen below if necessary.';
          hintSub.classList.remove('empty');
        }
      }
    }

    if (isImagePage) {
      setImagePageHintForMode();
      updateImagePageDropZoneVisibility();
      const state = getCurrentCapturedState();
      if (state) {
        if (sourceImage) {
          sourceImage.src = getCurrentCapturedImageURL() || '';
          sourceImage.style.display = 'block';
        }
        if (clearBtn) clearBtn.classList.add('visible');
        overlay.width = state.width;
        overlay.height = state.height;
        applyCapturedView();
      } else {
        if (sourceImage) { sourceImage.src = ''; sourceImage.style.display = 'none'; }
        if (clearBtn) clearBtn.classList.remove('visible');
        clearOverlay();
        renderMetrics([]);
      }
      return;
    }

    if (imageSourceURL) {
      analyzeImage(imageSourceURL);
      return;
    }
    const state = getCurrentCapturedState();
    if (state) {
      showFrozenFrame();
      retakeBtn.classList.add('visible');
      applyCapturedView();
    } else {
      hideFrozenFrame();
      retakeBtn.classList.remove('visible');
      stableFrameCount = 0;
      noseTipBuffer.length = 0;
      clearOverlay();
      renderMetrics([]);
      scheduleNextSend();
    }
  }

  modeSwitch.addEventListener('change', () => setMode(modeSwitch.checked));

  // ---------- Camera controls ----------
  // Mirror: CSS scaleX(-1) on the wrapper flips display only. We do NOT transform landmark
  // coordinates: MediaPipe and canvas use the same video pixel space, so distances/angles stay correct.
  if (mirrorBtn && videoWrapper) {
    mirrorBtn.addEventListener('click', () => {
      isMirrored = !isMirrored;
      videoWrapper.classList.toggle('mirror', isMirrored);
    });
  }

  const getFullscreenElement = () =>
    document.fullscreenElement ?? document.webkitFullscreenElement ?? null;
  const requestFs = (el) =>
    (el && (el.requestFullscreen?.() ?? el.webkitRequestFullscreen?.())) || null;
  const exitFs = () =>
    (document.exitFullscreen?.() ?? document.webkitExitFullscreen?.()) || null;

  if (fullscreenBtn && videoWrapper) {
    fullscreenBtn.addEventListener('click', () => {
      if (!getFullscreenElement()) {
        requestFs(videoWrapper);
      } else {
        exitFs();
      }
    });
    document.addEventListener('fullscreenchange', updateFullscreenButton);
    document.addEventListener('webkitfullscreenchange', updateFullscreenButton);
  }
  function updateFullscreenButton() {
    if (fullscreenBtn) {
      fullscreenBtn.textContent = getFullscreenElement() ? 'Exit Fullscreen' : 'Fullscreen';
    }
  }

  if (retakeBtn) retakeBtn.addEventListener('click', retake);
  if (clearBtn) clearBtn.addEventListener('click', () => { clearImagePage(); });

  // ---------- Paste / upload image (image page only) ----------
  function handleImageFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = () => {
      const url = reader.result;
      if (url) analyzeImage(url);
    };
    reader.readAsDataURL(file);
  }

  if (isImagePage) {
    document.addEventListener('paste', (e) => {
      if (!e.clipboardData?.items) return;
      for (const item of e.clipboardData.items) {
        if (item.type.indexOf('image') !== -1) {
          e.preventDefault();
          const file = item.getAsFile();
          if (file) {
            const url = URL.createObjectURL(file);
            analyzeImage(url);
          }
          break;
        }
      }
    });

    if (uploadBtn && fileInput) {
      uploadBtn.addEventListener('click', () => fileInput.click());
      fileInput.addEventListener('change', () => {
        const file = fileInput.files?.[0];
        fileInput.value = '';
        handleImageFile(file);
      });
    }

    // Drop zone: click opens file picker, drag-and-drop accepts images (works for side profile too)
    if (imageDropZone && fileInput) {
      imageDropZone.addEventListener('click', (e) => {
        if (getCurrentCapturedState() !== null) return;
        e.preventDefault();
        fileInput.click();
      });
      imageDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
      });
      imageDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        if (getCurrentCapturedState() !== null) return;
        const file = e.dataTransfer.files?.[0];
        if (file) handleImageFile(file);
      });
    }
  }

  // ---------- Init ----------
  async function initCameraPage() {
    setupMetricCardHover();
    await initFaceMesh();
    await startCamera();
    if (video) video.addEventListener('loadedmetadata', () => {
      resizeOverlay();
      scheduleNextSend();
    });
    setMode(false);
  }

  async function initImagePage() {
    setupMetricCardHover();
    await initFaceMesh();
    setMode(false);
  }

  if (isImagePage) initImagePage();
  else initCameraPage();
})();
