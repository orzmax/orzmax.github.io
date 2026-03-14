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
   * Head-roll-compensated canthal tilt. Measures tilt relative to the facial horizon (intercanthal line 133–362).
   * Uses same inverted-Y logic: facial horizon angle from left inner (133) to right inner (362).
   * Returns { leftDeg, rightDeg } with "outer corner higher than inner" = POSITIVE for both eyes.
   */
  function canthalTiltWithHeadRollCompensation(leftInner, leftOuter, rightInner, rightOuter) {
    if (!leftInner || !leftOuter || !rightInner || !rightOuter) return { leftDeg: 0, rightDeg: 0 };

    const toDeg = 180 / Math.PI;

    // 1. Facial horizon: angle of line from left inner (133) to right inner (362). Inverted Y: same convention.
    const horizonDx = rightInner.x - leftInner.x;
    const horizonDy = rightInner.y - leftInner.y; // canvas: right down => horizonDy > 0
    const headRollAngleRad = Math.atan2(horizonDy, horizonDx);

    // 2. Raw angles in degrees (already positive when outer is higher)
    const leftRawDeg = canthalTiltEyeAngleDeg(leftInner, leftOuter);
    const rightRawDeg = canthalTiltEyeAngleDeg(rightInner, rightOuter);

    // 3. Convert raw to radians, subtract head roll, back to degrees and normalize to [-90, 90]
    const normalize90 = (deg) => {
      let d = deg;
      while (d > 90) d -= 180;
      while (d < -90) d += 180;
      return d;
    };
    const leftRelativeRad = (leftRawDeg * Math.PI) / 180 - headRollAngleRad;
    const rightRelativeRad = (rightRawDeg * Math.PI) / 180 - headRollAngleRad;
    const leftDeg = normalize90(leftRelativeRad * toDeg);
    const rightDeg = normalize90(rightRelativeRad * toDeg);

    return { leftDeg, rightDeg };
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

  // ---------- Universal color mapping (legend system): canvas + dashboard ----------
  const METRIC_COLORS = {
    canthalTilt: '#FF1493',
    palpebralFissure: '#00FFFF',
    intercanthalRatio: '#FFD700',
    fwhr: '#39FF14',
    midfaceRatio: '#B026FF',
    bigonialToBizygomatic: '#FF8C00',
    philtrumToChin: '#FF7F50',
    gonialAngleRamus: '#FF0000',
    alarBase: '#8888a0',
    facialConvexity: '#8888a0',
  };

  const DOT_RADIUS = 1.5;
  const LINE_WIDTH_BASE = 1;
  const DEFAULT_OPACITY = 0.3;
  const HOVERED_OPACITY = 1.0;
  const HOVERED_LINE_WIDTH = 2;
  const NON_HOVERED_OPACITY = 0.1;

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
  function overlayScale(width, height) {
    return Math.max(1, Math.min(width, height) / OVERLAY_REF_SIZE);
  }

  function drawFrontalOverlay(landmarks, width, height) {
    const ctx = overlay.getContext('2d');
    const g = (i) => getLandmark(landmarks, i, width, height);
    const scale = overlayScale(width, height);
    const lw = (w) => Math.max(1, Math.round(w * scale));
    const dr = () => Math.max(1.5, DOT_RADIUS * scale);

    const leftInner = g(LANDMARKS.LEFT_EYE_INNER), leftOuter = g(LANDMARKS.LEFT_EYE_OUTER);
    const rightInner = g(LANDMARKS.RIGHT_EYE_INNER), rightOuter = g(LANDMARKS.RIGHT_EYE_OUTER);
    let key = 'canthalTilt';
    let opacity, lineWidth, lineOpacity;
    ({ opacity, lineWidth } = getOpacityAndLineWidth(key));
    lineOpacity = opacity;
    // Canthal tilt: draw only the segment from inner to outer canthus (no extension past corners).
    drawLine(ctx, leftInner, leftOuter, METRIC_COLORS.canthalTilt, lw(lineWidth), lineOpacity);
    drawLine(ctx, rightInner, rightOuter, METRIC_COLORS.canthalTilt, lw(lineWidth), lineOpacity);
    drawDot(ctx, leftInner, METRIC_COLORS.canthalTilt, opacity, dr());
    drawDot(ctx, leftOuter, METRIC_COLORS.canthalTilt, opacity, dr());
    drawDot(ctx, rightInner, METRIC_COLORS.canthalTilt, opacity, dr());
    drawDot(ctx, rightOuter, METRIC_COLORS.canthalTilt, opacity, dr());

    const leftTop = g(LANDMARKS.LEFT_EYE_TOP), leftBottom = g(LANDMARKS.LEFT_EYE_BOTTOM);
    const rightTop = g(LANDMARKS.RIGHT_EYE_TOP), rightBottom = g(LANDMARKS.RIGHT_EYE_BOTTOM);
    key = 'palpebralFissure';
    ({ opacity, lineWidth } = getOpacityAndLineWidth(key));
    lineOpacity = opacity;
    drawLine(ctx, leftTop, leftBottom, METRIC_COLORS.palpebralFissure, lw(lineWidth), lineOpacity);
    drawLine(ctx, rightTop, rightBottom, METRIC_COLORS.palpebralFissure, lw(lineWidth), lineOpacity);
    drawDot(ctx, leftTop, METRIC_COLORS.palpebralFissure, opacity, dr());
    drawDot(ctx, leftBottom, METRIC_COLORS.palpebralFissure, opacity, dr());
    drawDot(ctx, rightTop, METRIC_COLORS.palpebralFissure, opacity, dr());
    drawDot(ctx, rightBottom, METRIC_COLORS.palpebralFissure, opacity, dr());

    key = 'intercanthalRatio';
    ({ opacity, lineWidth } = getOpacityAndLineWidth(key));
    lineOpacity = opacity;
    drawLine(ctx, leftInner, rightInner, METRIC_COLORS.intercanthalRatio, lw(lineWidth), lineOpacity);
    drawDot(ctx, leftInner, METRIC_COLORS.intercanthalRatio, opacity, dr());
    drawDot(ctx, rightInner, METRIC_COLORS.intercanthalRatio, opacity, dr());

    const leftCheek = g(LANDMARKS.LEFT_CHEEK), rightCheek = g(LANDMARKS.RIGHT_CHEEK);
    const glabella = g(LANDMARKS.GLABELLA), philtrumTop = g(LANDMARKS.PHILTRUM_TOP);
    key = 'fwhr';
    ({ opacity, lineWidth } = getOpacityAndLineWidth(key));
    lineOpacity = opacity;
    drawLine(ctx, leftCheek, rightCheek, METRIC_COLORS.fwhr, lw(lineWidth), lineOpacity);
    drawLine(ctx, glabella, philtrumTop, METRIC_COLORS.fwhr, lw(lineWidth), lineOpacity);
    drawDot(ctx, leftCheek, METRIC_COLORS.fwhr, opacity, dr());
    drawDot(ctx, rightCheek, METRIC_COLORS.fwhr, opacity, dr());
    drawDot(ctx, glabella, METRIC_COLORS.fwhr, opacity, dr());
    drawDot(ctx, philtrumTop, METRIC_COLORS.fwhr, opacity, dr());

    const leftIris = landmarks[LANDMARKS.LEFT_IRIS_CENTER];
    const rightIris = landmarks[LANDMARKS.RIGHT_IRIS_CENTER];
    const leftIrisP = leftIris ? toPixel(leftIris, width, height) : { x: (leftInner.x + leftOuter.x) / 2, y: (leftInner.y + leftOuter.y) / 2 };
    const rightIrisP = rightIris ? toPixel(rightIris, width, height) : { x: (rightInner.x + rightOuter.x) / 2, y: (rightInner.y + rightOuter.y) / 2 };
    key = 'midfaceRatio';
    ({ opacity, lineWidth } = getOpacityAndLineWidth(key));
    lineOpacity = opacity;
    drawDot(ctx, leftIrisP, METRIC_COLORS.midfaceRatio, opacity, dr());
    drawDot(ctx, rightIrisP, METRIC_COLORS.midfaceRatio, opacity, dr());
    const midY = (leftIrisP.y + rightIrisP.y) / 2;
    const midP = { x: (leftIrisP.x + rightIrisP.x) / 2, y: midY };
    drawLine(ctx, leftIrisP, rightIrisP, METRIC_COLORS.midfaceRatio, lw(lineWidth), lineOpacity);
    drawLine(ctx, midP, philtrumTop, METRIC_COLORS.midfaceRatio, lw(lineWidth), lineOpacity);
    drawDot(ctx, philtrumTop, METRIC_COLORS.midfaceRatio, opacity, dr());

    const jawP1 = g(LANDMARKS.LEFT_GONION), jawP2 = g(LANDMARKS.RIGHT_GONION);
    const leftGonionDraw = jawP1 && jawP2 && jawP1.x <= jawP2.x ? jawP1 : jawP2;
    const rightGonionDraw = jawP1 && jawP2 && jawP1.x > jawP2.x ? jawP1 : jawP2;
    key = 'bigonialToBizygomatic';
    ({ opacity, lineWidth } = getOpacityAndLineWidth(key));
    lineOpacity = opacity;
    drawLine(ctx, leftGonionDraw, rightGonionDraw, METRIC_COLORS.bigonialToBizygomatic, lw(lineWidth), lineOpacity);
    drawDot(ctx, leftGonionDraw, METRIC_COLORS.bigonialToBizygomatic, opacity, dr());
    drawDot(ctx, rightGonionDraw, METRIC_COLORS.bigonialToBizygomatic, opacity, dr());

    const subnasale = g(LANDMARKS.SUBNASALE), upperLipTop = g(LANDMARKS.UPPER_LIP_TOP);
    const lowerLipBottom = g(LANDMARKS.LOWER_LIP_BOTTOM), menton = g(LANDMARKS.MENTON);
    key = 'philtrumToChin';
    ({ opacity, lineWidth } = getOpacityAndLineWidth(key));
    lineOpacity = opacity;
    drawLine(ctx, subnasale, upperLipTop, METRIC_COLORS.philtrumToChin, lw(lineWidth), lineOpacity);
    drawLine(ctx, lowerLipBottom, menton, METRIC_COLORS.philtrumToChin, lw(lineWidth), lineOpacity);
    drawDot(ctx, subnasale, METRIC_COLORS.philtrumToChin, opacity, dr());
    drawDot(ctx, upperLipTop, METRIC_COLORS.philtrumToChin, opacity, dr());
    drawDot(ctx, lowerLipBottom, METRIC_COLORS.philtrumToChin, opacity, dr());
    drawDot(ctx, menton, METRIC_COLORS.philtrumToChin, opacity, dr());

    const noseLeft = g(LANDMARKS.NOSE_LEFT_ALAR), noseRight = g(LANDMARKS.NOSE_RIGHT_ALAR);
    key = 'alarBase';
    ({ opacity, lineWidth } = getOpacityAndLineWidth(key));
    lineOpacity = opacity;
    drawLine(ctx, noseLeft, noseRight, METRIC_COLORS.alarBase, lw(lineWidth), lineOpacity);
    drawLine(ctx, leftInner, rightInner, METRIC_COLORS.alarBase, lw(lineWidth), lineOpacity);
    drawDot(ctx, noseLeft, METRIC_COLORS.alarBase, opacity, dr());
    drawDot(ctx, noseRight, METRIC_COLORS.alarBase, opacity, dr());
  }

  function drawProfileOverlay(landmarks, width, height, profileSide) {
    const ctx = overlay.getContext('2d');
    const g = (i) => getLandmark(landmarks, i, width, height);
    const scale = overlayScale(width, height);
    const lw = (w) => Math.max(1, Math.round(w * scale));
    const dr = () => Math.max(1.5, DOT_RADIUS * scale);
    const side = profileSide ?? getProfileSide(landmarks, width, height);

    // Gonial angle: only Tragion, Dynamic Gonion, Menton + two connecting lines
    const gonialPoints = getProfileGonialPoints(landmarks, width, height, side);
    if (gonialPoints) {
      const { tragion, gonion, menton } = gonialPoints;
      const key = 'gonialAngleRamus';
      const { opacity, lineWidth } = getOpacityAndLineWidth(key);
      const lineOpacity = opacity;
      drawLine(ctx, tragion, gonion, METRIC_COLORS.gonialAngleRamus, lw(lineWidth), lineOpacity);
      drawLine(ctx, gonion, menton, METRIC_COLORS.gonialAngleRamus, lw(lineWidth), lineOpacity);
      drawDot(ctx, tragion, METRIC_COLORS.gonialAngleRamus, opacity, dr());
      drawDot(ctx, gonion, METRIC_COLORS.gonialAngleRamus, opacity, dr());
      drawDot(ctx, menton, METRIC_COLORS.gonialAngleRamus, opacity, dr());
    }

    const glabella = g(LANDMARKS.GLABELLA);
    const subnasale = g(LANDMARKS.SUBNASALE);
    const menton = g(LANDMARKS.MENTON);
    const key2 = 'facialConvexity';
    const { opacity: op2, lineWidth: lw2 } = getOpacityAndLineWidth(key2);
    const lineOpacity2 = op2;
    drawLine(ctx, glabella, subnasale, METRIC_COLORS.facialConvexity, lw(lw2), lineOpacity2);
    drawLine(ctx, subnasale, menton, METRIC_COLORS.facialConvexity, lw(lw2), lineOpacity2);
    drawDot(ctx, glabella, METRIC_COLORS.facialConvexity, op2, dr());
    drawDot(ctx, subnasale, METRIC_COLORS.facialConvexity, op2, dr());
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
      if (hint) hint.textContent = 'Analyzing image…';
      if (hintSub) { hintSub.textContent = ''; hintSub.classList.add('empty'); }
      faceMesh.send({ image: img }).catch((e) => {
        console.warn('FaceMesh image error', e);
        analyzingImage = false;
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
      value: canthalAvg.toFixed(1) + '°',
      sub: `L: ${leftCanthalDeg.toFixed(1)}° · R: ${rightCanthalDeg.toFixed(1)}°`,
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
      sub: `L: ${leftPF.toFixed(2)} · R: ${rightPF.toFixed(2)}`,
      label: pfRatio >= 3.5 ? 'Wide' : pfRatio <= 2.5 ? 'Narrow' : 'Medium',
    });

    const intercanthalDist = euclidean2D(leftInner, rightInner);
    const oneEyeWidth = (leftWidth + rightWidth) / 2;
    const intercanthalRatio = oneEyeWidth > 0 ? intercanthalDist / oneEyeWidth : 0;
    metrics.push({
      key: 'intercanthalRatio',
      name: 'Intercanthal Ratio',
      value: intercanthalRatio.toFixed(2),
      sub: 'Inner corner dist / eye width',
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
      sub: 'Bizygomatic / upper face height',
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
      sub: 'Interpupillary / (pupil line to upper lip)',
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
      sub: 'Jaw width / cheekbone width',
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
      sub: 'Philtrum length / chin length',
      label: philtrumLabel,
    });

    const noseLeft = g(LANDMARKS.NOSE_LEFT_ALAR), noseRight = g(LANDMARKS.NOSE_RIGHT_ALAR);
    const alarWidth = euclidean2D(noseLeft, noseRight);
    const alarIntercanthalRatio = intercanthalDist > 0 ? alarWidth / intercanthalDist : 0;
    metrics.push({
      key: 'alarBase',
      name: 'Alar-Base to Intercanthal Ratio',
      value: alarIntercanthalRatio.toFixed(2),
      sub: 'Nose width / inner canthal distance',
      label: alarIntercanthalRatio >= 1.05 ? 'Wider nose' : alarIntercanthalRatio <= 0.85 ? 'Narrower nose' : 'Proportional',
    });

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
      sub: 'Glabella–Subnasale–Pogonion',
      label: convexityAngle >= 175 ? 'Flat' : convexityAngle <= 165 ? 'Convex' : 'Neutral',
    });

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
          const borderColor = METRIC_COLORS[key] || METRIC_COLORS.alarBase;
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
      const w = overlay.width || videoWidth();
      const h = overlay.height || videoHeight();
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
