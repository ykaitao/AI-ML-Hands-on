const fs = require('fs');
const path = require('path');
const PptxGenJS = require('pptxgenjs');
const {
  svgToDataUri,
  latexToSvgDataUri,
  codeToRuns,
  imageSizingContain,
  safeOuterShadow,
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require('./pptxgenjs_helpers');

const pptx = new PptxGenJS();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'OpenAI Codex';
pptx.company = 'OpenAI';
pptx.subject = 'LLM architecture components';
pptx.title = 'LLM Architecture Components';
pptx.lang = 'en-US';
pptx.theme = {
  headFontFace: 'Arial',
  bodyFontFace: 'Arial',
  lang: 'en-US',
};
pptx.defineSlideMaster({
  title: 'LLM_MASTER',
  background: { color: 'F8FAFC' },
  objects: [
    { rect: { x: 0, y: 0, w: 13.333, h: 0.22, fill: { color: '2563EB' }, line: { color: '2563EB' } } },
    { rect: { x: 0, y: 7.28, w: 13.333, h: 0.22, fill: { color: '0F172A' }, line: { color: '0F172A' } } },
  ],
  slideNumber: { x: 12.75, y: 7.02, w: 0.35, h: 0.18, color: 'E2E8F0', fontFace: 'Arial', fontSize: 9, align: 'right' },
});

const COLORS = {
  ink: '0F172A',
  slate: '475569',
  muted: '64748B',
  blue: '2563EB',
  cyan: '0891B2',
  purple: '7C3AED',
  green: '059669',
  amber: 'D97706',
  red: 'DC2626',
  panel: 'FFFFFF',
  panelAlt: 'F8FAFC',
  codeBg: '0F172A',
  codeLine: '334155',
  border: 'CBD5E1',
  canvas: 'F8FAFC',
  softBlue: 'DBEAFE',
  softCyan: 'ECFEFF',
  softPurple: 'F5F3FF',
  softGreen: 'ECFDF5',
  softAmber: 'FFF7ED',
};

const SECTION_STYLES = {
  position: { label: 'Position & order', accent: COLORS.blue, soft: COLORS.softBlue, soft2: 'EFF6FF' },
  norm: { label: 'Normalization & score shaping', accent: COLORS.cyan, soft: COLORS.softCyan, soft2: 'F0FDFF' },
  attention: { label: 'Attention & inference', accent: COLORS.purple, soft: COLORS.softPurple, soft2: 'FAF5FF' },
  sparse: { label: 'Sparse capacity', accent: COLORS.green, soft: COLORS.softGreen, soft2: 'F0FDF4' },
  activation: { label: 'Activations & gated MLPs', accent: COLORS.amber, soft: COLORS.softAmber, soft2: 'FFFBEB' },
};

const slideW = 13.333;
const slideH = 7.5;
const leftX = 0.5;
const gutter = 0.35;
const leftW = 6.0;
const rightX = leftX + leftW + gutter;
const rightW = slideW - rightX - 0.5;
const topY = 0.38;

function escapeXml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function addCard(slide, x, y, w, h, opts = {}) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.08,
    fill: { color: opts.fill || COLORS.panel },
    line: { color: opts.line || COLORS.border, width: opts.lineWidth || 1 },
    shadow: safeOuterShadow('94A3B8', 0.14, 45, 2, 1),
  });
  if (opts.accent) {
    slide.addShape(pptx.ShapeType.roundRect, {
      x: x + 0.08,
      y: y + 0.12,
      w: 0.08,
      h: h - 0.24,
      rectRadius: 0.04,
      fill: { color: opts.accent },
      line: { color: opts.accent, width: 1 },
    });
  }
  if (opts.title) {
    slide.addText(opts.title, {
      x: x + (opts.accent ? 0.26 : 0.18),
      y: y + 0.08,
      w: w - (opts.accent ? 0.42 : 0.36),
      h: 0.22,
      fontFace: 'Arial',
      fontSize: opts.titleSize || 12,
      bold: true,
      color: opts.titleColor || COLORS.ink,
      margin: 0,
    });
  }
}

function drawBulletList(slide, items, x, y, w, h, opts = {}) {
  const runs = [];
  items.forEach((item, i) => {
    runs.push({
      text: item,
      options: {
        bullet: { indent: 14 },
        hanging: 3,
      },
    });
    if (i !== items.length - 1) runs.push({ text: '\n', options: {} });
  });
  slide.addText(runs, {
    x,
    y,
    w,
    h,
    fontFace: 'Arial',
    fontSize: opts.fontSize || 13,
    color: opts.color || COLORS.slate,
    margin: 0,
    breakLine: false,
    valign: 'top',
    paraSpaceAfterPt: 5,
  });
}

function addSectionBadge(slide, label, x, y, w, style) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h: 0.3,
    rectRadius: 0.12,
    fill: { color: style.soft },
    line: { color: style.accent, width: 1 },
  });
  slide.addText(label, {
    x: x + 0.12,
    y: y + 0.055,
    w: w - 0.24,
    h: 0.18,
    fontFace: 'Arial',
    fontSize: 10.5,
    bold: true,
    color: style.accent,
    margin: 0,
    align: 'center',
  });
}

function sectionStyleForTitle(title) {
  if (/Positional Embedding|RoPE|ALiBi|NoPE/i.test(title)) return SECTION_STYLES.position;
  if (/Dropout|LayerNorm|RMSNorm|QK-Norm/i.test(title)) return SECTION_STYLES.norm;
  if (/Attention|KV Cache|MQA|GQA|MHA|MLA|Sliding Window|Scaled Dot-Product/i.test(title)) {
    return SECTION_STYLES.attention;
  }
  if (/Mixture-of-Experts|MoE/i.test(title)) return SECTION_STYLES.sparse;
  return SECTION_STYLES.activation;
}

function citationForTitle(title) {
  const map = {
    'Positional Embedding': {
      introduced: 'Vaswani et al., Attention Is All You Need (2017)',
      usedIn: 'GPT-2, BERT, GPT-3',
    },
    'RoPE (Rotary Position Embeddings)': {
      introduced: 'Su et al., RoFormer (2021)',
      usedIn: 'LLaMA/Llama 2/3, DeepSeek-V2/V3',
    },
    'ALiBi (Attention with Linear Biases)': {
      introduced: 'Press et al., Train Short, Test Long (2021)',
      usedIn: 'BLOOM, MPT',
    },
    'NoPE (No Positional Embeddings)': {
      introduced: 'Haviv et al., Transformer LMs without Positional Encodings... (2022)',
      usedIn: 'Mostly research decoder-only LMs; uncommon in mainstream frontier LLMs',
    },
    'Dropout': {
      introduced: 'Srivastava et al., Dropout (2014)',
      usedIn: 'Transformer, BERT, GPT-2',
    },
    'LayerNorm': {
      introduced: 'Ba et al., Layer Normalization (2016)',
      usedIn: 'Transformer, BERT, GPT-2, OPT',
    },
    'RMSNorm': {
      introduced: 'Zhang & Sennrich, RMSNorm (2019)',
      usedIn: 'LLaMA/Llama 2, Mistral, OpenELM',
    },
    'Scaled Dot-Product Attention': {
      introduced: 'Vaswani et al., Attention Is All You Need (2017)',
      usedIn: 'BERT, GPT-2, Llama 2/3',
    },
    'Multi-Head Attention (MHA)': {
      introduced: 'Vaswani et al., Attention Is All You Need (2017)',
      usedIn: 'Transformer, BERT, GPT-2/3',
    },
    'Multi-Query Attention (MQA)': {
      introduced: 'Shazeer, Fast Transformer Decoding (2019)',
      usedIn: 'PaLM, Falcon, StarCoder',
    },
    'Grouped-Query Attention (GQA)': {
      introduced: 'Ainslie et al., GQA (2023)',
      usedIn: 'Llama 2 70B, Llama 3, OpenELM',
    },
    'Multi-Head Latent Attention (MLA)': {
      introduced: 'DeepSeek-AI, DeepSeek-V2 (2024)',
      usedIn: 'DeepSeek-V2, DeepSeek-V3, DeepSeek-R1',
    },
    'Sliding Window Attention (SWA)': {
      introduced: 'Beltagy et al., Longformer (2020)',
      usedIn: 'Longformer, Mistral 7B',
    },
    'KV Cache': {
      introduced: 'Vaswani et al., Attention Is All You Need (2017); fast-decoding analysis in Shazeer (2019)',
      usedIn: 'GPT-2/3, Llama 2/3, DeepSeek',
    },
    'QK-Norm': {
      introduced: 'Henry et al., Query-Key Normalization for Transformers (2020)',
      usedIn: 'OLMo 2, Chameleon',
    },
    'Mixture-of-Experts (MoE)': {
      introduced: 'Jacobs et al., Adaptive Mixtures of Local Experts (1991); Transformer-scale MoE: Shazeer et al. (2017)',
      usedIn: 'Switch Transformer, Mixtral, DeepSeek-V2/V3',
    },
    'GELU': {
      introduced: 'Hendrycks & Gimpel, GELUs (2016)',
      usedIn: 'BERT, GPT-2, Falcon',
    },
    'SiLU / Swish': {
      introduced: 'Elfwing et al., SiLU (2017); popularized as Swish by Ramachandran et al. (2017)',
      usedIn: 'Inside SwiGLU in PaLM, Llama 2/3, Mistral',
    },
    'SwiGLU': {
      introduced: 'Shazeer, GLU Variants Improve Transformer (2020)',
      usedIn: 'PaLM, Llama 2/3, Mistral, OpenELM',
    },
  };
  return map[title] || {
    introduced: 'Primary citation to be filled',
    usedIn: 'Representative LLM models to be filled',
  };
}

function codeRuns(code) {
  return codeToRuns(code.trim(), 'python').map((run) => ({
    text: run.text,
    options: {
      ...run.options,
      fontFace: 'Courier New',
      fontSize: 9.4,
      breakLine: false,
    },
  }));
}

function drawMultilineText(lines, x, y, w, h, opts = {}) {
  const lineHeight = opts.lineHeight || 18;
  let currentY = y;
  const items = Array.isArray(lines) ? lines : String(lines).split('\n');
  const textBlock = items
    .map((line, idx) => {
      const dy = idx === 0 ? 0 : lineHeight;
      return `<tspan x="${x + w / 2}" dy="${dy}">${escapeXml(line)}</tspan>`;
    })
    .join('');
  return `<text x="${x + w / 2}" y="${currentY + h / 2 - ((items.length - 1) * lineHeight) / 2}" text-anchor="middle" font-family="Arial, Arial, sans-serif" font-size="${opts.fontSize || 16}" font-weight="${opts.fontWeight || 700}" fill="${opts.color || '#0F172A'}">${textBlock}</text>`;
}

function flowDiagramSvg(spec) {
  const width = spec.width || 800;
  const height = spec.height || 360;
  const frameAccent = spec.accent || '#E2E8F0';
  const defs = `
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748B" />
      </marker>
    </defs>`;

  const nodes = (spec.nodes || [])
    .map((node) => {
      const fill = node.fill || '#EEF2FF';
      const stroke = node.stroke || '#94A3B8';
      const labelLines = String(node.label || '').split('\n');
      const main = labelLines;
      const nodeSvg = `
        <rect x="${node.x}" y="${node.y}" rx="18" ry="18" width="${node.w}" height="${node.h}" fill="${fill}" stroke="${stroke}" stroke-width="2" />
        ${drawMultilineText(main, node.x, node.y - 4, node.w, node.h * 0.62, { fontSize: node.fontSize || 17, fontWeight: 700 })}
        ${node.shape ? `<text x="${node.x + node.w / 2}" y="${node.y + node.h - 16}" text-anchor="middle" font-family="Arial, Arial, sans-serif" font-size="13" fill="#475569">${escapeXml(node.shape)}</text>` : ''}`;
      return nodeSvg;
    })
    .join('');

  const edges = (spec.edges || [])
    .map((edge) => {
      const label = edge.label
        ? `<text x="${edge.lx || (edge.x1 + edge.x2) / 2}" y="${edge.ly || (edge.y1 + edge.y2) / 2 - 6}" text-anchor="middle" font-family="Arial, Arial, sans-serif" font-size="12" fill="#475569">${escapeXml(edge.label)}</text>`
        : '';
      return `<line x1="${edge.x1}" y1="${edge.y1}" x2="${edge.x2}" y2="${edge.y2}" stroke="#64748B" stroke-width="3" marker-end="url(#arrow)" />${label}`;
    })
    .join('');

  const caption = spec.caption
    ? `<text x="16" y="${height - 14}" font-family="Arial, Arial, sans-serif" font-size="13" fill="#475569">${escapeXml(spec.caption)}</text>`
    : '';

  return `
  <svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
    ${defs}
    <rect x="1" y="1" width="${width - 2}" height="${height - 2}" rx="22" ry="22" fill="#FFFFFF" stroke="${frameAccent}" stroke-width="2" />
    <rect x="18" y="16" width="120" height="8" rx="4" ry="4" fill="${frameAccent}" opacity="0.35" />
    ${edges}
    ${nodes}
    ${caption}
  </svg>`;
}

function addTitleSlide() {
  const slide = pptx.addSlide('LLM_MASTER');
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 9.72,
    y: 0.7,
    w: 2.48,
    h: 0.5,
    rectRadius: 0.16,
    fill: { color: 'E0E7FF' },
    line: { color: 'C7D2FE', width: 1 },
  });
  slide.addText('Merged overview', {
    x: 9.84,
    y: 0.85,
    w: 2.2,
    h: 0.18,
    fontFace: 'Arial',
    fontSize: 11,
    bold: true,
    color: SECTION_STYLES.attention.accent,
    margin: 0,
    align: 'center',
  });
  slide.addText('LLM Architecture Components', {
    x: 0.6,
    y: 0.68,
    w: 7.7,
    h: 0.55,
    fontFace: 'Arial',
    fontSize: 26,
    bold: true,
    color: COLORS.ink,
    margin: 0,
  });
  slide.addText('One-slide roadmap: from embeddings and positional structure to attention, normalization, and feed-forward variants used in modern LLMs', {
    x: 0.62,
    y: 1.3,
    w: 9.4,
    h: 0.28,
    fontFace: 'Arial',
    fontSize: 14,
    color: COLORS.slate,
    margin: 0,
  });

  addCard(slide, 0.6, 1.78, 5.95, 1.82, {
    title: 'Embedding',
    fill: SECTION_STYLES.position.soft2,
    accent: SECTION_STYLES.position.accent,
    titleColor: SECTION_STYLES.position.accent,
  });
  drawBulletList(slide, ['Positional Embedding', 'RoPE', 'ALiBi', 'NoPE'], 0.9, 2.25, 5.1, 0.98, { fontSize: 14.3 });
  slide.addText('These decide how token order is represented before or inside attention.', {
    x: 0.9,
    y: 3.32,
    w: 5.0,
    h: 0.16,
    fontFace: 'Arial',
    fontSize: 10.3,
    color: COLORS.slate,
    margin: 0,
  });

  addCard(slide, 6.78, 1.78, 5.95, 1.82, {
    title: 'Attention',
    fill: SECTION_STYLES.attention.soft2,
    accent: SECTION_STYLES.attention.accent,
    titleColor: SECTION_STYLES.attention.accent,
  });
  drawBulletList(slide, ['Scaled Dot-Product', 'MHA', 'MQA', 'GQA', 'MLA', 'Sliding Window Attention', 'KV Cache'], 7.08, 2.15, 5.0, 1.12, { fontSize: 13.1 });
  slide.addText('These govern token-token interaction patterns and decoding efficiency.', {
    x: 7.08,
    y: 3.32,
    w: 5.0,
    h: 0.16,
    fontFace: 'Arial',
    fontSize: 10.3,
    color: COLORS.slate,
    margin: 0,
  });

  addCard(slide, 0.6, 3.95, 5.95, 1.82, {
    title: 'Normalization',
    fill: SECTION_STYLES.norm.soft2,
    accent: SECTION_STYLES.norm.accent,
    titleColor: SECTION_STYLES.norm.accent,
  });
  drawBulletList(slide, ['Dropout', 'LayerNorm', 'RMSNorm', 'QK-Norm'], 0.9, 4.42, 5.05, 0.95, { fontSize: 14.3 });
  slide.addText('These control activation scale, training stability, and attention score magnitude.', {
    x: 0.9,
    y: 5.47,
    w: 5.0,
    h: 0.16,
    fontFace: 'Arial',
    fontSize: 10.3,
    color: COLORS.slate,
    margin: 0,
  });

  addCard(slide, 6.78, 3.95, 5.95, 1.82, {
    title: 'FeedForward',
    fill: 'FFFBEB',
    accent: SECTION_STYLES.activation.accent,
    titleColor: SECTION_STYLES.activation.accent,
  });
  drawBulletList(slide, ['Mixture-of-Experts', 'GELU', 'SiLU', 'SwiGLU'], 7.08, 4.42, 5.0, 0.95, { fontSize: 14.3 });
  slide.addText('These determine per-token nonlinear transformation and sparse capacity scaling.', {
    x: 7.08,
    y: 5.47,
    w: 5.0,
    h: 0.16,
    fontFace: 'Arial',
    fontSize: 10.3,
    color: COLORS.slate,
    margin: 0,
  });

  addCard(slide, 0.6, 5.98, 12.13, 0.76, {
    title: 'Notation used throughout',
    fill: 'FCFCFD',
    accent: SECTION_STYLES.position.accent,
    titleColor: SECTION_STYLES.position.accent,
    titleSize: 11,
  });
  slide.addText(`B=batch  •  T=sequence length  •  d=model dim  •  H=heads  •  d_h=per-head dim  •  G=KV groups  •  r=latent rank  •  E=experts  •  k=top-k routed experts
PyTorch snippets assume: import torch, torch.nn as nn, torch.nn.functional as F`, {
    x: 0.88,
    y: 6.38,
    w: 11.2,
    h: 0.22,
    fontFace: 'Arial',
    fontSize: 9.3,
    color: COLORS.slate,
    margin: 0,
    breakLine: false,
    valign: 'mid',
  });

  slide.addText('Created for lecture_notes/LLMs.pptx • editable PowerPoint deck authored with PptxGenJS', {
    x: 0.62,
    y: 6.96,
    w: 8.2,
    h: 0.18,
    fontFace: 'Arial',
    fontSize: 10,
    color: 'E2E8F0',
    margin: 0,
  });
  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

function addOverviewSlide() {
  const slide = pptx.addSlide('LLM_MASTER');
  slide.addText('Roadmap of component slides', {
    x: 0.62,
    y: 0.55,
    w: 6.8,
    h: 0.4,
    fontFace: 'Arial',
    fontSize: 22,
    bold: true,
    color: COLORS.ink,
    margin: 0,
  });
  slide.addText('The user prompt repeated GQA twice; this deck keeps a single dedicated GQA slide.', {
    x: 0.62,
    y: 0.98,
    w: 7.2,
    h: 0.24,
    fontFace: 'Arial',
    fontSize: 11,
    color: COLORS.muted,
    margin: 0,
  });

  addCard(slide, 0.6, 1.45, 5.9, 2.2, {
    title: SECTION_STYLES.position.label,
    fill: SECTION_STYLES.position.soft2,
    accent: SECTION_STYLES.position.accent,
    titleColor: SECTION_STYLES.position.accent,
  });
  drawBulletList(slide, ['Positional Embedding', 'RoPE', 'ALiBi', 'NoPE'], 0.9, 1.93, 5.1, 1.35, { fontSize: 14.5 });

  addCard(slide, 6.83, 1.45, 5.9, 2.2, {
    title: SECTION_STYLES.attention.label,
    fill: SECTION_STYLES.attention.soft2,
    accent: SECTION_STYLES.attention.accent,
    titleColor: SECTION_STYLES.attention.accent,
  });
  drawBulletList(
    slide,
    ['Scaled Dot-Product Attention', 'Multi-Head Attention', 'MQA', 'GQA', 'MLA', 'Sliding Window Attention', 'KV cache'],
    7.13,
    1.93,
    5.05,
    1.45,
    { fontSize: 13.2 }
  );

  addCard(slide, 0.6, 4.0, 5.9, 2.35, {
    title: SECTION_STYLES.norm.label,
    fill: SECTION_STYLES.norm.soft2,
    accent: SECTION_STYLES.norm.accent,
    titleColor: SECTION_STYLES.norm.accent,
  });
  drawBulletList(slide, ['Dropout', 'LayerNorm', 'RMSNorm', 'QK-Norm'], 0.9, 4.5, 5.1, 1.0, { fontSize: 14.5 });
  slide.addText('These slides focus on stability, regularization, and how score magnitudes are controlled during training and inference.', {
    x: 0.9,
    y: 5.88,
    w: 5.1,
    h: 0.28,
    fontFace: 'Arial',
    fontSize: 10.5,
    color: COLORS.slate,
    margin: 0,
  });

  addCard(slide, 6.83, 4.0, 5.9, 2.35, {
    title: 'Sparse capacity & activations',
    fill: 'FFFBEB',
    accent: SECTION_STYLES.activation.accent,
    titleColor: SECTION_STYLES.activation.accent,
  });
  drawBulletList(slide, ['Mixture-of-Experts', 'GELU', 'SiLU', 'SwiGLU'], 7.13, 4.5, 5.05, 1.0, { fontSize: 14.5 });
  slide.addText('Each slide is pedagogical rather than model-specific: shapes reflect the common PyTorch tensor contracts used in LLM implementations.', {
    x: 7.13,
    y: 5.88,
    w: 5.05,
    h: 0.28,
    fontFace: 'Arial',
    fontSize: 10.5,
    color: COLORS.slate,
    margin: 0,
  });

  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

function addComponentSlide(component) {
  const slide = pptx.addSlide('LLM_MASTER');
  const section = sectionStyleForTitle(component.title);
  const citation = citationForTitle(component.title);
  const formulaSvg = latexToSvgDataUri(component.formula);

  slide.addText(component.title, {
    x: 0.62,
    y: 0.52,
    w: 7.4,
    h: 0.36,
    fontFace: 'Arial',
    fontSize: 22,
    bold: true,
    color: COLORS.ink,
    margin: 0,
  });
  addSectionBadge(slide, section.label, 10.08, 0.56, 2.65, section);
  slide.addText(component.summary, {
    x: 0.64,
    y: 0.92,
    w: 9.2,
    h: 0.28,
    fontFace: 'Arial',
    fontSize: 13,
    color: COLORS.slate,
    margin: 0,
  });

  addCard(slide, leftX, 1.28, leftW, 1.34, {
    title: 'Math formula / tensor contract',
    fill: section.soft2,
    accent: section.accent,
    titleColor: section.accent,
  });
  slide.addImage({
    data: formulaSvg,
    ...imageSizingContain(formulaSvg, leftX + 0.22, 1.69, leftW - 0.44, 0.50),
  });
  slide.addText(component.formulaNote, {
    x: leftX + 0.22,
    y: 2.27,
    w: leftW - 0.44,
    h: 0.22,
    fontFace: 'Arial',
    fontSize: 10.5,
    color: COLORS.muted,
    margin: 0,
  });

  addCard(slide, leftX, 2.82, leftW, 3.85, {
    title: 'Shape-aware diagram',
    fill: COLORS.panel,
    titleColor: section.accent,
  });
  slide.addImage({
    data: svgToDataUri(flowDiagramSvg({ ...component.diagram, accent: section.accent })),
    x: leftX + 0.12,
    y: 3.18,
    w: leftW - 0.24,
    h: 3.22,
  });

  addCard(slide, rightX, 1.28, rightW, 5.39, {
    title: 'PyTorch implementation',
    fill: COLORS.codeBg,
    line: section.accent,
    accent: section.accent,
    titleColor: 'E2E8F0',
  });
  slide.addText(codeRuns(component.code), {
    x: rightX + 0.18,
    y: 1.62,
    w: rightW - 0.36,
    h: 4.88,
    margin: 0,
    valign: 'top',
    breakLine: false,
    color: 'FFFFFF',
  });

  slide.addText(component.legend, {
    x: 0.62,
    y: 6.76,
    w: 11.8,
    h: 0.16,
    fontFace: 'Arial',
    fontSize: 9.2,
    color: COLORS.muted,
    margin: 0,
  });
  slide.addText(`Introduced in: ${citation.introduced}   •   Used in: ${citation.usedIn}`, {
    x: 0.62,
    y: 6.95,
    w: 11.8,
    h: 0.20,
    fontFace: 'Arial',
    fontSize: 9.0,
    color: COLORS.ink,
    margin: 0,
  });

  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

const components = [
  {
    title: 'Positional Embedding',
    summary: 'The original Transformer paper uses fixed sinusoidal positional encodings, added to token embeddings at the bottoms of the encoder and decoder stacks.',
    formula: String.raw`\operatorname{PE}_{(pos,2i)} = \sin\!\left(pos / 10000^{2i/d_{model}}\right),\qquad \operatorname{PE}_{(pos,2i+1)} = \cos\!\left(pos / 10000^{2i/d_{model}}\right)`,
    formulaNote: 'In Attention Is All You Need, the stack input is √d_model · token_embedding + PE, with PE shared across the batch.',
    legend: 'Shapes: token ids [B,T], token embeddings [B,T,d_model], sinusoidal PE table [T,d_model], stack input [B,T,d_model].',
    diagram: {
      nodes: [
        { x: 25, y: 70, w: 165, h: 82, label: 'Token IDs', shape: '[B,T]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 220, y: 70, w: 175, h: 82, label: 'Embedding · √d_model', shape: '[B,T,d_model]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 25, y: 235, w: 165, h: 82, label: 'Positions', shape: '[T]', fill: '#ECFEFF', stroke: '#0891B2' },
        { x: 220, y: 235, w: 175, h: 82, label: 'Sin / Cos PE', shape: '[T,d_model]', fill: '#ECFEFF', stroke: '#0891B2' },
        { x: 455, y: 150, w: 110, h: 92, label: 'Add', shape: 'broadcast on B', fill: '#FEF3C7', stroke: '#D97706' },
        { x: 625, y: 150, w: 150, h: 92, label: 'Encoder / Decoder input', shape: '[B,T,d_model]', fill: '#ECFDF5', stroke: '#059669' },
      ],
      edges: [
        { x1: 190, y1: 111, x2: 220, y2: 111 },
        { x1: 190, y1: 276, x2: 220, y2: 276 },
        { x1: 395, y1: 111, x2: 455, y2: 178 },
        { x1: 395, y1: 276, x2: 455, y2: 214 },
        { x1: 565, y1: 196, x2: 625, y2: 196 },
      ],
      caption: 'This matches the original Transformer: fixed sinusoidal PE is added to scaled token embeddings before entering the stack.'
    },
    code: `class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, tok_emb):
        tok_emb = tok_emb * (tok_emb.size(-1) ** 0.5)
        return tok_emb + self.pe[:, :tok_emb.size(1)]`,
  },
  {
    title: 'RoPE (Rotary Position Embeddings)',
    summary: 'Rotates paired Q/K coordinates by position-dependent angles, turning dot products into a relative-position-aware similarity measure.',
    formula: String.raw`\langle R_m\mathbf{q}, R_n\mathbf{k}\rangle = \langle \mathbf{q}, R_{n-m}\mathbf{k}\rangle,\qquad R_t = \bigoplus_i \begin{bmatrix}\cos\theta_{t,i} & -\sin\theta_{t,i}\\ \sin\theta_{t,i} & \cos\theta_{t,i}\end{bmatrix}`,
    formulaNote: 'RoPE is applied after Q/K projection and before computing attention scores.',
    legend: 'Shapes: Q,K [B,H,T,d_h], angle table [T,d_h/2], rotated Q/K keep the same shape, scores [B,H,T,T].',
    diagram: {
      nodes: [
        { x: 30, y: 125, w: 150, h: 82, label: 'Q', shape: '[B, H, T, d_h]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 30, y: 235, w: 150, h: 82, label: 'K', shape: '[B, H, T, d_h]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 235, y: 60, w: 170, h: 82, label: 'Angles', shape: '[T, d_h/2]', fill: '#ECFEFF', stroke: '#0891B2' },
        { x: 235, y: 125, w: 170, h: 82, label: 'Rotate Q', shape: '[B, H, T, d_h]', fill: '#EDE9FE', stroke: '#7C3AED' },
        { x: 235, y: 235, w: 170, h: 82, label: 'Rotate K', shape: '[B, H, T, d_h]', fill: '#EDE9FE', stroke: '#7C3AED' },
        { x: 470, y: 180, w: 150, h: 82, label: 'QKᵀ / √d_h', shape: '[B, H, T, T]', fill: '#FEF3C7', stroke: '#D97706' },
        { x: 655, y: 180, w: 110, h: 82, label: 'Scores', shape: '[B, H, T, T]', fill: '#ECFDF5', stroke: '#059669' },
      ],
      edges: [
        { x1: 180, y1: 166, x2: 235, y2: 166 },
        { x1: 180, y1: 276, x2: 235, y2: 276 },
        { x1: 320, y1: 142, x2: 320, y2: 125 },
        { x1: 320, y1: 142, x2: 320, y2: 235 },
        { x1: 405, y1: 166, x2: 470, y2: 208 },
        { x1: 405, y1: 276, x2: 470, y2: 234 },
        { x1: 620, y1: 221, x2: 655, y2: 221 },
      ],
      caption: 'RoPE injects position into Q and K without adding a positional vector to the hidden state itself.'
    },
    code: `def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x * cos + rot * sin

q = apply_rope(q, cos, sin)   # [B,H,T,d_h]
k = apply_rope(k, cos, sin)   # [B,H,T,d_h]`,
  },
  {
    title: 'ALiBi (Attention with Linear Biases)',
    summary: 'Adds a head-specific linear distance penalty directly to attention scores so the model can represent recency bias without explicit positional vectors.',
    formula: String.raw`S_{ij}^{(h)} = \frac{\mathbf{q}_i^{(h)}\mathbf{k}_j^{(h)\top}}{\sqrt{d_h}} - m_h(i-j),\qquad j \le i`,
    formulaNote: 'Different heads use different slopes m_h, giving a spectrum from local to more global attention bias.',
    legend: 'Shapes: Q,K [B,H,T,d_h], raw scores [B,H,T,T], ALiBi bias [H,T,T], biased scores [B,H,T,T].',
    diagram: {
      nodes: [
        { x: 25, y: 85, w: 135, h: 80, label: 'Q', shape: '[B,H,T,d_h]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 25, y: 235, w: 135, h: 80, label: 'K', shape: '[B,H,T,d_h]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 215, y: 125, w: 155, h: 90, label: 'QKᵀ / √d_h', shape: '[B,H,T,T]', fill: '#FEF3C7', stroke: '#D97706' },
        { x: 215, y: 250, w: 155, h: 80, label: 'ALiBi slopes', shape: '[H] → [H,T,T]', fill: '#ECFEFF', stroke: '#0891B2' },
        { x: 430, y: 150, w: 145, h: 90, label: 'Add bias', shape: '[B,H,T,T]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 635, y: 150, w: 130, h: 90, label: 'Biased scores', shape: '[B,H,T,T]', fill: '#ECFDF5', stroke: '#059669' },
      ],
      edges: [
        { x1: 160, y1: 125, x2: 215, y2: 160 },
        { x1: 160, y1: 275, x2: 215, y2: 185 },
        { x1: 370, y1: 170, x2: 430, y2: 182 },
        { x1: 370, y1: 290, x2: 430, y2: 208 },
        { x1: 575, y1: 195, x2: 635, y2: 195 },
      ],
      caption: 'ALiBi changes only the score matrix, so no extra position embeddings or rotary rotations are stored.'
    },
    code: `def alibi_attention(q, k, v, slopes):
    B, H, T, D = q.shape
    pos = torch.arange(T, device=q.device)
    dist = (pos[:, None] - pos[None, :]).clamp_min(0)      # [T,T]
    bias = -slopes[:, None, None] * dist.unsqueeze(0)      # [H,T,T]
    scores = q @ k.transpose(-2, -1) / D ** 0.5
    scores = scores + bias.unsqueeze(0)
    weights = scores.softmax(dim=-1)
    return weights @ v`,
  },
  {
    title: 'NoPE (No Positional Embeddings)',
    summary: 'Removes explicit positional signals entirely; order then must emerge from masking, data statistics, or other architectural biases.',
    formula: String.raw`\mathbf{Y} = \operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_h}} + \mathbf{M}_{causal}\right)\mathbf{V},\qquad \text{with no } \mathbf{p}_t \text{ added to } \mathbf{x}_t`,
    formulaNote: 'The attention rule is unchanged, but there is no learned/sinusoidal/rotary positional term.',
    legend: 'Shapes: X [B,T,d], Q/K/V [B,H,T,d_h], scores [B,H,T,T], output [B,T,d].',
    diagram: {
      nodes: [
        { x: 28, y: 150, w: 140, h: 84, label: 'X', shape: '[B, T, d]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 220, y: 60, w: 145, h: 84, label: 'Q proj', shape: '[B, H, T, d_h]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 220, y: 150, w: 145, h: 84, label: 'K proj', shape: '[B, H, T, d_h]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 220, y: 240, w: 145, h: 84, label: 'V proj', shape: '[B, H, T, d_h]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 425, y: 110, w: 155, h: 94, label: 'Attention', shape: '[B, H, T, T]', fill: '#FEF3C7', stroke: '#D97706' },
        { x: 630, y: 150, w: 140, h: 84, label: 'Y', shape: '[B, T, d]', fill: '#ECFDF5', stroke: '#059669' },
      ],
      edges: [
        { x1: 168, y1: 192, x2: 220, y2: 102 },
        { x1: 168, y1: 192, x2: 220, y2: 192 },
        { x1: 168, y1: 192, x2: 220, y2: 282 },
        { x1: 365, y1: 102, x2: 425, y2: 142 },
        { x1: 365, y1: 192, x2: 425, y2: 157 },
        { x1: 365, y1: 282, x2: 425, y2: 173 },
        { x1: 580, y1: 157, x2: 630, y2: 192 },
      ],
      caption: 'NoPE keeps the core attention pipeline but omits explicit positional encodings.'
    },
    code: `def nope_attention(x, wq, wk, wv, mask=None):
    q = wq(x)  # [B,T,H*d_h]
    k = wk(x)
    v = wv(x)
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    return scores.softmax(dim=-1) @ v`,
  },
  {
    title: 'Dropout',
    summary: 'Randomly drops activations during training and rescales the survivors, reducing co-adaptation and overfitting.',
    formula: String.raw`\mathbf{y} = \frac{\mathbf{m} \odot \mathbf{x}}{1-p},\qquad m_i \sim \operatorname{Bernoulli}(1-p)` ,
    formulaNote: 'At evaluation time, dropout is disabled and the identity map is used.',
    legend: 'Shapes: x, mask, y all share the same tensor shape, often [B,T,d] or [B,H,T,T] in attention dropout.',
    diagram: {
      nodes: [
        { x: 55, y: 145, w: 150, h: 84, label: 'Input x', shape: '[B, T, d]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 275, y: 70, w: 170, h: 84, label: 'Bernoulli mask', shape: '[B, T, d]', fill: '#FEE2E2', stroke: '#DC2626' },
        { x: 275, y: 215, w: 170, h: 84, label: 'Scale by 1/(1-p)', shape: '[B, T, d]', fill: '#FEF3C7', stroke: '#D97706' },
        { x: 505, y: 145, w: 120, h: 84, label: 'Dropout', shape: 'elementwise', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 670, y: 145, w: 90, h: 84, label: 'y', shape: '[B, T, d]', fill: '#ECFDF5', stroke: '#059669' },
      ],
      edges: [
        { x1: 205, y1: 187, x2: 505, y2: 168 },
        { x1: 360, y1: 154, x2: 505, y2: 186 },
        { x1: 445, y1: 257, x2: 505, y2: 204 },
        { x1: 625, y1: 187, x2: 670, y2: 187 },
      ],
      caption: 'Dropout is a training-time regularizer applied elementwise to activations or attention weights.'
    },
    code: `class DropoutBlock(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return self.drop(x)`,
  },
  {
    title: 'LayerNorm',
    summary: 'Normalizes each token across its feature dimension using mean and variance, then applies learned affine parameters.',
    formula: String.raw`\operatorname{LN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x}-\mu}{\sqrt{\sigma^2+\epsilon}} + \boldsymbol{\beta}`,
    formulaNote: 'In Transformers, μ and σ² are computed over the last dimension d for each token independently.',
    legend: 'Shapes: x,y [B,T,d]; mean and variance are [B,T,1]; γ and β are [d].',
    diagram: {
      nodes: [
        { x: 40, y: 150, w: 130, h: 82, label: 'x', shape: '[B, T, d]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 225, y: 60, w: 165, h: 82, label: 'Mean μ', shape: '[B, T, 1]', fill: '#FEF3C7', stroke: '#D97706' },
        { x: 225, y: 150, w: 165, h: 82, label: 'Variance σ²', shape: '[B, T, 1]', fill: '#FEF3C7', stroke: '#D97706' },
        { x: 225, y: 240, w: 165, h: 82, label: 'γ, β', shape: '[d]', fill: '#ECFEFF', stroke: '#0891B2' },
        { x: 455, y: 150, w: 150, h: 82, label: 'Normalize', shape: '[B, T, d]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 660, y: 150, w: 100, h: 82, label: 'y', shape: '[B, T, d]', fill: '#ECFDF5', stroke: '#059669' },
      ],
      edges: [
        { x1: 170, y1: 191, x2: 225, y2: 101 },
        { x1: 170, y1: 191, x2: 225, y2: 191 },
        { x1: 170, y1: 191, x2: 455, y2: 191 },
        { x1: 390, y1: 101, x2: 455, y2: 174 },
        { x1: 390, y1: 191, x2: 455, y2: 191 },
        { x1: 390, y1: 281, x2: 455, y2: 208 },
        { x1: 605, y1: 191, x2: 660, y2: 191 },
      ],
      caption: 'LayerNorm subtracts the per-token mean and rescales by the per-token standard deviation.'
    },
    code: `class LayerNormLastDim(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta`,
  },
  {
    title: 'RMSNorm',
    summary: 'Normalizes only by root-mean-square magnitude and omits mean subtraction, which is cheaper and popular in modern LLMs.',
    formula: String.raw`\operatorname{RMSNorm}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}`,
    formulaNote: 'RMSNorm keeps the direction of x but rescales its RMS magnitude on the last dimension.',
    legend: 'Shapes: x,y [B,T,d]; rms factor [B,T,1]; learned scale γ [d].',
    diagram: {
      nodes: [
        { x: 40, y: 150, w: 130, h: 82, label: 'x', shape: '[B, T, d]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 225, y: 85, w: 180, h: 82, label: 'RMS factor', shape: '[B, T, 1]', fill: '#FEF3C7', stroke: '#D97706' },
        { x: 225, y: 235, w: 180, h: 82, label: 'γ', shape: '[d]', fill: '#ECFEFF', stroke: '#0891B2' },
        { x: 470, y: 150, w: 150, h: 82, label: 'Scale x / RMS', shape: '[B, T, d]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 670, y: 150, w: 95, h: 82, label: 'y', shape: '[B, T, d]', fill: '#ECFDF5', stroke: '#059669' },
      ],
      edges: [
        { x1: 170, y1: 191, x2: 225, y2: 126 },
        { x1: 170, y1: 191, x2: 470, y2: 191 },
        { x1: 405, y1: 126, x2: 470, y2: 174 },
        { x1: 405, y1: 276, x2: 470, y2: 208 },
        { x1: 620, y1: 191, x2: 670, y2: 191 },
      ],
      caption: 'RMSNorm uses a per-token scalar normalization factor, then a learned featurewise scale.'
    },
    code: `class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight`,
  },
  {
    title: 'Scaled Dot-Product Attention',
    summary: 'The core attention operator: compare queries against keys, normalize with softmax, then aggregate values.',
    formula: String.raw`\operatorname{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_h}} + \mathbf{M}\right)\mathbf{V}`,
    formulaNote: 'The mask M is often causal and/or padded; scores are normalized along the key axis.',
    legend: 'Shapes: Q [B,H,T_q,d_h], K [B,H,T_k,d_h], V [B,H,T_k,d_v], output [B,H,T_q,d_v].',
    diagram: {
      nodes: [
        { x: 25, y: 70, w: 135, h: 80, label: 'Q', shape: '[B,H,T_q,d_h]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 25, y: 170, w: 135, h: 80, label: 'K', shape: '[B,H,T_k,d_h]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 25, y: 270, w: 135, h: 80, label: 'V', shape: '[B,H,T_k,d_v]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 220, y: 115, w: 160, h: 90, label: 'QKᵀ / √d_h', shape: '[B,H,T_q,T_k]', fill: '#FEF3C7', stroke: '#D97706' },
        { x: 220, y: 250, w: 160, h: 80, label: 'Mask M', shape: '[1,1,T_q,T_k]', fill: '#FEE2E2', stroke: '#DC2626' },
        { x: 450, y: 115, w: 150, h: 90, label: 'Softmax', shape: '[B,H,T_q,T_k]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 650, y: 115, w: 120, h: 90, label: 'Output', shape: '[B,H,T_q,d_v]', fill: '#ECFDF5', stroke: '#059669' },
      ],
      edges: [
        { x1: 160, y1: 110, x2: 220, y2: 145 },
        { x1: 160, y1: 210, x2: 220, y2: 175 },
        { x1: 300, y1: 250, x2: 300, y2: 205 },
        { x1: 380, y1: 160, x2: 450, y2: 160 },
        { x1: 160, y1: 310, x2: 650, y2: 180 },
        { x1: 600, y1: 160, x2: 650, y2: 160 },
      ],
      caption: 'Attention scores live in token-token space [T_q,T_k], while values carry the features to be aggregated.'
    },
    code: `def scaled_dot_product_attention(q, k, v, mask=None):
    d_h = q.size(-1)
    scores = q @ k.transpose(-2, -1) / d_h ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    weights = scores.softmax(dim=-1)
    return weights @ v`,
  },
  {
    title: 'Multi-Head Attention (MHA)',
    summary: 'Projects Q/K/V into multiple subspaces, runs attention independently in each head, then concatenates the results.',
    formula: String.raw`\operatorname{MHA}(\mathbf{X}) = \operatorname{Concat}(\operatorname{head}_1,\dots,\operatorname{head}_H)W_O,\qquad \operatorname{head}_h = \operatorname{Attn}(\mathbf{Q}_h,\mathbf{K}_h,\mathbf{V}_h)` ,
    formulaNote: 'Each head gets its own learned projections and can specialize to different patterns.',
    legend: 'Shapes: X [B,T,d], Q/K/V [B,H,T,d_h], concat [B,T,H·d_h], projected output [B,T,d].',
    diagram: {
      nodes: [
        { x: 20, y: 150, w: 130, h: 82, label: 'X', shape: '[B, T, d]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 190, y: 60, w: 150, h: 82, label: 'QKV proj', shape: '[B,H,T,d_h]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 190, y: 240, w: 150, h: 82, label: 'Split heads', shape: 'H branches', fill: '#ECFEFF', stroke: '#0891B2' },
        { x: 390, y: 150, w: 155, h: 82, label: 'Per-head attn', shape: 'H × [B,T,d_h]', fill: '#FEF3C7', stroke: '#D97706' },
        { x: 595, y: 150, w: 120, h: 82, label: 'Concat', shape: '[B,T,H·d_h]', fill: '#EDE9FE', stroke: '#7C3AED' },
        { x: 730, y: 150, w: 55, h: 82, label: 'Wₒ', shape: '[B,T,d]', fill: '#ECFDF5', stroke: '#059669' },
      ],
      edges: [
        { x1: 150, y1: 191, x2: 190, y2: 101 },
        { x1: 150, y1: 191, x2: 190, y2: 281 },
        { x1: 340, y1: 101, x2: 390, y2: 174 },
        { x1: 340, y1: 281, x2: 390, y2: 208 },
        { x1: 545, y1: 191, x2: 595, y2: 191 },
        { x1: 715, y1: 191, x2: 730, y2: 191 },
      ],
      caption: 'Parallel heads let the model attend to different interactions in different learned subspaces.'
    },
    code: `class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.h = num_heads
        self.d_h = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.d_h).transpose(1, 2)
        k = k.view(B, T, self.h, self.d_h).transpose(1, 2)
        v = v.view(B, T, self.h, self.d_h).transpose(1, 2)
        y = scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.out(y)`,
  },
  {
    title: 'Multi-Query Attention (MQA)',
    summary: 'Uses many query heads but a single shared key/value head, giving the smallest KV cache among standard attention-sharing variants.',
    formula: String.raw`\operatorname{head}_h = \operatorname{Attn}(\mathbf{Q}_h,\mathbf{K}_1,\mathbf{V}_1),\qquad h=1,\dots,H`,
    formulaNote: 'MQA is the extreme sharing case: all query heads reuse the same K and V tensors.',
    legend: 'Shapes: Q [B,H,T,d_h], K/V [B,1,T,d_h], expanded shared KV [B,H,T,d_h], output [B,T,d].',
    diagram: {
      nodes: [
        { x: 20, y: 150, w: 130, h: 82, label: 'X', shape: '[B,T,d]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 185, y: 70, w: 165, h: 82, label: 'Q proj', shape: '[B,H,T,d_h]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 185, y: 230, w: 165, h: 82, label: 'Shared KV proj', shape: '[B,1,T,d_h]', fill: '#ECFEFF', stroke: '#0891B2' },
        { x: 410, y: 230, w: 165, h: 82, label: 'Broadcast KV', shape: '[B,H,T,d_h]', fill: '#EDE9FE', stroke: '#7C3AED' },
        { x: 410, y: 70, w: 165, h: 82, label: 'Per-head queries', shape: '[B,H,T,d_h]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 630, y: 150, w: 135, h: 82, label: 'Attention', shape: '[B,H,T,d_h]', fill: '#FEF3C7', stroke: '#D97706' },
      ],
      edges: [
        { x1: 150, y1: 191, x2: 185, y2: 111 },
        { x1: 150, y1: 191, x2: 185, y2: 271 },
        { x1: 350, y1: 111, x2: 410, y2: 111 },
        { x1: 350, y1: 271, x2: 410, y2: 271 },
        { x1: 575, y1: 111, x2: 630, y2: 174 },
        { x1: 575, y1: 271, x2: 630, y2: 208 },
      ],
      caption: 'MQA minimizes memory bandwidth during decoding because only one KV stream is cached per token.'
    },
    code: `class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.h = num_heads
        self.d_h = d_model // num_heads
        self.wq = nn.Linear(d_model, num_heads * self.d_h)
        self.wk = nn.Linear(d_model, self.d_h)
        self.wv = nn.Linear(d_model, self.d_h)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.d_h).transpose(1, 2)
        k = self.wk(x).unsqueeze(1)                         # [B,1,T,d_h]
        v = self.wv(x).unsqueeze(1)
        y = scaled_dot_product_attention(q, k.expand(-1, self.h, -1, -1), v.expand(-1, self.h, -1, -1))
        return y`,
  },
  {
    title: 'Grouped-Query Attention (GQA)',
    summary: 'Uses many query heads but fewer K/V groups, reducing KV-cache memory and decode bandwidth while keeping more query diversity than MQA.',
    formula: String.raw`\operatorname{head}_h = \operatorname{Attn}(\mathbf{Q}_h, \mathbf{K}_{g(h)}, \mathbf{V}_{g(h)}),\qquad G < H` ,
    formulaNote: 'Query head h maps to one KV group g(h); each group is shared across several heads.',
    legend: 'Shapes: Q [B,H,T,d_h], K/V [B,G,T,d_h], repeated KV after sharing [B,H,T,d_h], output [B,T,d].',
    diagram: {
      nodes: [
        { x: 20, y: 150, w: 130, h: 82, label: 'X', shape: '[B,T,d]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 185, y: 70, w: 165, h: 82, label: 'Q proj', shape: '[B,H,T,d_h]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 185, y: 230, w: 165, h: 82, label: 'KV proj', shape: '[B,G,T,d_h]', fill: '#ECFEFF', stroke: '#0891B2' },
        { x: 410, y: 230, w: 165, h: 82, label: 'Share / repeat KV', shape: '[B,H,T,d_h]', fill: '#EDE9FE', stroke: '#7C3AED' },
        { x: 410, y: 70, w: 165, h: 82, label: 'Per-head queries', shape: '[B,H,T,d_h]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 630, y: 150, w: 135, h: 82, label: 'Attention', shape: '[B,H,T,d_h]', fill: '#FEF3C7', stroke: '#D97706' },
      ],
      edges: [
        { x1: 150, y1: 191, x2: 185, y2: 111 },
        { x1: 150, y1: 191, x2: 185, y2: 271 },
        { x1: 350, y1: 111, x2: 410, y2: 111 },
        { x1: 350, y1: 271, x2: 410, y2: 271 },
        { x1: 575, y1: 111, x2: 630, y2: 174 },
        { x1: 575, y1: 271, x2: 630, y2: 208 },
      ],
      caption: 'Compared with full MHA, GQA shares fewer KV streams across more query heads.'
    },
    code: `class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_groups):
        super().__init__()
        self.h = num_q_heads
        self.g = num_kv_groups
        self.d_h = d_model // num_q_heads
        self.wq = nn.Linear(d_model, self.h * self.d_h)
        self.wk = nn.Linear(d_model, self.g * self.d_h)
        self.wv = nn.Linear(d_model, self.g * self.d_h)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.d_h).transpose(1, 2)
        k = self.wk(x).view(B, T, self.g, self.d_h).transpose(1, 2)
        v = self.wv(x).view(B, T, self.g, self.d_h).transpose(1, 2)
        repeat = self.h // self.g
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        return scaled_dot_product_attention(q, k, v)`,
  },
  {
    title: 'Multi-Head Latent Attention (MLA)',
    summary: 'Compresses KV information into a lower-rank latent state, then reconstructs or expands head-specific K/V features from that compact cache.',
    formula: String.raw`\mathbf{c} = \mathbf{X}W_c,\qquad \mathbf{K}_h = \mathbf{c}W_{K,h},\quad \mathbf{V}_h = \mathbf{c}W_{V,h},\qquad \mathbf{c} \in \mathbb{R}^{B\times T\times r},\; r \ll d` ,
    formulaNote: 'The exact DeepSeek implementation is more detailed; this slide shows the core low-rank compression idea.',
    legend: 'Shapes: X [B,T,d], latent cache C [B,T,r], Q [B,H,T,d_h], reconstructed K/V [B,H,T,d_h or d_v].',
    diagram: {
      nodes: [
        { x: 20, y: 150, w: 120, h: 82, label: 'X', shape: '[B,T,d]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 170, y: 60, w: 150, h: 82, label: 'Q proj', shape: '[B,H,T,d_h]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 170, y: 240, w: 150, h: 82, label: 'Latent KV', shape: 'C: [B,T,r]', fill: '#ECFEFF', stroke: '#0891B2' },
        { x: 385, y: 200, w: 150, h: 82, label: 'Expand K/V', shape: '[B,H,T,d_h]', fill: '#EDE9FE', stroke: '#7C3AED' },
        { x: 385, y: 70, w: 150, h: 82, label: 'Queries', shape: '[B,H,T,d_h]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 590, y: 135, w: 145, h: 96, label: 'Attention', shape: '[B,H,T,d_v]', fill: '#FEF3C7', stroke: '#D97706' },
      ],
      edges: [
        { x1: 140, y1: 191, x2: 170, y2: 101 },
        { x1: 140, y1: 191, x2: 170, y2: 281 },
        { x1: 320, y1: 101, x2: 385, y2: 111 },
        { x1: 320, y1: 281, x2: 385, y2: 241 },
        { x1: 535, y1: 111, x2: 590, y2: 168 },
        { x1: 535, y1: 241, x2: 590, y2: 198 },
      ],
      caption: 'MLA saves memory by caching the latent C instead of a full per-head KV tensor.'
    },
    code: `class LatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, latent_rank):
        super().__init__()
        self.h = num_heads
        self.d_h = d_model // num_heads
        self.wq = nn.Linear(d_model, num_heads * self.d_h)
        self.wc = nn.Linear(d_model, latent_rank)
        self.wk = nn.Linear(latent_rank, num_heads * self.d_h)
        self.wv = nn.Linear(latent_rank, num_heads * self.d_h)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.d_h).transpose(1, 2)
        c = self.wc(x)                            # [B,T,r]
        k = self.wk(c).view(B, T, self.h, self.d_h).transpose(1, 2)
        v = self.wv(c).view(B, T, self.h, self.d_h).transpose(1, 2)
        return scaled_dot_product_attention(q, k, v)`,
  },
  {
    title: 'Sliding Window Attention (SWA)',
    summary: 'Restricts each token to attend only to a local window of nearby keys, improving long-context efficiency by sparsifying the attention pattern.',
    formula: String.raw`\operatorname{Attn}_i = \sum_{j=\max(0, i-w+1)}^{i} \alpha_{ij}\mathbf{V}_j,\qquad \alpha_{ij}=\operatorname{softmax}_{j\in\mathcal{W}(i)}\!\left(\frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_h}}\right)` ,
    formulaNote: 'The local window size w replaces full O(T²) connectivity with a banded pattern.',
    legend: 'Shapes: Q/K/V [B,H,T,d_h], local scores [B,H,T,w], output [B,H,T,d_h].',
    diagram: {
      nodes: [
        { x: 30, y: 150, w: 125, h: 82, label: 'Q', shape: '[B,H,T,d_h]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 200, y: 75, w: 150, h: 82, label: 'Local K window', shape: '[B,H,T,w,d_h]', fill: '#ECFEFF', stroke: '#0891B2' },
        { x: 200, y: 225, w: 150, h: 82, label: 'Local V window', shape: '[B,H,T,w,d_h]', fill: '#ECFEFF', stroke: '#0891B2' },
        { x: 415, y: 110, w: 150, h: 92, label: 'Band scores', shape: '[B,H,T,w]', fill: '#FEF3C7', stroke: '#D97706' },
        { x: 415, y: 235, w: 150, h: 82, label: 'Softmax', shape: '[B,H,T,w]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 625, y: 150, w: 140, h: 82, label: 'Output', shape: '[B,H,T,d_h]', fill: '#ECFDF5', stroke: '#059669' },
      ],
      edges: [
        { x1: 155, y1: 191, x2: 200, y2: 116 },
        { x1: 155, y1: 191, x2: 415, y2: 156 },
        { x1: 350, y1: 116, x2: 415, y2: 156 },
        { x1: 350, y1: 266, x2: 415, y2: 276 },
        { x1: 490, y1: 202, x2: 490, y2: 235 },
        { x1: 565, y1: 276, x2: 625, y2: 191 },
      ],
      caption: 'Only a local band of K/V around each token participates in the attention computation.'
    },
    code: `def sliding_window_attention(q, k, v, window):
    T = q.size(-2)
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5
    idx = torch.arange(T, device=q.device)
    too_old = idx[None, :] < (idx[:, None] - window + 1)
    causal = idx[None, :] > idx[:, None]
    mask = too_old | causal
    scores = scores.masked_fill(mask, float('-inf'))
    return scores.softmax(dim=-1) @ v`,
  },
  {
    title: 'KV Cache',
    summary: 'Stores past keys and values during autoregressive decoding so each new token only computes attention against the growing cache instead of recomputing old projections.',
    formula: String.raw`\mathbf{K}_{cache}^{(t)} = \operatorname{Concat}(\mathbf{K}_{cache}^{(t-1)}, \mathbf{k}_t),\qquad \mathbf{V}_{cache}^{(t)} = \operatorname{Concat}(\mathbf{V}_{cache}^{(t-1)}, \mathbf{v}_t)`,
    formulaNote: 'At decode step t, the query has sequence length 1 while the cached K/V length grows with t.',
    legend: 'Shapes: x_t [B,1,d], q_t [B,H,1,d_h], cached K/V [B,H,t,d_h], output y_t [B,H,1,d_h].',
    diagram: {
      nodes: [
        { x: 20, y: 150, w: 120, h: 82, label: 'x_t', shape: '[B,1,d]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 170, y: 60, w: 145, h: 82, label: 'q_t proj', shape: '[B,H,1,d_h]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 170, y: 240, w: 145, h: 82, label: 'k_t / v_t proj', shape: '[B,H,1,d_h]', fill: '#ECFEFF', stroke: '#0891B2' },
        { x: 375, y: 240, w: 165, h: 82, label: 'Append to cache', shape: '[B,H,t,d_h]', fill: '#EDE9FE', stroke: '#7C3AED' },
        { x: 590, y: 150, w: 160, h: 92, label: 'Decode attention', shape: '[B,H,1,d_h]', fill: '#FEF3C7', stroke: '#D97706' },
      ],
      edges: [
        { x1: 140, y1: 191, x2: 170, y2: 101 },
        { x1: 140, y1: 191, x2: 170, y2: 281 },
        { x1: 315, y1: 281, x2: 375, y2: 281 },
        { x1: 315, y1: 101, x2: 590, y2: 180 },
        { x1: 540, y1: 281, x2: 590, y2: 212 },
      ],
      caption: 'The KV cache is an inference-time optimization: it trades memory for much faster decode steps.'
    },
    code: `def decode_step(x_t, k_cache, v_cache, wq, wk, wv):
    q_t = wq(x_t)                                  # [B,H,1,d_h]
    k_t = wk(x_t)
    v_t = wv(x_t)
    k_cache = torch.cat([k_cache, k_t], dim=-2)    # [B,H,t,d_h]
    v_cache = torch.cat([v_cache, v_t], dim=-2)
    y_t = scaled_dot_product_attention(q_t, k_cache, v_cache)
    return y_t, k_cache, v_cache`,
  },
  {
    title: 'QK-Norm',
    summary: 'Normalizes query and key vectors before the dot product, controlling score magnitude and often improving optimization stability.',
    formula: String.raw`\widehat{\mathbf{Q}} = \frac{\mathbf{Q}}{\lVert \mathbf{Q} \rVert_2},\qquad \widehat{\mathbf{K}} = \frac{\mathbf{K}}{\lVert \mathbf{K} \rVert_2},\qquad \operatorname{Attn} = \operatorname{softmax}(s\,\widehat{\mathbf{Q}}\widehat{\mathbf{K}}^\top)\mathbf{V}` ,
    formulaNote: 'A learned scalar s or per-head scale often replaces the usual 1/√d_h factor.',
    legend: 'Shapes: Q,K [B,H,T,d_h], normalized Q/K same shape, scores [B,H,T,T], output [B,H,T,d_v].',
    diagram: {
      nodes: [
        { x: 25, y: 85, w: 130, h: 82, label: 'Q', shape: '[B,H,T,d_h]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 25, y: 235, w: 130, h: 82, label: 'K', shape: '[B,H,T,d_h]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 210, y: 85, w: 150, h: 82, label: 'L2 normalize', shape: '[B,H,T,d_h]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 210, y: 235, w: 150, h: 82, label: 'L2 normalize', shape: '[B,H,T,d_h]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 430, y: 150, w: 150, h: 92, label: 'Scale · QKᵀ', shape: '[B,H,T,T]', fill: '#FEF3C7', stroke: '#D97706' },
        { x: 645, y: 150, w: 120, h: 92, label: 'Attn', shape: '[B,H,T,d_v]', fill: '#ECFDF5', stroke: '#059669' },
      ],
      edges: [
        { x1: 155, y1: 126, x2: 210, y2: 126 },
        { x1: 155, y1: 276, x2: 210, y2: 276 },
        { x1: 360, y1: 126, x2: 430, y2: 178 },
        { x1: 360, y1: 276, x2: 430, y2: 214 },
        { x1: 580, y1: 196, x2: 645, y2: 196 },
      ],
      caption: 'Normalizing Q and K constrains dot-product magnitude before softmax.'
    },
    code: `def qk_norm_attention(q, k, v, scale):
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    scores = scale * (q @ k.transpose(-2, -1))
    weights = scores.softmax(dim=-1)
    return weights @ v`,
  },
  {
    title: 'Mixture-of-Experts (MoE)',
    summary: 'Routes each token to a small subset of experts so the model gets high total capacity while activating only a few feed-forward blocks per token.',
    formula: String.raw`\mathbf{y} = \sum_{e \in \operatorname{TopK}(g(\mathbf{x}))} g_e(\mathbf{x})\,\operatorname{Expert}_e(\mathbf{x})` ,
    formulaNote: 'A shared expert path is sometimes added on top of the routed experts in modern sparse LLMs.',
    legend: 'Shapes: x [B,T,d], router logits [B,T,E], top-k weights [B,T,k], expert outputs [B,T,d], final y [B,T,d].',
    diagram: {
      nodes: [
        { x: 20, y: 150, w: 120, h: 82, label: 'x', shape: '[B,T,d]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 170, y: 70, w: 160, h: 82, label: 'Router', shape: '[B,T,E]', fill: '#EDE9FE', stroke: '#7C3AED' },
        { x: 170, y: 240, w: 160, h: 82, label: 'Top-k weights', shape: '[B,T,k]', fill: '#FEF3C7', stroke: '#D97706' },
        { x: 390, y: 35, w: 150, h: 72, label: 'Expert 1', shape: '[B,T,d]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 390, y: 125, w: 150, h: 72, label: 'Expert 2', shape: '[B,T,d]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 390, y: 215, w: 150, h: 72, label: 'Expert k', shape: '[B,T,d]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 590, y: 150, w: 170, h: 92, label: 'Weighted sum', shape: '[B,T,d]', fill: '#ECFDF5', stroke: '#059669' },
      ],
      edges: [
        { x1: 140, y1: 191, x2: 170, y2: 111 },
        { x1: 140, y1: 191, x2: 170, y2: 281 },
        { x1: 330, y1: 111, x2: 390, y2: 71 },
        { x1: 330, y1: 111, x2: 390, y2: 161 },
        { x1: 330, y1: 111, x2: 390, y2: 251 },
        { x1: 330, y1: 281, x2: 590, y2: 196 },
        { x1: 540, y1: 71, x2: 590, y2: 171 },
        { x1: 540, y1: 161, x2: 590, y2: 188 },
        { x1: 540, y1: 251, x2: 590, y2: 205 },
      ],
      caption: 'Routing is sparse: each token activates only a few experts rather than all of them.'
    },
    code: `class SimpleMoE(nn.Module):
    def __init__(self, d_model, d_hidden, num_experts, k=2):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_model))
            for _ in range(num_experts)
        ])
        self.k = k

    def forward(self, x):
        gates = self.router(x).softmax(dim=-1)
        topw, topi = gates.topk(self.k, dim=-1)
        out = 0
        for weight, idx in zip(topw.unbind(-1), topi.unbind(-1)):
            expert_out = torch.stack([self.experts[j](x[n]) for n, j in enumerate(idx.view(-1))])
            out = out + weight.unsqueeze(-1) * expert_out.view_as(x)
        return out`,
  },
  {
    title: 'GELU',
    summary: 'A smooth nonlinearity that keeps small negative values instead of hard-clipping them like ReLU.',
    formula: String.raw`\operatorname{GELU}(x)=x\,\Phi(x)\approx 0.5x\left(1+\tanh\!\left(\sqrt{\frac{2}{\pi}}(x+0.044715x^3)\right)\right)` ,
    formulaNote: 'Widely used in feed-forward blocks of GPT/BERT-style Transformers.',
    legend: 'Shapes: pointwise activation on x and y with identical shape, typically [*, d_ff].',
    diagram: {
      nodes: [
        { x: 70, y: 150, w: 145, h: 84, label: 'Pre-activation', shape: '[*, d_ff]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 315, y: 150, w: 170, h: 84, label: 'GELU', shape: 'pointwise', fill: '#FEF3C7', stroke: '#D97706' },
        { x: 585, y: 150, w: 145, h: 84, label: 'Post-activation', shape: '[*, d_ff]', fill: '#ECFDF5', stroke: '#059669' },
      ],
      edges: [
        { x1: 215, y1: 192, x2: 315, y2: 192 },
        { x1: 485, y1: 192, x2: 585, y2: 192 },
      ],
      caption: 'GELU is a smooth gate: larger positive values pass through more strongly.'
    },
    code: `class GELUActivation(nn.Module):
    def forward(self, x):
        return F.gelu(x)

# or explicitly
# y = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))`,
  },
  {
    title: 'SiLU / Swish',
    summary: 'Multiplies each value by its sigmoid gate, giving a smooth self-gating activation widely used in modern gated MLPs.',
    formula: String.raw`\operatorname{SiLU}(x) = x\,\sigma(x)` ,
    formulaNote: 'SiLU often appears inside gated feed-forward blocks such as SwiGLU variants.',
    legend: 'Shapes: pointwise activation on x and y with identical shape, typically [*, d_ff].',
    diagram: {
      nodes: [
        { x: 55, y: 150, w: 145, h: 84, label: 'x', shape: '[*, d_ff]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 270, y: 70, w: 150, h: 84, label: 'Sigmoid(x)', shape: '[*, d_ff]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 270, y: 230, w: 150, h: 84, label: 'Identity x', shape: '[*, d_ff]', fill: '#ECFEFF', stroke: '#0891B2' },
        { x: 500, y: 150, w: 120, h: 84, label: 'Multiply', shape: 'pointwise', fill: '#FEF3C7', stroke: '#D97706' },
        { x: 670, y: 150, w: 90, h: 84, label: 'y', shape: '[*, d_ff]', fill: '#ECFDF5', stroke: '#059669' },
      ],
      edges: [
        { x1: 200, y1: 192, x2: 270, y2: 111 },
        { x1: 200, y1: 192, x2: 270, y2: 271 },
        { x1: 420, y1: 111, x2: 500, y2: 174 },
        { x1: 420, y1: 271, x2: 500, y2: 210 },
        { x1: 620, y1: 192, x2: 670, y2: 192 },
      ],
      caption: 'SiLU is a self-gating activation: the sigmoid controls how much of x is passed through.'
    },
    code: `class SiLUActivation(nn.Module):
    def forward(self, x):
        return F.silu(x)

# equivalent:
# y = x * torch.sigmoid(x)`,
  },
  {
    title: 'SwiGLU',
    summary: 'A gated feed-forward block that combines a SiLU gate branch with a value branch before projecting back to the model dimension.',
    formula: String.raw`\operatorname{SwiGLU}(\mathbf{x}) = \left(\operatorname{SiLU}(\mathbf{x}W_g)\odot(\mathbf{x}W_u)\right)W_d`,
    formulaNote: 'SwiGLU is commonly used in modern Transformer MLPs because the gate improves expressivity at similar compute cost.',
    legend: 'Shapes: x [B,T,d], gate/up [B,T,d_{ff}], gated product [B,T,d_{ff}], output [B,T,d].',
    diagram: {
      nodes: [
        { x: 30, y: 150, w: 120, h: 82, label: 'x', shape: '[B,T,d]', fill: '#E8F0FE', stroke: '#2563EB' },
        { x: 190, y: 65, w: 150, h: 82, label: 'Gate proj', shape: '[B,T,d_ff]', fill: '#EEF2FF', stroke: '#6366F1' },
        { x: 190, y: 150, w: 150, h: 82, label: 'SiLU', shape: '[B,T,d_ff]', fill: '#FEF3C7', stroke: '#D97706' },
        { x: 190, y: 235, w: 150, h: 82, label: 'Up proj', shape: '[B,T,d_ff]', fill: '#ECFEFF', stroke: '#0891B2' },
        { x: 400, y: 150, w: 135, h: 82, label: 'Multiply', shape: '[B,T,d_ff]', fill: '#EDE9FE', stroke: '#7C3AED' },
        { x: 600, y: 150, w: 150, h: 82, label: 'Down proj', shape: '[B,T,d]', fill: '#ECFDF5', stroke: '#059669' },
      ],
      edges: [
        { x1: 150, y1: 191, x2: 190, y2: 106 },
        { x1: 150, y1: 191, x2: 190, y2: 276 },
        { x1: 340, y1: 106, x2: 340, y2: 150 },
        { x1: 340, y1: 191, x2: 400, y2: 176 },
        { x1: 340, y1: 276, x2: 400, y2: 206 },
        { x1: 535, y1: 191, x2: 600, y2: 191 },
      ],
      caption: 'SwiGLU replaces a plain activation with a learned gate/value interaction in the feed-forward block.'
    },
    code: `class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.wg = nn.Linear(d_model, d_ff, bias=False)
        self.wu = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.wg(x))
        up = self.wu(x)
        return self.wo(gate * up)`,
  },
];

addTitleSlide();
components.forEach(addComponentSlide);

const outPptx = path.join(__dirname, 'LLMs.pptx');
const outJs = path.join(__dirname, 'LLMs.js');

fs.copyFileSync(__filename, outJs);
pptx.writeFile({ fileName: outPptx }).then(() => {
  const finalPptx = path.resolve(__dirname, '..', 'LLMs.pptx');
  const finalJs = path.resolve(__dirname, '..', 'LLMs.js');
  fs.copyFileSync(outPptx, finalPptx);
  fs.copyFileSync(outJs, finalJs);
  console.log(`Wrote ${finalPptx}`);
  console.log(`Wrote ${finalJs}`);
}).catch((err) => {
  console.error(err);
  process.exit(1);
});
