const PptxGenJS = require('pptxgenjs');
const { svgToDataUri, latexToSvgDataUri } = require('./pptxgenjs_helpers');
const pptx = new PptxGenJS();
pptx.layout = 'LAYOUT_WIDE';
const slide = pptx.addSlide();
slide.addText('Test', {x:0.5,y:0.3,w:2,h:0.3});
slide.addImage({ data: svgToDataUri('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100"><rect x="0" y="0" width="200" height="100" fill="#eef" stroke="#00f"/><text x="100" y="55" text-anchor="middle" font-size="20">SVG</text></svg>'), x:0.5,y:1,w:4,h:2 });
slide.addImage({ data: latexToSvgDataUri(String.raw`x = \frac{a}{b}`), x:5,y:1,w:3,h:1 });
pptx.writeFile({ fileName: 'lecture_notes/.llms_deck_work/test_image.pptx' });
