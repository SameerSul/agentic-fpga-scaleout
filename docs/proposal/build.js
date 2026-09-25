const pptx = require('pptxgenjs');
const p = new pptx();
p.layout = 'LAYOUT_WIDE';            // 13.33 x 7.5
p.author = 'Group 9'; p.title = 'fpgAI';

const INK='13161A', PAPER='F5F3EF', CARD='FFFFFF', COPPER='C2410C',
      TEAL='0F766E', MUTE='6A6F78', LINE='DCD8D0', PALE='F0E4DA';
const H='Calibri', M='Courier New';
const L=0.62, W=12.09;                // left margin, content width

// Repeating motif: a small copper chip square carrying the slide number.
function chip(s, n, y){
  s.addShape(p.ShapeType.roundRect, {x:L, y:y||0.42, w:0.36, h:0.36, fill:{color:COPPER},
    rectRadius:0.06, line:{color:COPPER}});
  s.addText(String(n), {x:L, y:y||0.42, w:0.36, h:0.36, align:'center', valign:'middle',
    fontSize:13, bold:true, color:'FFFFFF', fontFace:H, isTextBox:true, margin:0});
}
function title(s, n, t, sub){
  chip(s, n);
  s.addText(t, {x:L+0.52, y:0.34, w:W-0.52, h:0.56, fontSize:32, bold:true, color:INK,
    fontFace:H, isTextBox:true, margin:0, valign:'middle'});
  if(sub) s.addText(sub, {x:L+0.52, y:0.92, w:W-0.52, h:0.36, fontSize:14, color:MUTE,
    fontFace:H, isTextBox:true, margin:0});
}
function light(){ const s=p.addSlide(); s.background={color:PAPER}; return s; }
function dark(){ const s=p.addSlide(); s.background={color:INK}; return s; }
function card(s,x,y,w,h,fill){ s.addShape(p.ShapeType.roundRect,{x,y,w,h,fill:{color:fill||CARD},
  rectRadius:0.08, line:{color:LINE, width:0.75}}); }

/* 1 — title */
let s = dark();
s.addText('fpgAI', {x:L, y:2.05, w:W, h:1.15, fontSize:66, bold:true, color:'FFFFFF',
  fontFace:H, isTextBox:true, margin:0});
s.addText('Agent-designed FPGA scaleout for local LLM inference',
  {x:L, y:3.18, w:W, h:0.6, fontSize:23, color:PALE, fontFace:H, isTextBox:true, margin:0});
s.addShape(p.ShapeType.roundRect,{x:L,y:4.05,w:1.7,h:0.05,fill:{color:COPPER},rectRadius:0.02,line:{color:COPPER}});
s.addText('Group 9   ·   Sameer Suleman, Yax Patel, Vaibhav Gopalakrishnan, Siddh Patel',
  {x:L, y:4.45, w:W, h:0.34, fontSize:15, color:'FFFFFF', fontFace:H, isTextBox:true, margin:0});
s.addText('Supervisor: Dr. Shirani', {x:L, y:4.83, w:W, h:0.34, fontSize:15, color:MUTE,
  fontFace:H, isTextBox:true, margin:0});
s.addNotes('SAMEER. Two sentences: we make software agents design the chip that runs a given language model, and we make it scale across cheap boards. Then move on.');

/* 2 — the problem */
s = light();
title(s, 1, 'Two walls, one project');
const walls=[['Designing the chip','A chip is specified years before the models it will run. Closing the loop takes teams of specialists months of tool iteration.'],
             ['Affording the chip','A board a student can buy holds a fraction of a useful model. One that fits costs more than a research group’s whole budget.']];
walls.forEach((c,i)=>{ const x=L+i*6.2;
  card(s,x,1.7,5.85,2.45);
  s.addText(c[0],{x:x+0.35,y:1.95,w:5.15,h:0.4,fontSize:21,bold:true,color:COPPER,fontFace:H,isTextBox:true,margin:0});
  s.addText(c[1],{x:x+0.35,y:2.42,w:5.15,h:1.5,fontSize:15,color:INK,fontFace:H,isTextBox:true,margin:0});
});
s.addText('Cloud inference solves neither, and sends your data off-premises. Medical, legal and industrial users cannot use it at all.',
  {x:L,y:4.55,w:W,h:0.5,fontSize:17,italic:true,color:MUTE,fontFace:H,isTextBox:true,margin:0});
s.addNotes('SAMEER. Both walls are real and they compound. The second one is why students cannot participate in this field at all.');

/* 3 — the gap */
s = light();
title(s, 2, 'What Redwood proved, and what it left open',
  'Architect Labs, 2026. An AI system designed a working accelerator in under two weeks.');
card(s,L,1.85,5.85,2.9);
s.addText('Proved',{x:L+0.35,y:2.08,w:5.15,h:0.36,fontSize:19,bold:true,color:TEAL,fontFace:H,isTextBox:true,margin:0});
s.addText([{text:'Agents wrote RTL, UVM, firmware and drivers',options:{bullet:true,breakLine:true}},
           {text:'Over 95% functional coverage',options:{bullet:true,breakLine:true}},
           {text:'Ran Qwen3-0.6B on a real FPGA',options:{bullet:true}}],
  {x:L+0.35,y:2.52,w:5.15,h:1.9,fontSize:15,color:INK,fontFace:H,isTextBox:true,paraSpaceAfter:7,margin:0});
card(s,L+6.24,1.85,5.85,2.9,PALE);
s.addText('Left open',{x:L+6.59,y:2.08,w:5.15,h:0.36,fontSize:19,bold:true,color:COPPER,fontFace:H,isTextBox:true,margin:0});
s.addText([{text:'Architecture written by two human architects, not derived from the model',options:{bullet:true,breakLine:true}},
           {text:'One chip. No inter-chip scaleout at all',options:{bullet:true,breakLine:true}},
           {text:'12.1 tok/s, slower than a $250 Jetson',options:{bullet:true}}],
  {x:L+6.59,y:2.52,w:5.15,h:1.9,fontSize:15,color:INK,fontFace:H,isTextBox:true,paraSpaceAfter:7,margin:0});
s.addText('Those two gaps are our project.',{x:L,y:5.0,w:W,h:0.45,fontSize:20,bold:true,color:INK,fontFace:H,isTextBox:true,margin:0});
s.addNotes('SAMEER. Be generous about Redwood, it is good work. Their result was about the design process, not speed. The two gaps are exactly what we do.');

/* 4 — the loop */
s = light();
title(s, 3, 'The system', 'A model specification goes in. Working hardware comes out.');
const steps=[['Model spec','Qwen3-0.6B, int8'],['Derive','datapath and\nfabric widths'],['LLM writes RTL','compute and\nlink endpoint'],
             ['Five tool gates','sim, synth, timing,\nmap, place-and-route'],['Bitstream','runs on the board']];
steps.forEach((t,i)=>{ const x=L+i*2.44;
  card(s,x,2.0,2.2,1.75, i===2?PALE:CARD);
  s.addText(t[0],{x:x+0.16,y:2.18,w:1.88,h:0.5,fontSize:15,bold:true,color:i===2?COPPER:INK,fontFace:H,isTextBox:true,margin:0,align:'center'});
  s.addText(t[1],{x:x+0.16,y:2.68,w:1.88,h:0.95,fontSize:11.5,color:MUTE,fontFace:H,isTextBox:true,margin:0,align:'center'});
  if(i<4) s.addText('→',{x:x+2.18,y:2.6,w:0.3,h:0.5,fontSize:19,color:COPPER,fontFace:H,isTextBox:true,margin:0,align:'center'});
});
s.addShape(p.ShapeType.roundRect,{x:L+1.1,y:4.1,w:9.7,h:0.42,fill:{color:PALE},rectRadius:0.1,line:{color:PALE}});
s.addText('Any tool failure is parsed and fed back to the agent, which revises and retries',
  {x:L+1.1,y:4.1,w:9.7,h:0.42,fontSize:13.5,italic:true,color:COPPER,fontFace:H,isTextBox:true,margin:0,align:'center',valign:'middle'});
s.addText('A sizing layer then says how many boards the model needs, and checks that answer packet by packet before anything is bought.',
  {x:L,y:4.85,w:W,h:0.5,fontSize:16,color:INK,fontFace:H,isTextBox:true,margin:0});
s.addNotes('SAMEER. Walk left to right once. Emphasise the feedback arrow: that is the agentic part. Hand to Yax.');

/* 5 — derivation */
s = light();
title(s, 4, 'The hardware is derived, not guessed');
s.addText('The model tells us the arithmetic it needs. We compute the datapath from it.',
  {x:L,y:1.5,w:W,h:0.4,fontSize:16,color:MUTE,fontFace:H,isTextBox:true,margin:0});
card(s,L,2.05,7.4,2.35,CARD);
s.addText('accumulator = weight bits + activation bits + ceil(log2(deepest reduction))',
  {x:L+0.3,y:2.3,w:6.8,h:0.62,fontSize:14,bold:true,color:COPPER,fontFace:M,isTextBox:true,margin:0});
s.addText('For Qwen3-0.6B that is 8 + 8 + 12 = 28 bits. Wide enough that overflow is mathematically impossible, and not one bit wider. Re-quantize the model to 4 bits and the pipeline emits a smaller design automatically.',
  {x:L+0.3,y:3.0,w:6.8,h:1.2,fontSize:14.5,color:INK,fontFace:H,isTextBox:true,margin:0});
card(s,L+7.75,2.05,4.34,2.35,PALE);
s.addText('The endpoint too',{x:L+8.05,y:2.28,w:3.74,h:0.36,fontSize:16,bold:true,color:COPPER,fontFace:H,isTextBox:true,margin:0});
s.addText('The link width is derived from the wire speed. When the obvious datapath missed timing by 0.02 ns, the flow widened it, halved the clock, reran the whole loop and signed off with 1.1 ns spare.',
  {x:L+8.05,y:2.72,w:3.74,h:1.5,fontSize:13.5,color:INK,fontFace:H,isTextBox:true,margin:0});
s.addText('That last one matters: the system changed what it was building, not just how it was written.',
  {x:L,y:4.62,w:W,h:0.45,fontSize:17,bold:true,color:INK,fontFace:H,isTextBox:true,margin:0});
s.addNotes('YAX. The formula is the whole idea in one line. The endpoint story is the moment the system made an architecture decision on its own.');

/* 6 — agent fixing itself */
s = light();
title(s, 5, 'The loop catching its own mistake', 'Real output from the compute block. Three iterations, 2.7 seconds.');
const rows=[['1','none','FAIL  wide_product: expected 65328, got 304','13161A'],
            ['2','widen product register','FAIL  sync_clear: expected 0, got 129845','13161A'],
            ['3','+ implement sync clear','PASS  594 checks, 1052 cells, +4.44 ns slack','0F766E']];
s.addText([{text:'iter',options:{bold:true}}],{x:L+0.25,y:1.95,w:0.5,h:0.3,fontSize:11.5,color:MUTE,fontFace:H,isTextBox:true,margin:0});
s.addText([{text:'agent applied',options:{bold:true}}],{x:L+0.95,y:1.95,w:3.0,h:0.3,fontSize:11.5,color:MUTE,fontFace:H,isTextBox:true,margin:0});
s.addText([{text:'tool result',options:{bold:true}}],{x:L+4.2,y:1.95,w:7.4,h:0.3,fontSize:11.5,color:MUTE,fontFace:H,isTextBox:true,margin:0});
rows.forEach((r,i)=>{ const y=2.35+i*0.72;
  card(s,L,y,W,0.62, i===2?'E8F0EC':CARD);
  s.addText(r[0],{x:L+0.25,y:y,w:0.5,h:0.62,fontSize:14,bold:true,color:COPPER,fontFace:M,isTextBox:true,margin:0,valign:'middle'});
  s.addText(r[1],{x:L+0.95,y:y,w:3.1,h:0.62,fontSize:13,color:INK,fontFace:H,isTextBox:true,margin:0,valign:'middle'});
  s.addText(r[2],{x:L+4.2,y:y,w:7.5,h:0.62,fontSize:12.5,color:r[3],fontFace:M,isTextBox:true,margin:0,valign:'middle'});
});
s.addText('Nobody told it what was wrong. It read the testbench failure and revised the design.',
  {x:L,y:4.72,w:W,h:0.45,fontSize:17,bold:true,color:INK,fontFace:H,isTextBox:true,margin:0});
s.addNotes('YAX. This is the single most convincing slide. The agent shipped a too-narrow register, the testbench caught it, the agent widened it, then hit the next bug. Claude Haiku has also done both blocks first try.');

/* 7 — engineering finding, native chart */
s = light();
title(s, 6, 'We size the cluster before buying anything',
  'Qwen3-0.6B on two boards. How much Ethernet do we actually need?');
s.addChart(p.ChartType.bar, [{name:'Tokens per second', labels:['100 Mbps','1 Gbps','2.5 Gbps','10 Gbps','25 Gbps'],
  values:[69,342,464,565,570]}],
  {x:L, y:1.85, w:7.0, h:3.2, barDir:'col', chartColors:[COPPER],
   showTitle:false, showLegend:false, showValue:true, dataLabelPosition:'outEnd',
   dataLabelFontSize:11, dataLabelColor:INK, dataLabelFontFace:H,
   catAxisLabelColor:MUTE, valAxisLabelColor:MUTE, catAxisLabelFontSize:11, valAxisLabelFontSize:10,
   catAxisLabelFontFace:H, valAxisLabelFontFace:H,
   valGridLine:{color:LINE, size:0.75}, catGridLine:{style:'none'}, valAxisMaxVal:650});
card(s,L+7.4,1.85,4.69,3.2,PALE);
s.addText('What this settles',{x:L+7.7,y:2.08,w:4.1,h:0.36,fontSize:17,bold:true,color:COPPER,fontFace:H,isTextBox:true,margin:0});
s.addText([{text:'1 Gbps gets 60% of what 10 Gbps does',options:{bullet:true,breakLine:true}},
           {text:'Past 10 Gbps buys nothing: the limit becomes memory, not the wire',options:{bullet:true,breakLine:true}},
           {text:'100 Mbps is genuinely too slow',options:{bullet:true,breakLine:true}},
           {text:'So we buy 1 Gbps and spend the difference on a second board',options:{bullet:true}}],
  {x:L+7.7,y:2.52,w:4.1,h:2.4,fontSize:13.5,color:INK,fontFace:H,isTextBox:true,paraSpaceAfter:7,margin:0});
s.addNotes('YAX. The point is not the number, it is that we can answer hardware questions before spending money. This killed a week of shopping for 10 gigabit boards.');

/* 8 — the comparison */
s = dark();
s.addText('Same model. Same result.', {x:L, y:0.72, w:W, h:0.62, fontSize:34, bold:true,
  color:'FFFFFF', fontFace:H, isTextBox:true, margin:0});
s.addText('Qwen3-0.6B, tokens per second', {x:L, y:1.35, w:W, h:0.38, fontSize:16, color:MUTE,
  fontFace:H, isTextBox:true, margin:0});
const cmp=[['Redwood','AMD Versal VPK180','$17,995','12.1','measured on silicon',MUTE],
           ['fpgAI','Two FPGA boards','~$600','43.3','predicted by our model',COPPER]];
cmp.forEach((c,i)=>{ const x=L+i*6.2;
  s.addShape(p.ShapeType.roundRect,{x:x,y:2.0,w:5.85,h:2.9,fill:{color: i?'241A14':'1C2024'},
    rectRadius:0.1, line:{color: i?COPPER:'2E343A', width: i?1.5:1}});
  s.addText(c[0],{x:x+0.4,y:2.22,w:5.05,h:0.42,fontSize:20,bold:true,color: i?COPPER:'FFFFFF',fontFace:H,isTextBox:true,margin:0});
  s.addText(c[1],{x:x+0.4,y:2.64,w:5.05,h:0.34,fontSize:13.5,color:MUTE,fontFace:H,isTextBox:true,margin:0});
  s.addText(c[2],{x:x+0.4,y:3.02,w:5.05,h:0.44,fontSize:19,bold:true,color:'FFFFFF',fontFace:H,isTextBox:true,margin:0});
  s.addText(c[3],{x:x+0.4,y:3.52,w:5.05,h:0.95,fontSize:56,bold:true,color: i?COPPER:'FFFFFF',fontFace:H,isTextBox:true,margin:0});
  s.addText(c[4],{x:x+2.3,y:3.95,w:3.1,h:0.4,fontSize:12,italic:true,color:MUTE,fontFace:H,isTextBox:true,margin:0});
});
s.addText('Their budget bought headroom they never used. It did not buy throughput, and their own paper says so.',
  {x:L,y:5.2,w:W,h:0.45,fontSize:16,color:PALE,fontFace:H,isTextBox:true,margin:0});
s.addNotes('VAIBHAV. Say clearly that ours is a prediction and theirs is silicon. The honest claim is the cost ratio, not that we beat them. Their design was a deliberately scaled-down tile.');

/* 9 — modules */
s = light();
title(s, 7, 'Four modules, four owners, all in parallel',
  'Owners are accountable, not exclusive. Work that crosses a boundary is done jointly.');
const mods=[['A','Agentic generation and signoff','Sameer','Derives the spec, runs the agents, owns the sizing model. Pure software.'],
            ['B','Fabric and collectives','Yax','Framing, checksums, flow control, ring all-reduce, and the multi-agent architecture.'],
            ['C','Physical implementation','Vaibhav','Vivado, constraints, place-and-route, bitstreams, board bring-up.'],
            ['D','Verification and the link','Siddh','Coverage, formal proofs, and the physical board-to-board link.']];
mods.forEach((m,i)=>{ const x=L+(i%2)*6.2, y=1.85+Math.floor(i/2)*1.72;
  card(s,x,y,5.85,1.5);
  s.addShape(p.ShapeType.roundRect,{x:x+0.28,y:y+0.26,w:0.42,h:0.42,fill:{color:COPPER},rectRadius:0.07,line:{color:COPPER}});
  s.addText(m[0],{x:x+0.28,y:y+0.26,w:0.42,h:0.42,fontSize:15,bold:true,color:'FFFFFF',fontFace:H,isTextBox:true,margin:0,align:'center',valign:'middle'});
  s.addText(m[1],{x:x+0.85,y:y+0.22,w:3.5,h:0.36,fontSize:16,bold:true,color:INK,fontFace:H,isTextBox:true,margin:0});
  s.addText(m[2],{x:x+4.3,y:y+0.22,w:1.3,h:0.36,fontSize:14,bold:true,color:TEAL,fontFace:H,isTextBox:true,margin:0,align:'right'});
  s.addText(m[3],{x:x+0.85,y:y+0.64,w:4.7,h:0.72,fontSize:13,color:MUTE,fontFace:H,isTextBox:true,margin:0});
});
s.addText('The interfaces between them are files that already exist, so each module is built against a stub of its neighbours.',
  {x:L,y:5.4,w:W,h:0.42,fontSize:15,italic:true,color:MUTE,fontFace:H,isTextBox:true,margin:0});
s.addNotes('VAIBHAV. Stress that nobody is idle waiting for hardware. Two of the four modules are pure software.');

/* 10 — deliverables */
s = light();
title(s, 8, 'What we are committing to', 'Bronze is the floor. Gold is cut first if the schedule slips.');
const tiers=[['Bronze','minimum viable product',['Agents sign off both blocks through all five gates','A bitstream from agent-written RTL runs on one board','Simulated cluster within 15% of prediction','100+ automated tests'],'Needs no purchase',COPPER],
             ['Silver','core result',['Two boards linked over the generated fabric','Ring all-reduce bit-exact across both','Qwen3-0.6B at 40+ tok/s measured','A block fixed by an LLM from tool output'],'Second board already owned',TEAL],
             ['Gold','stretch',['Larger models by adding boards','Agent writes the verification too','Interconnect study including photonics','Optionally, silicon on a shuttle'],'Dropped first',MUTE]];
tiers.forEach((t,i)=>{ const x=L+i*4.09;
  card(s,x,1.85,3.85,3.3, i===0?PALE:CARD);
  s.addText(t[0],{x:x+0.3,y:2.05,w:3.25,h:0.4,fontSize:21,bold:true,color:t[5],fontFace:H,isTextBox:true,margin:0});
  s.addText(t[1],{x:x+0.3,y:2.44,w:3.25,h:0.3,fontSize:12.5,italic:true,color:MUTE,fontFace:H,isTextBox:true,margin:0});
  s.addText(t[2].map((b,j)=>({text:b,options:{bullet:true,breakLine:j<t[2].length-1}})),
    {x:x+0.3,y:2.8,w:3.25,h:1.85,fontSize:12,color:INK,fontFace:H,isTextBox:true,paraSpaceAfter:5,margin:0});
  s.addText(t[4],{x:x+0.3,y:4.72,w:3.25,h:0.3,fontSize:11.5,bold:true,color:t[5],fontFace:H,isTextBox:true,margin:0});
});
s.addText('Bronze deliberately depends on neither a hardware purchase nor the unproven LLM iteration, which are our two riskiest items.',
  {x:L,y:5.42,w:W,h:0.42,fontSize:15,italic:true,color:MUTE,fontFace:H,isTextBox:true,margin:0});
s.addNotes('SIDDH. The last line is the answer to "what if it does not work". Say it out loud, do not let them ask it.');

/* 11 — schedule */
s = light();
title(s, 9, 'Schedule', '28 weeks, September to April. Four lanes run concurrently throughout.');
const lanes=[['A  Sameer',0.0,0.93,COPPER],['B  Yax',0.03,0.93,TEAL],['C  Vaibhav',0.0,0.79,COPPER],['D  Siddh',0.0,0.93,TEAL]];
const tx=L+1.75, tw=10.0;
['Sep','Nov','Jan','Mar','Apr'].forEach((mo,i)=>{
  s.addText(mo,{x:tx+i*(tw/4.6),y:1.72,w:1.0,h:0.28,fontSize:11.5,color:MUTE,fontFace:H,isTextBox:true,margin:0});
});
lanes.forEach((ln,i)=>{ const y=2.15+i*0.62;
  s.addText(ln[0],{x:L,y:y,w:1.65,h:0.38,fontSize:13,bold:true,color:INK,fontFace:H,isTextBox:true,margin:0,valign:'middle'});
  s.addShape(p.ShapeType.roundRect,{x:tx,y:y+0.06,w:tw,h:0.26,fill:{color:'E6E2DA'},rectRadius:0.05,line:{color:'E6E2DA'}});
  s.addShape(p.ShapeType.roundRect,{x:tx+ln[1]*tw,y:y+0.06,w:ln[2]*tw,h:0.26,fill:{color:ln[3]},rectRadius:0.05,line:{color:ln[3]}});
});
[['Bronze',0.27],['Silver',0.66],['Gold',0.91]].forEach(m=>{
  const x=tx+m[1]*tw;
  s.addShape(p.ShapeType.diamond,{x:x-0.11,y:4.72,w:0.22,h:0.22,fill:{color:INK},line:{color:INK}});
  s.addText(m[0],{x:x-0.55,y:4.97,w:1.1,h:0.28,fontSize:11.5,bold:true,color:INK,fontFace:H,isTextBox:true,margin:0,align:'center'});
});
s.addText('Bronze at week 8, Silver at 19, Gold at 26, with five weeks of float at the end for hardware delays.',
  {x:L,y:5.52,w:W,h:0.42,fontSize:15,color:INK,fontFace:H,isTextBox:true,margin:0});
s.addNotes('SIDDH. Point at the overlap. Milestones land on two-week boundaries so they match the bi-weekly instructor meetings.');

/* 12 — risks */
s = light();
title(s, 10, 'What could go wrong, and what we do about it');
const risks=[['Place-and-route is 500x slower than the rest of the loop','Two-tier loop. Agents iterate on the fast gates; routing runs once when they all pass, with retries capped.'],
             ['An LLM may not fix its own RTL from tool output','The deterministic agent already in the repo satisfies the loop. We narrow the claim honestly rather than overstate it.'],
             ['Vendor tools do not run on every team machine','Bitstream generation is assigned to one owner with a documented environment, not assumed to be available to all.'],
             ['Hardware delays','Two boards already in hand, so Bronze and Silver need no purchase at all.']];
risks.forEach((r,i)=>{ const y=1.75+i*0.94;
  s.addShape(p.ShapeType.roundRect,{x:L,y:y+0.08,w:0.34,h:0.34,fill:{color:COPPER},rectRadius:0.06,line:{color:COPPER}});
  s.addText(String(i+1),{x:L,y:y+0.08,w:0.34,h:0.34,fontSize:12,bold:true,color:'FFFFFF',fontFace:H,isTextBox:true,margin:0,align:'center',valign:'middle'});
  s.addText(r[0],{x:L+0.52,y:y,w:5.0,h:0.52,fontSize:14.5,bold:true,color:INK,fontFace:H,isTextBox:true,margin:0,valign:'middle'});
  s.addText(r[1],{x:L+5.7,y:y,w:6.39,h:0.8,fontSize:13,color:MUTE,fontFace:H,isTextBox:true,margin:0});
});
s.addNotes('SIDDH. Do not rush this slide. Showing you have costed the failure modes is worth more than another feature.');

/* 13 — status */
s = light();
title(s, 11, 'Where we are today', 'The software half already works. This is not a plan for something that does not exist.');
const stats=[['66','automated tests passing'],['4','tool gates already in the loop'],['2.7 s','for a full three-iteration convergence'],['0','lines of our RTL on silicon yet']];
stats.forEach((st,i)=>{ const x=L+i*3.07;
  card(s,x,1.9,2.85,1.75, i===3?PALE:CARD);
  s.addText(st[0],{x:x+0.22,y:2.08,w:2.41,h:0.75,fontSize:40,bold:true,color: i===3?COPPER:INK,fontFace:H,isTextBox:true,margin:0});
  s.addText(st[1],{x:x+0.22,y:2.86,w:2.41,h:0.68,fontSize:12.5,color:MUTE,fontFace:H,isTextBox:true,margin:0});
});
s.addText('Working now: hardware derived from the model spec, the endpoint derived from the link rate with an automatic retry when timing fails, real device resource mapping, batching, and decode validated in a packet-level fabric simulator. Claude Haiku wrote both blocks correctly first try.',
  {x:L,y:4.0,w:W,h:1.0,fontSize:15,color:INK,fontFace:H,isTextBox:true,margin:0});
s.addText('The last number is the honest one, and it is what the next eight weeks are for.',
  {x:L,y:5.05,w:W,h:0.42,fontSize:16,bold:true,color:COPPER,fontFace:H,isTextBox:true,margin:0});
s.addNotes('SAMEER. Lead with the zero. Saying the weak number yourself is more convincing than any of the others.');

/* 14 — close */
s = dark();
s.addText('Given a model, build the hardware that runs it.', {x:L, y:2.2, w:W, h:1.5,
  fontSize:40, bold:true, color:'FFFFFF', fontFace:H, isTextBox:true, margin:0});
s.addShape(p.ShapeType.roundRect,{x:L,y:3.85,w:1.7,h:0.05,fill:{color:COPPER},rectRadius:0.02,line:{color:COPPER}});
s.addText('Then scale it by adding boards, not by buying a bigger one.', {x:L, y:4.2, w:W, h:0.5,
  fontSize:21, color:PALE, fontFace:H, isTextBox:true, margin:0});
s.addText('github.com/SameerSul/agentic-fpga-scaleout', {x:L, y:5.35, w:W, h:0.4,
  fontSize:15, color:MUTE, fontFace:M, isTextBox:true, margin:0});
s.addNotes('SAMEER. Close on the one sentence. Then questions. Likely ones: why FPGAs not GPUs, how do you know the LLM is not just copying, what happens if the boards do not arrive.');

p.writeFile({fileName:'fpgAI-proposal.pptx'}).then(f=>console.log('wrote',f));
