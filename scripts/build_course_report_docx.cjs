/* Optional report authoring: use the bundled Node runtime and docx/sharp packages. */
const fs=require('node:fs');
const path=require('node:path');
const sharp=require('sharp');
const {Document,Packer,Paragraph,TextRun,Table,TableRow,TableCell,ImageRun,Footer,
       AlignmentType,HeadingLevel,WidthType,ShadingType,BorderStyle,PageNumber,
       ExternalHyperlink,VerticalAlign,TableLayoutType}=require('docx');
const ROOT=path.resolve(__dirname,'..');
const OUT=path.join(ROOT,'reports');
const inputPath=process.argv[2]?path.resolve(ROOT,process.argv[2]):path.join(OUT,'course_report_content.json');
const outputPath=process.argv[3]?path.resolve(ROOT,process.argv[3]):path.join(OUT,'CLUENER2020_课程实验报告.docx');
const content=JSON.parse(fs.readFileSync(inputPath,'utf8'));
const WIDTH=9638;
const FONT={name:'Times New Roman',eastAsia:'SimSun'};
const HEAD={name:'Arial',eastAsia:'SimHei'};
const figureDir=path.join(OUT,'figures','course_report');
fs.mkdirSync(figureDir,{recursive:true});

function architecture(){
  const box=(x,y,w,h,title,lines=[])=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="#f4f6f8" stroke="#56616d" stroke-width="1.7"/>
  <text x="${x+w/2}" y="${y+31}" text-anchor="middle" font-size="22" font-weight="bold">${title}</text>`+
  lines.map((s,i)=>`<text x="${x+w/2}" y="${y+59+26*i}" text-anchor="middle" font-size="18">${s}</text>`).join('');
  const arrow=(d)=>`<path d="${d}" fill="none" stroke="#495661" stroke-width="2.2" marker-end="url(#arrow)"/>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="725" viewBox="0 0 1200 725">
  <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#495661"/></marker></defs>
  <rect width="1200" height="725" fill="white"/>
  <g font-family="Microsoft YaHei,SimSun,Arial" fill="#111111">
  ${box(30,20,300,90,'官方文件与本地原始包',['Train / Dev 有标签，Test无标签'])}
  ${box(435,20,330,90,'审计与统一字符跨度',['重复/冲突/切片检查 → JSONL'])}
  ${box(870,20,300,90,'固定划分与证据',['seed / ID / SHA256'])}
  ${arrow('M330 65 H430')}${arrow('M765 65 H865')}
  ${box(30,165,300,100,'Train 9,539句',['跳过嵌套样本，训练可用9,538句'])}
  ${box(435,165,330,100,'Dev 1,343句',['官方Dev不变；传统模型选参'])}
  ${box(870,165,300,100,'Test 1,209句',['固定原文推断；金标只供评分'])}
  ${arrow('M1020 110 V135 H180 V160')}${arrow('M600 135 V160')}${arrow('M1020 135 V160')}
  ${box(30,330,510,125,'词典 / HMM / CRF',['Train建模 → Dev选参 → Test推断','3组最终预测；保留Dev参数实验'])}
  ${box(660,330,510,125,'DeepSeek / GLM / MiMo',['Train固定例句 + Test原文 → API','各0/2/4/8-shot；不发送Test金标'])}
  ${arrow('M180 265 V325')}${arrow('M600 265 V290 H350 V325')}
  ${arrow('M1020 265 V325')}${arrow('M330 215 H390 V305 H720 V325')}
  ${arrow('M870 215 H815 V280 H495 V325')}
  <text x="88" y="307" font-size="16">Train训练</text>
  <text x="675" y="274" font-size="16">Test原文</text>
  <text x="1048" y="308" font-size="16">Test原文</text>
  ${box(30,530,310,135,'原始记录与运行证据',['响应 / 失败 / usage / 延迟','缓存 / 预算账本 / 冻结协议'])}
  ${box(435,530,330,135,'统一预测与双层评分',['实体跨度 → 双层P/R/F1','BIO按规则投影；实体保留重叠'])}
  ${box(870,530,300,135,'分析与可复现产出',['主表 / 参数 / 错误归因','配对区间 / 费用 / 离线重算'])}
  ${arrow('M285 455 V495 H600 V525')}${arrow('M915 455 V495 H600')}
  ${arrow('M700 455 V480 H185 V525')}${arrow('M765 590 H865')}
  ${arrow('M1135 265 H1185 V515 H730 V525')}
  <text x="1000" y="510" text-anchor="middle" font-size="16">Test金标仅进入评分</text>
  <text x="600" y="704" text-anchor="middle" font-size="17">新版LLM跳过新的完整Dev选参；Test金标仅在离线评估阶段读取</text>
  </g></svg>`;
}

function inline(text,options={}){
  return new TextRun({text:String(text),font:FONT,...options});
}
function prose(text){
  return String(text).replace(/([\u3400-\u9fff])([A-Za-z0-9])/g,'$1 $2')
    .replace(/([A-Za-z0-9])([\u3400-\u9fff])/g,'$1 $2');
}
function paragraph(text,options={}){
  return new Paragraph({children:[inline(text)],spacing:{after:95,line:330},indent:{firstLine:420},
    wordWrap:true,widowControl:true,...options});
}
function source(text){
  // Source paths remain editable text and are wrapped at natural slash boundaries by Word.
  return new Paragraph({children:[inline('来源 '+text,{size:16,color:'555555'})],
    spacing:{before:35,after:110,line:220},keepNext:false,wordWrap:true});
}

function buildTable(block){
  const widths=block.widths||Array(block.headers.length).fill(Math.floor(WIDTH/block.headers.length));
  widths[widths.length-1]+=WIDTH-widths.reduce((a,b)=>a+b,0);
  const border={style:BorderStyle.SINGLE,size:3,color:'BCC3CA'};
  const makeRow=(row,header=false)=>new TableRow({tableHeader:header,cantSplit:true,children:row.map((value,i)=>new TableCell({
    width:{size:widths[i],type:WidthType.DXA},verticalAlign:VerticalAlign.CENTER,
    margins:{top:block.compact?35:70,bottom:block.compact?35:70,left:95,right:95},
    shading:header?{fill:'E9EEF2',type:ShadingType.CLEAR}:undefined,
    borders:{top:border,bottom:border,left:border,right:border},
    children:[new Paragraph({children:[inline(value,{size:19,bold:header})],
      alignment:(i===0||(!header && /[\u3400-\u9fff]/.test(String(value))))?AlignmentType.LEFT:AlignmentType.CENTER,
      wordWrap:true,spacing:{after:0,line:245},keepNext:false})]
  }))});
  return [new Paragraph({children:[inline(block.title,{size:20,bold:true})],keepNext:true,spacing:{before:105,after:80}}),
    new Table({width:{size:WIDTH,type:WidthType.DXA},layout:TableLayoutType.FIXED,columnWidths:widths,
      rows:[makeRow(block.headers,true),...block.rows.map(r=>makeRow(r))]}),
    ...(block.source?[source(block.source)]:[])];
}

async function main(){
  const svg=architecture();fs.writeFileSync(path.join(figureDir,'system_architecture.svg'),svg);
  await sharp(Buffer.from(svg)).resize(1800).png().toFile(path.join(figureDir,'system_architecture.png'));
  const body=[new Paragraph({style:'Title',children:[new TextRun({text:content.title,font:HEAD,bold:true,size:34,color:'000000'})],
      spacing:{after:150}}),
    new Paragraph({children:[inline(content.subtitle,{size:24,bold:true})],spacing:{after:120}}),
    paragraph(content.metadata,{indent:{firstLine:0},spacing:{after:75},children:[inline(content.metadata,{size:19})]}),
    paragraph(content.date,{indent:{firstLine:0},spacing:{after:160},children:[inline(content.date,{size:19})]})];
  for(const b of content.blocks){
    if(b.type==='heading'){
      body.push(new Paragraph({heading:b.level===1?HeadingLevel.HEADING_1:HeadingLevel.HEADING_2,
        children:[new TextRun({text:b.text,font:HEAD,bold:true,color:'000000',size:b.level===1?28:23})],
        spacing:{before:b.level===1?210:145,after:100},keepNext:true,keepLines:true,pageBreakBefore:!!b.pageBreakBefore}));
    }else if(b.type==='paragraph')body.push(paragraph(b.literal?b.text:prose(b.text),{keepNext:!!b.keepNext,keepLines:!!b.keepLines}));
    else if(b.type==='equation')body.push(paragraph(b.text,{indent:{firstLine:0},alignment:AlignmentType.CENTER,spacing:{before:65,after:100,line:300}}));
    else if(b.type==='code'){
      const lines=b.text.split('\n');
      lines.forEach((line,index)=>body.push(paragraph(line,{indent:{firstLine:0},keepNext:index<lines.length-1,
        keepLines:true,spacing:{before:index===0?65:0,after:index===lines.length-1?100:0,line:215},
        children:[new TextRun({text:line,font:{name:'Consolas',eastAsia:'SimSun'},size:17})],
        shading:{fill:'F4F5F6',type:ShadingType.CLEAR}})));
    }
    else if(b.type==='table')body.push(...buildTable(b));
    else if(b.type==='figure'){
      const f=path.join(OUT,b.path);const metadata=await sharp(f).metadata();
      const width=b.width||620,height=Math.round(width*metadata.height/metadata.width);
      body.push(new Paragraph({children:[new ImageRun({type:'png',data:fs.readFileSync(f),transformation:{width,height},
        altText:{title:b.caption,description:b.caption,name:path.basename(f)}})],alignment:AlignmentType.CENTER,
        keepNext:true,spacing:{before:100,after:75}}));
      body.push(paragraph(b.caption,{indent:{firstLine:0},spacing:{after:60,line:250},children:[inline(b.caption,{size:19})]}));
      body.push(source(b.source));
    }else if(b.type==='reference')body.push(new Paragraph({children:[inline(b.text+' ',{size:19}),new ExternalHyperlink({link:b.url,
      children:[inline(b.url,{size:18,color:'222222'})]})],spacing:{after:80,line:260},indent:{firstLine:0}}));
  }
  const doc=new Document({creator:'NLP course project',title:content.title,description:'CLUENER2020 course experiment report',
    styles:{default:{document:{run:{font:FONT,size:22,color:'000000'},paragraph:{spacing:{after:95,line:330}}}},
      paragraphStyles:[{id:'Title',name:'Title',basedOn:'Normal',next:'Normal',run:{font:HEAD,size:34,bold:true,color:'000000'},paragraph:{spacing:{after:150}}},
        {id:'Heading1',name:'Heading 1',basedOn:'Normal',next:'Normal',quickFormat:true,run:{font:HEAD,size:28,bold:true,color:'000000'},paragraph:{outlineLevel:0,keepNext:true}},
        {id:'Heading2',name:'Heading 2',basedOn:'Normal',next:'Normal',quickFormat:true,run:{font:HEAD,size:23,bold:true,color:'000000'},paragraph:{outlineLevel:1,keepNext:true}}]},
    sections:[{properties:{page:{size:{width:11906,height:16838},margin:{top:1134,bottom:1134,left:1134,right:1134,footer:500}}},
      footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,children:[inline('第 ',{size:18}),
        new TextRun({children:[PageNumber.CURRENT],font:FONT,size:18}),inline(' 页',{size:18})]})]})},children:body}]});
  const target=outputPath;
  fs.writeFileSync(target,await Packer.toBuffer(doc));console.log('DOCX CREATED',target);
}
main().catch(error=>{console.error(error);process.exitCode=1;});
