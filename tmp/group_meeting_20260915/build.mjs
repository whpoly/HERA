import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const ROOT = 'C:/Users/User/Desktop/HERA';
const BUILD = `${ROOT}/tmp/group_meeting_20260915`;
const OUT = `${ROOT}/results/group_meeting_20260915`;
const SKILL = 'C:/Users/User/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.22227/skills/presentations';
const PYTHON = 'C:/Users/User/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } = await import(pathToFileURL(`${SKILL}/container_tools/artifact_tool_utils.mjs`).href);
const FONT = resolvePresentationFont({fontFamily:'Times New Roman'});
const D = JSON.parse(await fs.readFile(`${BUILD}/evidence.json`, 'utf8'));
const logo = await fs.readFile(`${BUILD}/assets/polyu-logo.png`);
const P = Presentation.create({slideSize:{width:1280,height:720}});
const C = {red:'#A32638',blue:'#346F96',teal:'#247D73',gray:'#7D8891',ink:'#17212A',light:'#E9EDF0',white:'#FFFFFF'};
const chartOwners=[],tableOwners=[],notes=[];
const abs = p => p.startsWith('C:') ? p : `${ROOT}/${p}`;
function txt(s,t,x,y,w,h,size=30,opt={}) {
  const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  a.text=t; a.text.style={typeface:FONT,fontSize:size,color:C.ink,autoFit:'none',verticalAlignment:'top',insets:{left:0,right:0,top:0,bottom:0},...opt};
  return a;
}
function slide(title, foot='') {
  const s=P.slides.add();s.background.fill=C.white;
  s.images.add({blob:logo,contentType:'image/png',alt:'The Hong Kong Polytechnic University logo from the supplied presentation',position:{left:977,top:20,width:278,height:54},fit:'contain'});
  txt(s,title,50,30,900,100,43);
  if(foot)txt(s,foot,50,657,1140,49,19,{color:'#555C62'});
  txt(s,String(P.slides.items.length),1210,666,35,30,18,{color:C.gray,alignment:'right'});
  return s;
}
function note(s,body,sources=[]){
  const content=body+'\n\nSources\n'+sources.map(abs).join('\n');
  s.speakerNotes.textFrame.setText(content);notes.push({slide:P.slides.items.length,text:content});
}
function table(s,values,{x=50,y=160,w=1180,h=330,widths,size=26,highlight=[]}={}) {
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,...(widths?{columnWidths:widths}:{})});
  t.styleOptions={headerRow:true,bandedRows:false};
  t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({fill:C.white,textStyle:{typeface:FONT,fontSize:size,color:C.ink},margins:{left:12,right:12,top:10,bottom:8},anchor:'center'});
  t.borders.assign({fill:'#D4DADF',width:0.7,style:'solid'});
  for(let c=0;c<values[0].length;c++){t.getCell(0,c).fill=C.red;t.getCell(0,c).text.style={typeface:FONT,fontSize:size,color:C.white,bold:true};}
  for(const r of highlight)for(let c=0;c<values[0].length;c++){t.getCell(r,c).fill='#F7EBED';t.getCell(r,c).text.style={typeface:FONT,fontSize:size,color:C.red,bold:true};}
  tableOwners.push(P.slides.items.length);return t;
}
function chart(s,type,categories,series,{x=55,y=160,w=1170,h=450,ymax,ymin=0,ytitle='MAE (meV/defect)',labels=true,legend=true,fmt='0.0',majorUnit,xTitle}={}){
  const obj=s.charts.add(type,{
    position:{left:x,top:y,width:w,height:h},categories,series:series.map((v,i)=>({name:v.name,values:v.values.map(n=>Number(n.toFixed(6))),...(v.xValues?{xValues:v.xValues}:{}),valuesFormatCode:fmt,fill:v.color??C.blue,line:{fill:v.color??C.blue,width:type==='line'?3:0},...(type==='line'?{marker:{symbol:'circle',size:7}}:{}),...(v.points?{points:v.points}:{})})),
    hasLegend:legend,legend:{position:'bottom',overlay:false,textStyle:{typeface:FONT,fontSize:22,fill:C.ink}},
    barOptions:{direction:'column',grouping:'clustered',gapWidth:100},lineOptions:{smooth:false},
    chartFill:C.white,plotAreaFill:C.white,chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0},
    xAxis:{visible:true,tickLabelPosition:'low',textStyle:{typeface:FONT,fontSize:22,fill:C.ink},...(xTitle?{title:{text:xTitle,textStyle:{typeface:FONT,fontSize:22,fill:C.ink}}}:{}),line:{fill:C.gray,width:1},majorGridlines:null},
    yAxis:{visible:true,min:ymin,...(ymax!==undefined?{max:ymax}:{}),...(majorUnit?{majorUnit}:{}),numberFormatCode:fmt,title:{text:ytitle,textStyle:{typeface:FONT,fontSize:22,fill:C.ink}},textStyle:{typeface:FONT,fontSize:21,fill:C.ink},majorGridlines:{fill:C.light,width:1},line:{fill:'none',width:0}},
    dataLabels:{showValue:labels,position:'outEnd',textStyle:{typeface:FONT,fontSize:21,fill:C.ink}},
  });
  applyPresentationChartFont(obj,{fontFamily:FONT});chartOwners.push(P.slides.items.length);return obj;
}
const mev=a=>a*1000;

// 1
{
const s=P.slides.add();s.background.fill=C.white;
s.images.add({blob:logo,contentType:'image/png',alt:'The Hong Kong Polytechnic University',position:{left:35,top:35,width:325,height:63},fit:'contain'});
txt(s,'Hetero GNNs for defects',65,226,1150,90,64,{alignment:'center'});
txt(s,'Low-to-high generalization in 2DMD',110,335,1060,60,36,{alignment:'center',color:C.red});
txt(s,'WU Hao',100,529,1080,45,29,{alignment:'center'});
txt(s,'Group meeting    2026/9/15',100,582,1080,36,25,{alignment:'center'});
note(s,'这次汇报接着之前的异构图工作，重点收敛到 2DMD 的低缺陷浓度训练、高缺陷浓度测试。首先说明数据与评估口径，然后介绍共享关系参数和逐缺陷能量读出，再讨论材料差异、超图消融和可复现性。日期按本次整理日期填写，可按实际组会日期修改。',['C:/Users/User/Desktop/group meeting 20250805.pptx']);
}
// 2
{
const s=slide('Recent progress','Current results use seed 123. Historical comparisons do not isolate individual changes.');
txt(s,'Shared hetero improves the mixed-low result on MoS₂',60,160,1120,54,33,{bold:true,color:C.red});
txt(s,'High-test MAE: 0.04554 eV/defect on MoS₂ and 0.05905 on WSe₂',60,222,1120,70,30);
txt(s,'Energy mean readout matches the graph-level target',60,331,1120,50,33,{bold:true});
txt(s,'Predict a scalar from each defect representation, then average over defects.',60,390,1120,80,29);
txt(s,'Dense-defect bias remains the main limitation',60,499,1120,50,33,{bold:true});
txt(s,'Hypergraph updates have not improved the current matched baseline.',60,557,1120,50,29);
note(s,'最重要的新结果来自同一份 mixed-low 训练的共享 hetero checkpoint：MoS₂ high 误差 0.04554，WSe₂ high 误差 0.05905。相对于历史记录，MoS₂ 改善显著，但历史实验没有统一代码和推理精度，所以不能把改善全部归因于某一个改动。当前仍有明显的高浓度系统性高估。超图方面，新的读出明显缓解了旧版本的大误差，但在严格对应的超图更新消融里，关闭更新的版本最好。',['logs/hetero_energy_mean_benchmark/high_test_predictions_fp32/low_to_high_report.md','logs/2dmd_mos2_hypergraph_ablation/alignn/2dmd_mos2/summary.txt']);
}
// 3
{
const s=slide('2DMD data and prediction target','Counts below describe each raw MoS₂/WSe₂ material subset. Mixed-low preprocessing retains 11,864 structures.');
txt(s,'Target:  y = Eformation / Ndefects',60,139,1130,62,38,{bold:true,color:C.red});
table(s,[['Low-density data','Structures per material'],['1 defect','4'],['2 defects','127'],['3 defects','5,802'],['Total','5,933']],{x:60,y:235,w:550,h:320,widths:[320,230],size:27});
table(s,[['High-density data','Structures per material'],['4, 9, 14, 19, 24 defects','100 at each count'],['Total','500']],{x:655,y:235,w:565,h:230,widths:[325,240],size:27});
txt(s,'97.8% of low structures have exactly 3 defects.',665,503,535,90,31,{color:C.red});
note(s,'预测目标核对为 formation_energy 除以缺陷数量，单位是 eV/defect，不是对所有原子平均。每种材料的 low 原始数据有 5933 条，其中 5802 条恰好 3 个缺陷，占约 97.8%。high 包含 4、9、14、19、24 个缺陷，每组 100 条。低浓度验证误差小只能说明 low 域拟合好，不能代表模型已经学会高浓度相互作用。mixed-low 经预处理保留 11864 条，和两个原始 low 子集简单相加的 11866 条有区别。',['logs/hetero_validation/wse2_evaluation_findings.md','logs/hetero_validation/audit_mixed_vs_single_wse2.json']);
}
// 4
{
const s=slide('Evaluation protocols','Checkpoint selection and learning-rate scheduling use validation data from the training domain.');
table(s,[['Experiment','Training / validation','Test set'],['Low-domain benchmark','Mixed MoS₂ + WSe₂ low\n7,118 / 2,373','Held-out low: 2,373'],['High-domain benchmark','Six-material high\n1,800 / 600','Held-out high: 600'],['Single-material transfer','One material low\n4,746 / 1,187','Same material high: 500'],['Mixed-material transfer','Mixed low checkpoint\nSame low split as above','MoS₂ high: 500\nWSe₂ high: 500']],{y:155,h:425,widths:[320,455,405],size:26});
txt(s,'Low-domain MAE and low-to-high MAE answer different questions.',60,599,1160,40,29,{color:C.red});
note(s,'这里有四种口径，必须分开。low benchmark 的 test 仍然来自 low；high benchmark 在包含六种材料的 high 数据中做域内划分。single-material 是某一种材料的 low 训练，测试同一材料的全部 high。mixed-material 使用混合 MoS₂/WSe₂ low 训练的一份模型，分别测试两个材料的 high，不包括另外四种训练未见材料。所有 checkpoint 都按训练域验证集选择。最近反复查看 high 来提出模型修改，因此 high 已参与开发反馈，后续更强的泛化结论需要冻结实验设置并增加独立外部验证。',['logs/hetero_validation/audit_mixed_vs_single_wse2.json','logs/hetero_energy_mean_benchmark/high_test_predictions_fp32/low_to_high_analysis.json','logs/hetero_energy_mean_benchmark/alignn/summary.txt','main.py']);
}
// 5
{
const s=slide('Models compared in the current study','Architectures below refer to the HERA implementations. These comparisons change more than pooling.');
const modelTable=table(s,[['Model','Graph / messages','Readout'],['MEGNet sparse','Defect-only graph, 12 Å\nCurrent + original species\nMax in message/state blocks','Node + edge Set2Set\nand global state'],['ALIGNN attention','Atom/bond-angle graph, 6 Å\nType-aware neighbor attention\nShared message network','Type- and context-aware\nglobal attention'],['Shared ALIGNN hetero','Four directed relation types\nOne message core per layer\nRank-8 relation adapters','Mean of per-defect\nscalar predictions'],['DefiNet-style ALIGNN','Marker-pair and distance gates\nScalar branch implemented in HERA','Mean over all nodes,\nthen MLP']],{y:153,h:445,widths:[300,515,365],size:25});
modelTable.rows[0].height=54;
note(s,'这里避免把 sparse、attention 和 hetero 当成只差一个 pooling 的模型。sparse 使用 MEGNet 的缺陷图和 12 Å 距离，输入包含当前及原位元素，块内用了 max，但最终读出是节点和边的 Set2Set 再加全局状态。attention 和 hetero 使用 ALIGNN 的距离与角度主干，通常 cutoff 6 Å。attention 用节点类型参与局部与全局注意力，hetero 则保留四种关系，并共享消息主体。HERA 的 DefiNet 是标记对门控的 scalar 实现，不应当称为原论文所有分支的完整复现。',['models/alignn.py','models/modules.py','config','logs/hetero_validation/compare_mos2_sparse_config.json']);
}
// 6
{
const s=slide('Shared message core with relation adapters','Sharing applies within each layer. Different depths retain separate parameters.');
txt(s,'r ∈ {pristine–pristine, defect–defect, pristine–defect, defect–pristine}',55,138,1165,58,27);
txt(s,'[δzᵣ, δvᵣ] = sigmoid(aᵣ) Bᵣ SiLU(Aᵣ [hⱼ, hᵢ, eᵢⱼ])',60,229,1150,57,35,{color:C.red});
txt(s,'mᵢⱼ,ᵣ = sigmoid(zᵢⱼ + δzᵣ) ⊙ (Wₘ hⱼ + δvᵣ)',60,306,1150,57,35);
txt(s,'Shared core learns reusable interactions. Each relation adds a small correction.',60,404,1130,80,29);
table(s,[['Independent','Shared','Shared + rank-8 adapters'],['916,289 parameters','614,465 parameters','679,193 parameters']],{x:60,y:509,w:1160,h:103,widths:[350,350,460],size:27,highlight:[1]});
note(s,'四类有向边在每一层共享同一套消息主体，包括源、目标、边的 gate 投影和消息 value 投影。不同层之间没有共享。同一层同一种关系只有一个 adapter，所有该类边都调用它。adapter 根据 h_j、h_i、e_ij 产生 gate 和 value 的修正，瓶颈宽度 8，输出投影从零初始化，缩放参数初始为 -3，因此模型初始等同纯 shared。四类初始距离编码、节点类型 embedding 和关系融合仍有类型区分，不能说模型的所有参数都共享。参数从独立版本的 916289 降到 679193，约减少 25.9%。',['docs/hetero_alignn_shared_relations.md','models/alignn.py']);
}
// 7
{
const s=slide('Averaging predicted defect contributions','Only actual defects enter the readout. Pristine atoms still influence defect representations through messages.');
txt(s,'Previous readout',65,158,530,48,32,{bold:true});
txt(s,'ŷ = g(meanᵢ hᵢ)',65,229,530,72,44);
txt(s,'Average latent features before\napplying the nonlinear head.',65,337,535,104,30);
txt(s,'Current readout',680,158,535,48,32,{bold:true,color:C.red});
txt(s,'ŷ = meanᵢ g(hᵢ)',680,229,535,72,44,{color:C.red});
txt(s,'Predict each latent contribution,\nthen average over actual defects.',680,337,535,104,30);
txt(s,'Same MLP: 64 – 64 – 32 – 1. Same parameter shapes and graph-level loss.',65,485,1150,76,30);
txt(s,'Latent contributions have no unique per-defect energy labels.',65,578,1150,43,29,{color:C.red});
note(s,'改动只改变非线性预测头和平均的先后顺序。原来先将缺陷表示平均再输入 MLP，新版先用同一个 MLP 预测每个实际缺陷的标量贡献，再对这些标量平均。非线性一般不与 mean 交换，所以两种函数不同。它与平均缺陷形成能的目标更一致，但并不证明每个标量就是唯一的真实缺陷能量，因为我们只有结构标签。环境信息仍通过主干影响 h_i，高浓度 h_i 的分布变化仍可能使新版失败。该改动没有增加预测头参数，scaler 在图级输出上只逆变换一次。',['docs/hetero_defect_energy_mean.md','models/alignn.py']);
}
// 8
{
const s=slide('Mixed-low training: high-test results','Same 500 test IDs and targets per material. Historical baselines use earlier training versions / precision.');
const hist=D.mixed.historical_mixed_baselines;
chart(s,'bar',['MoS₂ high','WSe₂ high'],[
 {name:'Sparse, historical',values:hist[0]?[mev(hist[0].MoS2.mae),mev(hist[0].WSe2.mae)]:[],color:C.gray},
 {name:'Attention, historical',values:[mev(hist[1].MoS2.mae),mev(hist[1].WSe2.mae)],color:C.blue},
 {name:'Hetero, historical',values:[mev(hist[2].MoS2.mae),mev(hist[2].WSe2.mae)],color:C.teal},
 {name:'Shared hetero, current',values:[mev(D.mixed.materials.MoS2.mae),mev(D.mixed.materials.WSe2.mae)],color:C.red},
],{y:160,h:420,ymax:210,majorUnit:50});
txt(s,'Current model: MoS₂ 45.54 meV/defect, WSe₂ 59.05 meV/defect',60,600,1160,40,29,{bold:true,color:C.red});
note(s,'这页所有模型都用 mixed low 训练。当前共享 hetero 通过一份最佳 epoch 303 的 checkpoint 分别预测两个 high 子集，用 FP32、batch 1。历史 baseline 已核对测试 ID 和标签一致，但没有统一代码和精度重训，因此只做历史结果比较。MoS₂ 相对旧 hetero 从 179.64 降至 45.54 meV，改善约 74.7%；WSe₂ 从 59.77 降至 59.05，基本相近。不能将当前结果直接宣称为所有配置下都优于 attention 或 sparse，更不能与单材料训练结果混为一组。',['logs/hetero_energy_mean_benchmark/high_test_predictions_fp32/low_to_high_analysis.json']);
}
// 9
{
const s=slide('Error grows with defect count','Current mixed-low shared hetero. Each point contains 100 high-density structures per material.');
const counts=['4','9','14','19','24'];
chart(s,'line',counts,['MoS2','WSe2'].map((m,i)=>({name:m==='MoS2'?'MoS₂':'WSe₂',values:counts.map(n=>mev(D.mixed.materials[m].by_defect_count[n].mae)),color:i?C.red:C.blue})),{x:40,y:172,w:590,h:396,ymax:140,majorUnit:35,labels:false,xTitle:'Defects per structure'});
chart(s,'line',counts,['MoS2','WSe2'].map((m,i)=>({name:m==='MoS2'?'MoS₂':'WSe₂',values:counts.map(n=>mev(D.mixed.materials[m].by_defect_count[n].bias)),color:i?C.red:C.blue})),{x:650,y:172,w:590,h:396,ymax:140,ymin:-35,majorUnit:35,ytitle:'Mean bias (meV/defect)',labels:false,xTitle:'Defects per structure'});
txt(s,'At 24 defects, mean bias is +83.6 meV on MoS₂ and +116.6 meV on WSe₂.',60,597,1160,47,29,{color:C.red});
note(s,'将 500 条 high 按缺陷数分组后，误差随数量增加明显上升。4 个缺陷时 WSe₂ 比 MoS₂ 更低，但到了 24 个缺陷时 WSe₂ 达到 121.19 meV。右图定义偏差为预测减真实。24 缺陷时 MAE 与正偏差非常接近，说明误差主要由系统性高估构成，单纯增加优化轮数并不能直接针对这一现象。数据支持高浓度泛化是主要限制，但还不能唯一确定是几何交互、关系频率还是表示和读出的相关性造成。',['logs/hetero_energy_mean_benchmark/high_test_predictions_fp32/low_to_high_analysis.json']);
}
// 10
{
const s=slide('Single-material rankings differ','High-test MAE, eV/defect. Seed 123. *MoS₂ shared hetero: supplied history, checkpoint not verified locally.');
txt(s,'Each model trains on the low subset of its test material.',60,129,1140,37,27);
table(s,[['Model','MoS₂ high','WSe₂ high'],['MEGNet sparse','0.161786','0.036967'],['ALIGNN attention','0.056992','0.046233'],['Shared hetero + energy mean','0.036838*','0.052188']],{y:180,h:300,widths:[420,380,380],size:29,highlight:[3]});
txt(s,'MoS₂: shared hetero has the lowest reported error in this comparison.',60,522,1150,55,30);
txt(s,'WSe₂: sparse and attention remain better than the current hetero.',60,590,1150,44,30,{color:C.red});
note(s,'这页改为单材料训练口径，不能与上一页 mixed-low 混合排名。MoS₂ 的新 hetero 0.036838 来自用户贴出的训练 history 和明确的实验目录，该 checkpoint 尚未本地核对，标星展示。WSe₂ 新 hetero checkpoint 已下载，保存的 test MAE 为 0.052188；它仍比 sparse 0.036967、attention 0.046233 大。旧版 WSe₂ sparse、attention、hetero 的全 high 复核保留了排序，因此目前没有证据说明优势来自简单的测试集读错。但该比较没有固定 backbone、特征与所有训练条件，不能把差异唯一归于材料本身。',[D.mos_hetero_reported.source,'logs/hetero_energy_mean_benchmark/alignn/summary.txt','logs/hetero_validation/wse2_evaluation_findings.md','docs/hypergraph_v3_completed_diagnosis.md','logs/hetero_validation/compare_mos2_sparse_retest_history.json']);
}
// 11
{
const s=slide('Mixed and single-material training differ','Same WSe₂ high test set. Training composition and the number of WSe₂ training examples both change.');
chart(s,'bar',['Sparse','Attention','Shared hetero'],[
 {name:'Mixed low training',values:[113.74063890224,106.536772943588,59.05464739048],color:C.gray},
 {name:'WSe₂-only low training',values:[36.96663837044,46.23291768840,52.18847595155],color:C.red},
],{y:161,h:380,ymax:140,majorUnit:35});
txt(s,'WSe₂ training samples: 3,588 in mixed low versus 4,746 in WSe₂-only low.',60,555,1160,46,28,{color:C.red});
txt(s,'Material interference is a hypothesis. Historical model versions also differ.',60,610,1160,37,26);
note(s,'同样在 WSe₂ high 上，mixed-low sparse 与 attention 的历史结果比单材料训练差很多。一个明确的混杂因素是 mixed 60/20/20 划分只用了 3588 条 WSe₂ 训练结构，而单材料 80/20 划分用了 4746 条 WSe₂。也存在历史版本不同，例如 mixed attention 的旧 BN 与单材料较新配置。因而不能直接归因于负迁移。更可靠的对照是保持相同 WSe₂ 训练 ID，再加入 MoS₂ 数据，并匹配总优化步数、材料采样与 checkpoint 选择。当前 shared hetero 的 mixed 和 single WSe₂ 差距较小，但仍只有单 seed。',['logs/hetero_validation/audit_mixed_vs_single_wse2.json','logs/hetero_energy_mean_benchmark/high_test_predictions_fp32/low_to_high_analysis.json','logs/hetero_energy_mean_benchmark/alignn/summary.txt']);
}
// 12
{
const s=slide('Hypergraph results depend on the version','MoS₂ low-to-high, seed 123. Different rows change readout and sometimes other architecture components.');
table(s,[['Version','Readout','High MAE (eV/defect)'],['Original attention reference','Global attention','0.056992'],['Hypergraph v2','Earlier hierarchical readout','0.168732'],['Hypergraph v3, earlier run','Hierarchical attention','0.250031'],['Hypergraph v3, later run','Defect mean, local + global','0.068386']],{y:163,h:362,widths:[435,460,285],size:26,highlight:[4]});
txt(s,'The 0.25 and 0.068 results belong to different trained models.',60,567,1150,43,31,{bold:true,color:C.red});
txt(s,'A better readout does not establish a benefit from hypergraph messages.',60,614,1150,34,27);
note(s,'之前以为结果在 0.1 左右，后来看到 0.25 或 0.06，关键是版本不一致。v2 为 0.168732；旧 v3 使用分层 attention 读出，checkpoint 的预测头输入 256 维，结果 0.250031。后来的 defect_mean 去掉额外背景读出分支，预测头输入 64 维，重新训练的 local+global 为 0.068386。不能将推理时临时换 pooling 的实验解释为新版本训练成绩，也不能把不同版本所有改善都归因于唯一因素。接下来用同一版本的更新消融判断超边消息是否有贡献。',['docs/hypergraph_v3_completed_diagnosis.md','logs/2dmd_mos2_hypergraph_ablation/alignn/2dmd_mos2/summary.txt']);
}
// 13
{
const s=slide('Hypergraph updates: matched ablation','All MAEs: meV/defect. Fixed v3, defect mean, seed 123. “None” retains the HyperALIGNN backbone.');
chart(s,'bar',['No hypergraph update','Local only','Local + global'],[{name:'High test',values:[61.94071662426,75.82313859463,68.38602530956],color:C.blue,points:[{idx:0,fill:C.red}]}],{x:50,y:160,w:735,h:420,ymax:100,majorUnit:25,legend:false});
table(s,[['Update','Low val MAE'],['None','2.066'],['Local','2.313'],['Local + global','2.239']],{x:833,y:217,w:392,h:253,widths:[200,192],size:25});
txt(s,'Neither enabled update improves on “none” in this run.',60,603,1160,45,30,{bold:true,color:C.red});
note(s,'这三组才是当前相对明确的超图消息对照，物理主干、读出和初始权重一致。关闭整个超图更新块为 0.06194，local 0.07582，local+global 0.06839。global 相对 local 减少了一部分误差，但仍然没超过关闭更新。这个结果不支持“local 永远没用”，也不能评价尚未完成的 global-only 或 energy-mean 组合。当前局部超边由中心 defect 和 3 Å 内 pristine 构成，不包含其他 defect；全局超边连接全部 defects，没有直接输入成对的 defect 距离。单 seed 结果需要复验。',['docs/hypergraph_update_ablation.md','docs/hypergraph_energy_global_ablation.md','logs/2dmd_mos2_hypergraph_ablation/alignn/2dmd_mos2/summary.txt']);
}
// 14
{
const s=slide('Reproducibility and training sensitivity','MAE: eV/defect. Short-run numerical differences do not explain the full historical accuracy gap.');
table(s,[['MoS₂ sparse run','Best low val MAE','High MAE'],['Local saved run','0.002972','0.161786'],['New supplied run','0.002466','0.192789']],{y:156,h:192,widths:[500,340,340],size:28});
txt(s,'Lower low-domain validation error accompanied worse high-domain error.',60,379,1160,48,29,{color:C.red});
table(s,[['Independent-process audit','Finding'],['Fixed checkpoint inference','Exact across repeat GPU runs'],['100 training steps on GPU','First gradient differs by up to 7.45 × 10⁻⁹'],['100 training steps on CPU, 1 thread','Exact repeat']],{y:455,h:177,widths:[570,610],size:25});
note(s,'MoS₂ sparse 新跑出的 high 误差比本地记录高约 19.2%，虽然 low 验证更好。已确认本地旧 checkpoint 与当前 sparse 默认配置和参数形状匹配，但新远端权重与环境未完全核对，不能断言两次一切相同。独立进程短测试固定了代码、环境、图、初始化和 batch 顺序。GPU 固定权重推理完全一致，第一步梯度开始出现约 7.45e-9 的差异，100 步后最大预测差约 1.86e-5；CPU 单线程完全一致。这证明训练数值可重复性有限，不能证明这么小的差异必然导致历史 0.03 的 MAE 差。后续要记录代码、环境和划分并跑多 seed。',['logs/hetero_validation/compare_mos2_sparse_retest_history.json','logs/hetero_validation/compare_mos2_sparse_config.json','logs/sparse_reproducibility/20260915_021549_556655/report.json']);
}
// 15
{
const s=slide('Next controlled hetero experiments','Implemented options. Full low-to-high accuracy results are not yet available locally.');
table(s,[['Variant','Single intended change','Question tested'],['Current baseline','Shared + rank-8 adapters\nEnergy mean readout','Reference result'],['Cross-relation attention','Normalize gate logits across\nall incoming relations','Does relation-wise normalization\nlose useful competition?'],['Sparse residual','Add 12 Å defect–defect messages\nDistance RBF + ordinary mean\nZero-initialized output projection','Does direct defect geometry\nhelp at higher density?']],{y:164,h:362,widths:[295,460,425],size:25});
txt(s,'Compare each change separately with the same split and initialization.',60,565,1150,47,30,{color:C.red});
txt(s,'A zero initial residual matches the baseline initially, but gives no accuracy guarantee.',60,614,1150,35,26);
note(s,'当前基线已经完成结果分析。下一步两个选项都已实现，但没有本地完整训练结果。跨关系注意力复用 gate logits，对同一目标的所有关系入边统一逐通道 softmax，仍保留关系消息槽。它不是原 attention 的四头网络完全复制。sparse residual 在主干读出之前加入 12 Å 的真实缺陷间消息，使用距离 RBF 和 gated message，按照用户确定的普通 mean 聚合，不能写成 max。输出投影从零初始化，因此初始模型与 baseline 完全相同。优先分开测试，避免把多个改动叠加后无法解释。',['docs/hetero_interaction_ablation.md','models/alignn.py']);
}
// 16
{
const s=slide('Full benchmark protocol','Prepared command and inference support. The complete benchmark has no new local result table yet.');
table(s,[['Stage','Coverage','Output'],['1. Domain benchmarks','CGCNN, MEGNet, ALIGNN\nAll supported modes on low and high','32 modes × 2 datasets\n64 training runs at seed 123'],['2. Low-to-high evaluation','Load only 2dmd_low checkpoints\nEvaluate MoS₂ and WSe₂ separately','32 models × 2 materials\nPer-model predictions and MAE'],['3. Stability follow-up','Repeat selected comparisons\nwith multiple fixed seeds','Mean ± standard deviation\nError and bias by defect count']],{y:161,h:373,widths:[300,460,420],size:25});
txt(s,'Freeze the protocol before interpreting the final model ranking.',60,570,1160,48,31,{color:C.red});
txt(s,'“All” includes supported WAS variants. Duplicate hetero fixed-pool r=0 is skipped.',60,620,1160,28,24);
note(s,'已经补齐 main 与 low checkpoint 推理脚本。all 覆盖三个 backbone 的适用模式，共 32 个非重复配置，两个数据集合计 64 组 seed123 训练。训练完只筛选 low 权重，在 MoS₂ high 和 WSe₂ high 分开测试，每个 500 条。high 域内 benchmark 则包含六种材料，不能与这两个子集混淆。当前命令是单 seed，不能生成有意义的 seed 方差；多 seed 是下一步。也要明确，各 backbone 保留适用的原配置，这是一组 benchmark，不是每一项都属于单因素消融。',['README.md','main.py','predict_2dmd_low_checkpoints.py','tests/test_predict_2dmd_low_checkpoints.py']);
}
// 17
{
const s=slide('Conclusions and next steps','Current conclusions are limited to the available configurations and seed-123 runs.');
txt(s,'Shared hetero is a stronger current baseline',60,157,1150,54,34,{bold:true,color:C.red});
txt(s,'Mixed-low high MAE: MoS₂ 0.04554, WSe₂ 0.05905 eV/defect.',60,219,1150,52,30);
txt(s,'Dense-defect generalization remains unresolved',60,337,1150,54,34,{bold:true});
txt(s,'The residual error increasingly reflects positive bias as defect count rises.',60,399,1150,64,30);
txt(s,'The next decision needs matched experiments',60,512,1150,52,34,{bold:true});
txt(s,'Finish the all-model benchmark, then compare relation attention and\ndistance-based defect residuals with repeated seeds.',60,574,1150,74,29);
note(s,'收束到三个结论。第一，共享消息主体、关系 adapter 和能量均值读出形成了目前更强的 hetero 基线，尤其是 mixed-low 的 MoS₂，但单材料 WSe₂ 仍未超过 sparse 和 attention。第二，主要剩余问题是高浓度正偏差，不能只追求更低的 low 验证误差。第三，超图更新没有显示额外收益，应先完成可追溯的全模型 benchmark，再做跨关系 attention 和距离缺陷残差的独立实验，用多 seed 判断提升是否稳定。这里不承诺任何一个结构一定超过 attention。',['logs/hetero_energy_mean_benchmark/high_test_predictions_fp32/low_to_high_report.md','logs/2dmd_mos2_hypergraph_ablation/alignn/2dmd_mos2/summary.txt','docs/hetero_interaction_ablation.md']);
}
// 18, backup
{
const s=slide('Appendix: attention architecture','Mechanistic interpretation below is a hypothesis about generalization, not a controlled causal result.');
txt(s,'Local message and neighbor weighting',60,141,1150,45,31,{bold:true});
txt(s,'uᵢⱼ = sigmoid(G[hᵢ,hⱼ,eᵢⱼ]) ⊙ MLP([hᵢ,hⱼ,eᵢⱼ])',60,202,1150,55,31);
txt(s,'αᵢⱼ = mean of 4 head-wise neighbor softmax weights',60,270,1150,50,31,{color:C.red});
txt(s,'Global readout',60,359,1150,44,31,{bold:true});
txt(s,'βᵢ = softmaxᵢ(score[hᵢ, typeᵢ, meanⱼ hⱼ])',60,419,1150,50,32);
txt(s,'ŷ = MLP(Σᵢ βᵢ hᵢ)',60,484,1150,51,35,{color:C.red});
txt(s,'Shared messages, neighbor competition and learned defect emphasis may help transfer.',60,573,1140,68,28);
note(s,'默认配置是 hidden64、3 个 attention ALIGNN block 加 3 个 GCN block，后接类型和全局上下文条件的 attention 读出。局部消息同时看源、目标和边特征，并有 sigmoid gate；注意力打分再额外看两个端点类型。四个 head 分别对邻居归一化，最后平均权重乘同一条消息，不是标准 Transformer 把四套 value 拼接。全局 score 同时看 h_i、类型 embedding 和全图平均上下文。共享参数和邻居竞争是值得借鉴的机制，但不能从性能排名证明它们就是泛化提升原因，也不能把注意力权重当作因果重要性。',['models/alignn.py','models/modules.py']);
}
// 19
{
const s=slide('Appendix: adapters and DefiNet-style gates','The current baseline uses a linear source-message projection plus relation adapters.');
table(s,[['Component','Conditioning','Role'],['Hetero relation adapter','Source, destination and edge state\nOne adapter per relation per layer','Corrects both gate logits\nand message values'],['DefiNet-style gate','Marker-pair embedding\nwith separate distance modulation','Reweights a shared message\nwithout neighbor softmax'],['Pair-message option','MLP([source, destination, edge])','Makes the shared value\nexplicitly pair-conditioned'],['Shared-distance option','Shared RBF encoder\nwith zero-initialized relation offsets','Reduces independent\nrelation-distance encoders']],{y:153,h:405,widths:[320,510,350],size:25});
txt(s,'Pair-message and shared-distance options need separate performance tests.',60,592,1160,45,29,{color:C.red});
note(s,'用户之前关心 adapter 和 DefiNet 是否一样。两者都利用类型信息，但参数化不同。adapter 是每层、每关系一套根据源目标边状态计算的低秩修正，同时改变 gate 和 value；DefiNet-style scalar 门控根据标记对和距离调制共享消息，不等同四个独立关系修正网络。前面还实现过 pair_message 与 shared_distance 选项。当前带正式结果的基线没有同时开启这些选项，也没有启用下一轮的 cross_relation_attention 或 sparse_residual，因此报告不能把它们写成已有性能来源。',['docs/hetero_alignn_shared_relations.md','docs/hetero_message_distance_ablation.md','models/modules.py','models/alignn.py']);
}
// 20
{
const s=slide('Appendix: checkpoint and metric identity','All MAEs below use eV/defect. Paths, source tables and caveats are recorded in the speaker notes.');
table(s,[['Saved model / evaluation','Best epoch','Test MAE'],['Mixed low, held-out low','303','0.002633'],['Same mixed-low weights, MoS₂ high','303','0.045536'],['Same mixed-low weights, WSe₂ high','303','0.059055'],['Separate high-trained model, high-domain test','214','0.052567'],['Separate WSe₂-low model, WSe₂ high','243','0.052188']],{y:162,h:358,widths:[750,170,260],size:25});
txt(s,'Shared hetero: r=0, LayerNorm, rank 8, defect_energy_mean',60,556,1160,42,28,{color:C.red});
txt(s,'Mixed-low checkpoint SHA-256 begins aaf4995019b6dbe7',60,608,1160,32,24);
note(s,'这页用于追溯常见的数字混淆。0.002633 是 mixed-low 模型的 low 域内 test，不是 high；0.052567 是另一个用 high 训练的模型做 high 域内测试，不是 low-to-high。0.045536 和 0.059055 才来自同一份 mixed-low checkpoint 的两个外部 high 子集。shared hetero 的全程 LayerNorm、r0、rank8 和 energy mean 由 checkpoint config 核对。混合低浓度 checkpoint SHA256 为 aaf4995019b6dbe70d4da0c5aa65ff9eefead9f37e49f4b64c7cc1a048548a49。本次外部评估为 FP32、batch1、RTX5060Ti，500 条唯一 ID 和原始 high 标签已匹配。',['logs/hetero_energy_mean_benchmark/high_test_predictions_fp32/low_to_high_analysis.json','logs/hetero_energy_mean_benchmark/alignn/summary.txt','logs/hetero_energy_mean_benchmark/alignn/2dmd_high/hetero/r0/features_layernorm/pool_defect_energy_mean/relations_shared_residual_rank8/seed123_history.csv','logs/hetero_energy_mean_benchmark/alignn/2dmd_wse2/hetero/r0/features_layernorm/pool_defect_energy_mean/relations_shared_residual_rank8/seed123_history.csv']);
}

await fs.mkdir(OUT,{recursive:true});
await fs.mkdir(`${BUILD}/draft_render`,{recursive:true});
await fs.writeFile(`${BUILD}/notes.json`,JSON.stringify(notes,null,2));
await fs.writeFile(`${BUILD}/requirements.json`,JSON.stringify({chartOwners,tableOwners,font:FONT},null,2));
const candidate=`${BUILD}/candidate.pptx`;
await (await PresentationFile.exportPptx(P)).save(candidate);
console.log(`Exported ${P.slides.items.length} slides`);
for(let i=0;i<P.slides.items.length;i++){
  const blob=await P.export({slide:P.slides.items[i],format:'png',scale:1});
  await fs.writeFile(`${BUILD}/draft_render/slide-${i+1}.png`,new Uint8Array(await blob.arrayBuffer()));
  const lay=await P.slides.items[i].export({format:'layout'});
  await fs.writeFile(`${BUILD}/draft_render/slide-${i+1}.layout.json`,await lay.text());
}
console.log('Rendered all slides');
if(process.argv.includes('--finalize')) {
 const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:`${OUT}/HERA_group_meeting_20260915.pptx`,pythonExecutable:PYTHON,integrityValidatorPath:`${SKILL}/container_tools/inspect_presentation_package_integrity.py`,layoutValidatorPath:`${SKILL}/container_tools/inspect_presentation_layout_geometry.py`,layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[...new Set(tableOwners)].flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:20,requiredNativeTableOwnerSlides:[...new Set(tableOwners)],requiredNativeChartOwnerSlides:[...new Set(chartOwners)],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:`${BUILD}/final.validation.json`});
 console.log(JSON.stringify(result));
}
