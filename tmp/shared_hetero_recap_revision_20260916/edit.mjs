import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const ROOT='C:/Users/User/Desktop/HERA';
const B=`${ROOT}/tmp/shared_hetero_recap_revision_20260916`;
const SOURCE=`${ROOT}/results/group_meeting_20260915/HERA_group_meeting_20260915_shared_formulas.pptx`;
const P=await PresentationFile.importPptx(await FileBlob.load(SOURCE));
const C={red:'#A32638',ink:'#17212A',gray:'#555C62'};
function update(id,text,x,y,w,h,size,opt={}) {
 const o=P.resolve(id);
 o.text=text;
 o.position={left:x,top:y,width:w,height:h};
 o.text.style={typeface:'Times New Roman',fontSize:size,color:C.ink,bold:false,autoFit:'none',verticalAlignment:'top',insets:{left:0,right:0,top:0,bottom:0},...opt};
 return o;
}
const s=P.resolve('sl/hwbqtkby');
update('sh/9072xkry','Recap: limited low-to-high transfer',50,30,900,100,43);
update('sh/x4r21kru','Previous MoS₂ low-to-high runs. MAE in eV/defect.',60,141,1160,43,28);
update('sh/a10jqpsj','Lower low-val MAE, yet 2.51× the high-test error of attention.',60,388,1160,60,32,{bold:true,color:C.red});
update('sh/w3i1sfa9','Motivation for the current changes',60,466,1160,45,32,{bold:true});
update('sh/r65knqtk','Shared core + adapters: train the message core using all four relations.',60,519,1160,44,29);
update('sh/q5wjelsz','Energy mean readout: predict each defect contribution before averaging.',60,568,1160,44,29);
update('sh/ove9o7yd','Separate ablations are needed to determine the contribution of each change.',60,619,1160,32,24,{color:C.gray});
update('sh/ozy1ofad','Historical results from supplied training logs. Remote checkpoint configuration was not verified.',50,670,1140,33,18,{color:C.gray});
const t=s.tables.add({rows:3,columns:3,left:60,top:202,width:1160,height:162,columnWidths:[450,355,355],values:[
 ['Previous model','Best low-val MAE','High-test MAE'],
 ['ALIGNN attention','0.002183','0.051590'],
 ['ALIGNN hetero, independent','0.001807','0.129310'],
]});
t.styleOptions={headerRow:true,bandedRows:false};
t.cells.block({row:0,column:0,rowCount:3,columnCount:3}).assign({fill:'#FFFFFF',textStyle:{typeface:'Times New Roman',fontSize:28,color:C.ink},margins:{left:12,right:12,top:9,bottom:8},anchor:'center'});
t.borders.assign({fill:'#D4DADF',width:0.7,style:'solid'});
for(let col=0;col<3;col++){
 t.getCell(0,col).fill=C.red;
 t.getCell(0,col).text.style={typeface:'Times New Roman',fontSize:28,color:'#FFFFFF',bold:true};
 t.getCell(2,col).fill='#F7EBED';
 t.getCell(2,col).text.style={typeface:'Times New Roman',fontSize:28,color:C.red,bold:true};
}
s.speakerNotes.textFrame.setText(`开场先回顾问题：之前的 hetero 并没有达到我们希望的 low-to-high 泛化效果。这一页引用的是同一材料 MoS₂ 的两份用户提供的完整训练 history，用户明确第一份为 attention，第二份为 hetero。hetero 目录为 features_layernorm/pool_defect_mean，对应当时尚未引入关系共享的版本。这里展示历史记录，不与后文 mixed-low 的当前结果拼成单因素消融。该远端 checkpoint 的内部配置和逐样本预测尚未核验，图上保留这一来源限制。\n\nattention 的最佳 low 验证 MAE 为 0.002183，最终记录的 high 测试 MAE 为 0.051590。hetero 的最佳 low 验证 MAE 更低，为 0.001807，但 high 测试 MAE 达到 0.129310，是 attention 的 2.5065 倍。最重要的观察是：low 域拟合得更好，并没有带来更好的高缺陷浓度外推。两者都在最佳验证 epoch 后继续运行了 50 个 epoch，因此已有日志没有明显支持单纯训练不足的解释。\n\n这引出后面的设计改动。第一，旧 hetero 的四种关系各有一套消息主体。对本地 MoS₂ 物理图的统计显示，low 中 defect–defect 邻居稀少，而 high 中明显增多。这使我们提出一个待检验假设：关系完全独立可能限制稀有关系的数据利用。于是引入每层共享消息主体，再用小型 relation adapter 保留关系差别。共享核心可以从所有四种关系的训练消息中学习，是否改善泛化仍需消融检验。\n\n第二，另行检验读出：原来先平均实际缺陷的 latent representation，再通过非线性 MLP。现在让同一个 MLP 先对每个实际缺陷输出一个标量，再对这些标量求均值。目标仍是平均缺陷形成能，只使用图级标签，逐缺陷贡献不具有唯一真实能量的含义。这一步与参数共享是两个不同改动，应分别判断贡献。\n\n过渡讲稿：所以今天的内容从这个泛化差距出发，先介绍共享消息主体与关系 adapter，再介绍逐缺陷预测后求平均的读出，随后展示结果和仍然存在的高浓度误差。原有图、节点类型融合以及其他具体保留的模块在后面的公式和备注中说明。\n\nSources\n${ROOT}/logs/hetero_validation/supplied_remote_history_comparison.json\n${ROOT}/docs/hetero_relation_shift_review.md\n${ROOT}/docs/hetero_alignn_shared_relations.md\n${ROOT}/docs/hetero_defect_energy_mean.md`);
await (await PresentationFile.exportPptx(P)).save(`${B}/draft.pptx`);
const png=await s.export({format:'png',scale:1});
await fs.writeFile(`${B}/after-2.png`,new Uint8Array(await png.arrayBuffer()));
await fs.writeFile(`${B}/after-2.layout.json`,await (await s.export({format:'layout'})).text());
console.log('Recap slide edited');
