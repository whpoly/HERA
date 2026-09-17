import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const ROOT='C:/Users/User/Desktop/HERA';
const B=`${ROOT}/tmp/shared_hetero_formula_revision_20260916`;
const SOURCE=`${ROOT}/results/group_meeting_20260915/HERA_group_meeting_20260915.pptx`;
const P=await PresentationFile.importPptx(await FileBlob.load(SOURCE));
const C={red:'#A32638',ink:'#17212A',gray:'#555C62'};
function update(id,text,x,y,w,h,size,opt={}) {
 const o=P.resolve(id);
 o.text=text;
 o.position={left:x,top:y,width:w,height:h};
 o.text.style={typeface:'Times New Roman',fontSize:size,color:C.ink,autoFit:'none',verticalAlignment:'top',insets:{left:0,right:0,top:0,bottom:0},...opt};
 return o;
}
function add(s,text,x,y,w,h,size,opt={}) {
 const o=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 o.text=text;
 o.text.style={typeface:'Times New Roman',fontSize:size,color:C.ink,autoFit:'none',verticalAlignment:'top',insets:{left:0,right:0,top:0,bottom:0},...opt};
 return o;
}
const s6=P.resolve('sl/x8f69ofe');
update('sh/yhg7epsj','Shared hetero: previous and current formulas',50,30,900,100,43);
update('sh/98rqt4r6','r ∈ {pristine–pristine, defect–defect, pristine–defect, defect–pristine}',55,138,1165,45,27);
add(s6,'Previous: independent cores',60,195,560,45,31,{bold:true});
add(s6,'Current: shared core + adapters',665,195,555,45,31,{bold:true,color:C.red});
update('sh/7m98ru9g','zᵣ = Wₛʳhⱼ + Wₜʳhᵢ + Wₑʳeᵢⱼ',60,250,560,46,32);
add(s6,'z = Wₛhⱼ + Wₜhᵢ + Wₑeᵢⱼ',665,250,555,46,32,{color:C.red});
add(s6,'mᵣ = σ(zᵣ) ⊙ Wₘʳhⱼ',60,305,560,46,32);
add(s6,'mᵣ = σ(z + δzᵣ) ⊙ (Wₘhⱼ + δvᵣ)',665,305,555,46,32,{color:C.red});
update('sh/ml07i9sv','[δzᵣ, δvᵣ] = σ(aᵣ) Bᵣ SiLU(Aᵣ [hⱼ, hᵢ, eᵢⱼ])',60,385,1160,50,35,{color:C.red});
update('sh/oryp8fah','W: shared across relations. Aᵣ, Bᵣ, aᵣ: relation-specific. Bottleneck = 8.',60,452,1160,42,27);
update('sh/zi98nu94','Each layer has its own weights. Superscript r denotes relation-specific weights.\nEdge indices and affine biases are omitted. Relation-wise aggregation is unchanged.',50,650,1140,58,19,{color:C.gray});
s6.speakerNotes.textFrame.setText(`这页左侧是之前的 independent，右侧是当前实验的 shared_residual，不应把纯 shared 与当前基线混为一谈。\n\n对一条 j→i、关系为 r 的边，旧版完整公式（省略层指标）是：z_ij^r = W_s^r h_j + W_t^r h_i + W_e^r e_ij + b_z^r，v_ij^r = W_m^r h_j + b_m^r，m_ij^r = sigmoid(z_ij^r) ⊙ v_ij^r。四种关系各自有一套完整参数。\n\n当前版本：z_ij = W_s h_j + W_t h_i + W_e e_ij + b_z，v_ij = W_m h_j + b_m，[delta_z_ij^r, delta_v_ij^r] = sigmoid(a_r) [B_r SiLU(A_r [h_j,h_i,e_ij] + b_A,r) + b_B,r]，m_ij^r = sigmoid(z_ij + delta_z_ij^r) ⊙ (v_ij + delta_v_ij^r)。slide 中省略了边指标和 affine biases，W_s/W_t/W_e/W_m 统称 W。\n\n同一层的四种关系共享 W。每种关系各有 A_r、B_r、a_r，同一关系内所有边共用该 adapter。不同层之间不共享。隐层 H=64，adapter 为 192→8→128，128 维分为 64 维 gate 修正和 64 维 value 修正。B 的权重和 bias 零初始化，a 初值 -3，因此初始残差为 0。\n\n聚合公式没有改变：M_i^r = [sum_{j in N_r(i)} m_ij^r] / [sum_{j in N_r(i)} sigmoid(z_ij^r, effective) + 1e-6]，逐通道相除。各关系分别归一化，再按节点类型拼接关系槽做融合。不是所有关系的边共同做一次 softmax。边状态仍按 e'_ij = e_ij + SiLU(LN(z_ij^r, effective)) 更新。\n\n节点类型 embedding、四套初始距离编码、节点类型融合仍有区别。参数数目沿用该 PPT 原来的同宽度对照：independent 916289，shared 614465，shared_residual 679193。只把四套消息主体改为共享不能消除所有关系差别。\n\nSources\n${ROOT}/models/alignn.py (HeteroRelationConv, RelationResidualAdapter, SharedHeteroRelations, _hetero_relation_update)\n${ROOT}/docs/hetero_alignn_shared_relations.md`);

const s7=P.resolve('sl/gnmp4jqx');
update('sh/cb2tkvap','Defect readout: previous and current formulas',50,30,900,100,43);
update('sh/zedcfa9g','Previous: defect_mean',65,158,540,48,32,{bold:true});
update('sh/eh0ba1sr','Current: defect_energy_mean',665,158,555,48,32,{bold:true,color:C.red});
update('sh/kzmdova1','ŷ = g((Σᵢ hᵢ) / N)',65,239,540,72,43);
update('sh/fi9c369c','ŷ = (Σᵢ g(hᵢ)) / N',665,239,555,72,43,{color:C.red});
update('sh/l0vuh0rm','Average defect representations,\nthen apply the nonlinear head g.',65,337,540,94,30);
update('sh/baxkjqlk','Apply the same head g to each defect,\nthen average the scalar predictions.',665,337,555,94,30);
update('sh/a9ojq5kz','i ∈ D, where D contains actual defects and N = |D|.\nSame MLP: 64 – 64 – 32 – 1. Same graph-level loss.',65,466,1155,92,29);
update('sh/xcz2l03q','For nonlinear g, averaging and prediction generally do not commute.',65,584,1155,42,29,{color:C.red});
update('sh/dcbud0ra','Readout is a separate change from relation sharing. Individual scalar contributions have no per-defect labels.',50,657,1140,49,19,{color:C.gray});
s7.speakerNotes.textFrame.setText(`这里是另一个独立于参数共享的改动。之前的 defect_mean 为 y_hat = g((1/N) sum_{i in D} h_i)，当前实验使用的 defect_energy_mean 为 y_hat = (1/N) sum_{i in D} g(h_i)。D 只包含真实缺陷，N=|D|，所有缺陷使用同一个非线性 MLP g。g 的层宽仍然是 64→64→32→1，参数形状没有改变。代码默认值与具体实验配置要分开理解：本页“当前”指本 PPT 的当前实验配置。\n\n非线性 g 通常不与 mean 交换。旧版先把所有缺陷的 latent representation 混合再预测，新版保留逐缺陷的非线性映射，最后对标量平均。h_i 已经包含环境和其他节点传来的消息，因此 pristine 节点仍间接影响读出。只在图级输出上逆变换 scaler 一次，图级目标仍是平均缺陷形成能。\n\n逐缺陷标量没有单独监督，不能解释为唯一真实的逐缺陷形成能。该读出与 relation sharing 是两个改动，需要分别做消融才能确定各自贡献。\n\nSources\n${ROOT}/models/alignn.py (HeteroALIGNN readout)\n${ROOT}/docs/hetero_defect_energy_mean.md`);

await (await PresentationFile.exportPptx(P)).save(`${B}/draft.pptx`);
for (const n of [6,7]) {
 const slide=P.slides.items[n-1];
 const png=await slide.export({format:'png',scale:1});
 await fs.writeFile(`${B}/after-${n}.png`,new Uint8Array(await png.arrayBuffer()));
 await fs.writeFile(`${B}/after-${n}.layout.json`,await (await slide.export({format:'layout'})).text());
}
console.log('Draft and changed-slide previews ready');
