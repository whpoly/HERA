import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const ROOT='C:/Users/User/Desktop/HERA';
const B=`${ROOT}/tmp/shared_hetero_recap_revision_20260916`;
const SKILL='C:/Users/User/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.22227/skills/presentations';
const PYTHON='C:/Users/User/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const SOURCE=`${ROOT}/results/group_meeting_20260915/HERA_group_meeting_20260915_shared_formulas.pptx`;
const FINAL=`${ROOT}/results/group_meeting_20260915/HERA_group_meeting_20260915_shared_recap_final.pptx`;
const {finalizePresentation}=await import(pathToFileURL(`${SKILL}/container_tools/artifact_tool_utils.mjs`).href);
const tableOwners=[2,3,4,5,6,10,12,13,14,15,16,19,20];
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:`${B}/candidate_preserved.pptx`,finalPath:FINAL,
 pythonExecutable:PYTHON,integrityValidatorPath:`${SKILL}/container_tools/inspect_presentation_package_integrity.py`,
 layoutValidatorPath:`${SKILL}/container_tools/inspect_presentation_layout_geometry.py`,
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],
 requiredNativeTableOwnerSlides:tableOwners,requiredNativeChartOwnerSlides:[8,9,11,13],
 fontPolicy:{basis:'reference',families:['Times New Roman'],referencePath:SOURCE,referenceSha256:'0df90070e5cf73a09652745896bb7c74127be7cdf23a4d1d00b0f2c4150b5c8c'},
 verifyArtifactToolImport:true,receiptPath:`${B}/final-reviewed.validation.json`});
console.log(JSON.stringify({finalPath:result.finalPath,sha256:result.finalSha256,package:result.packageIntegrity.status,charts:result.nativeChartValidation.passed}));
const P=await PresentationFile.importPptx(await FileBlob.load(FINAL));
await fs.mkdir(`${B}/final_render`,{recursive:true});
for(let i=0;i<P.slides.items.length;i++) {
 const b=await P.slides.items[i].export({format:'png',scale:1});
 await fs.writeFile(`${B}/final_render/slide-${i+1}.png`,new Uint8Array(await b.arrayBuffer()));
}
console.log(`Rendered all ${P.slides.items.length} final slides.`);
