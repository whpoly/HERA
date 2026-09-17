import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const ROOT='C:/Users/User/Desktop/HERA';
const B=`${ROOT}/tmp/shared_hetero_formula_revision_20260916`;
const SKILL='C:/Users/User/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.22227/skills/presentations';
const PYTHON='C:/Users/User/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const SOURCE=`${ROOT}/results/group_meeting_20260915/HERA_group_meeting_20260915.pptx`;
const FINAL=`${ROOT}/results/group_meeting_20260915/HERA_group_meeting_20260915_shared_formulas.pptx`;
const {finalizePresentation}=await import(pathToFileURL(`${SKILL}/container_tools/artifact_tool_utils.mjs`).href);
const tableOwners=[3,4,5,6,10,12,13,14,15,16,19,20];
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:`${B}/candidate_preserved.pptx`,finalPath:FINAL,
 pythonExecutable:PYTHON,integrityValidatorPath:`${SKILL}/container_tools/inspect_presentation_package_integrity.py`,
 layoutValidatorPath:`${SKILL}/container_tools/inspect_presentation_layout_geometry.py`,
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],
 requiredNativeTableOwnerSlides:tableOwners,requiredNativeChartOwnerSlides:[8,9,11,13],
 fontPolicy:{basis:'reference',families:['Times New Roman'],referencePath:SOURCE,referenceSha256:'36f2d0b86296032bebddefbda8ced3247ddb9cceb762dd7a91f04cca0ee120f3'},
 verifyArtifactToolImport:true,receiptPath:`${B}/final.validation.json`});
console.log(JSON.stringify(result));
const P=await PresentationFile.importPptx(await FileBlob.load(FINAL));
await fs.mkdir(`${B}/final_render`,{recursive:true});
for(let i=0;i<P.slides.items.length;i++) {
 const b=await P.slides.items[i].export({format:'png',scale:1});
 await fs.writeFile(`${B}/final_render/slide-${i+1}.png`,new Uint8Array(await b.arrayBuffer()));
}
console.log(`Rendered all ${P.slides.items.length} final slides.`);
