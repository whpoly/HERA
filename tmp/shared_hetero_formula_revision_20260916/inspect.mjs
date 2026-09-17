import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const B='C:/Users/User/Desktop/HERA/tmp/shared_hetero_formula_revision_20260916';
const P=await PresentationFile.importPptx(await FileBlob.load('C:/Users/User/Desktop/HERA/results/group_meeting_20260915/HERA_group_meeting_20260915.pptx'));
const snapshot=await P.inspect({kind:'slide,textbox,shape,table',maxChars:100000});
await fs.writeFile(`${B}/source.inspect.ndjson`,snapshot.ndjson);
console.log(snapshot.ndjson.split('\n').filter(x=> /"slide":(6|7)[,}]/.test(x)).join('\n'));
for(const n of [6,7]) {
 const s=P.slides.items[n-1];
 const layout=await s.export({format:'layout'});
 await fs.writeFile(`${B}/before-${n}.layout.json`,await layout.text());
}
