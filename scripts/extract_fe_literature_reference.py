"""Extract independent Olsson et al. DFT values in Table S3 of the released Fe supplement."""
import argparse
import csv
import io
from pathlib import Path
import re
import zipfile

from pypdf import PdfReader


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with zipfile.ZipFile(args.bundle) as zipped:
        label={r['source_id']:float(r['binding_eV'])
               for r in csv.DictReader(io.StringIO(zipped.read('records.csv').decode()))}
        pdf=PdfReader(io.BytesIO(zipped.read('Supplementary_materials.pdf')))
    text='\n'.join(page.extract_text() for page in pdf.pages)
    s3=text.split('Table S3:',1)[1].split('Figure S1:',1)[0]
    num=r'[-+]?\d+(?:\.\d+)?'
    triplet=rf'({num}|–)\s*\(\s*({num})\s*\)'
    line=re.compile(r'^([A-Z][a-z]?)\s+'+r'\s+'.join([triplet]*3)+r'$')
    rows=[]
    for content in s3.splitlines():
        m=line.fullmatch(content.strip().replace('−','-'))
        if not m:
            continue
        element=m[1]
        for shell in range(1,4):
            current,independent=m[2*shell],float(m[2*shell+1])
            sid=f'{element}/{shell}NN'
            if element in ('Ag','Fe'):
                continue  # Ag has no paired study prediction. Fe is the zero reference.
            if current=='–' or float(current)!=label[sid]:
                raise ValueError(f'Outside-parentheses Table S3 value differs from study workbook: {sid}')
            rows.append(dict(source_id=sid,solute=element,shell=shell,reference_binding_eV=independent,
                             original_doi='10.1103/PhysRevB.81.054102',table='Kartamyshev supplement S3, parentheses'))
    if len(rows)!=66 or len({v['solute'] for v in rows})!=22 or len({v['source_id'] for v in rows})!=66:
        raise ValueError('Incomplete Table S3 reference coverage')
    with args.output.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    print('Validated 66 outside-parentheses values and extracted 66 independent labels.')


if __name__=='__main__':
    main()
