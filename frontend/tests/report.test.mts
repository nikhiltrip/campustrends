import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {validateReport} from '../src/report.ts';
const sample=JSON.parse(readFileSync(new URL('../public/demo-report.json',import.meta.url),'utf8'));
test('accepts a pipeline report without losing its sample provenance',()=>{
 const report=validateReport(sample);
 assert.equal(report.is_sample,true);
 assert.equal(report.summary.total_posts,report.posts.length);
});
test('rejects incomplete files before rendering charts',()=>{
 assert.throws(()=>validateReport({schema_version:1,posts:[]}));
 const altered=structuredClone(sample);altered.evaluation.splits=[];
 assert.throws(()=>validateReport(altered));
});
test('rejects invalid nested metrics and unknown topic references',()=>{
 const altered=structuredClone(sample);altered.posts[0].topic_id=97;
 assert.throws(()=>validateReport(altered));
 const invalid=structuredClone(sample);invalid.evaluation.test.roc_auc=9;
 assert.throws(()=>validateReport(invalid));
});
test('rejects inconsistent dataset summaries',()=>{
 const altered=structuredClone(sample);altered.summary.total_posts=1;
 assert.throws(()=>validateReport(altered));
});
