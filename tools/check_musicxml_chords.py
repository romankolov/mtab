#!/usr/bin/env python3
"""Validate chord symbols and MusicXML 3.1 schema (test-only)."""
import pathlib
import sys
from lxml import etree

schema_dir = pathlib.Path(__file__).resolve().parent / 'musicxml-3.1-xsd'


class LocalResolver(etree.Resolver):
    def resolve(self, url, public_id, context):
        name = url.rsplit('/', 1)[-1]
        if name in ('xlink.xsd', 'xml.xsd'):
            return self.resolve_filename(str(schema_dir / name), context)
        return None


parser = etree.XMLParser(no_network=True, resolve_entities=False)
parser.resolvers.add(LocalResolver())
schema = etree.XMLSchema(etree.parse(str(schema_dir / 'musicxml.xsd'), parser))
score = etree.parse(sys.argv[1], parser)
schema.assertValid(score)

part = score.getroot().findall('part')[0]
found = []
for measure in part.findall('measure'):
    for harmony in measure.findall('harmony'):
        step = harmony.findtext('root/root-step')
        alter = harmony.findtext('root/root-alter')
        if alter is not None:
            step += alter
        found.append((measure.get('number'), step, harmony.findtext('kind'),
                      harmony.findtext('staff')))
# Chords appear once, on the first staff, only in the measure they start in.
assert found == [
    ('1', 'E', 'minor-seventh', '1'),
    ('1', 'C', 'major', '1'),
    ('2', 'G', 'major', '1'),
], found
print('MusicXML 3.1 XSD and chord symbols verified')
