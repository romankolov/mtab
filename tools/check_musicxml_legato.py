#!/usr/bin/env python3
"""Validate hammer-on/slide technical marks and MusicXML 3.1 schema (test-only)."""
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
found = {'hammer-on': [], 'slide-start': [], 'slide-stop': []}
for measure in part.findall('measure'):
    for note in measure.findall('note'):
        staff = note.findtext('staff')
        hammer = note.find('notations/technical/hammer-on')
        slide = note.find('notations/slide')
        if hammer is not None:
            found['hammer-on'].append((measure.get('number'), staff, hammer.get('type')))
        if slide is not None:
            key = 'slide-' + slide.get('type')
            found[key].append((measure.get('number'), staff, slide.get('number')))
assert found == {
    'hammer-on': [('1', '1', 'start'), ('1', '2', 'start')],
    'slide-start': [('1', '1', '1'), ('1', '2', '1')],
    'slide-stop': [('1', '1', '1'), ('1', '2', '1')],
}, found
print('MusicXML 3.1 XSD and hammer/slide technical marks verified')
