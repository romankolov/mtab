#!/usr/bin/env python3
"""Validate the synthetic fixture against MusicXML 3.1 XSD and expected score."""
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
root = score.getroot()
assert root.tag == 'score-partwise' and root.get('version') == '3.1'
assert root.findtext('work/work-title') == 'Café & <guitar> \'one\' "two"'
parts = root.findall('part')
assert len(parts) == 2 and [len(p.findall('measure')) for p in parts] == [2, 2]
assert parts[0].findtext('measure/attributes/staff-details/capo') == '2'
assert parts[1].findtext('measure/attributes/staff-details/staff-lines') == '7'
for part in parts:
    for measure in part.findall('measure'):
        for staff in ('1', '2'):
            notes = [n for n in measure.findall('note') if n.findtext('staff') == staff]
            total = sum(int(n.findtext('duration')) for n in notes if n.find('chord') is None)
            assert total == 3840, (staff, total)
        assert measure.findtext('backup/duration') == '3840'
notes = parts[0].findall('measure/note')
assert any(n.find('chord') is not None for n in notes)
assert any(n.find("tie[@type='start']") is not None for n in notes)
assert any(n.find("tie[@type='stop']") is not None for n in parts[0].findall('measure/note'))
assert any(n.findtext('notations/technical/other-technical') == 'palm mute' for n in notes)
assert any(n.findtext('notehead') == 'x' for n in notes)
assert any(n.findtext('notations/technical/string') == '6' and n.findtext('notations/technical/fret') == '3' for n in notes)
assert any(n.findtext('notations/technical/string') == '7' and n.findtext('notations/technical/fret') == '1' for n in parts[1].findall('measure/note'))
print('MusicXML 3.1 XSD and independent score semantics verified')
