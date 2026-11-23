#!/usr/bin/env python3
"""Validate vibrato ornaments and MusicXML 3.1 schema independently (test-only)."""
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
parts = score.getroot().findall('part')
assert len(parts) == 2, len(parts)
marked = []
for measure in parts[0].findall('measure'):
    for note in measure.findall('note'):
        wavy = note.find("notations/ornaments/wavy-line[@type='start']")
        if wavy is not None:
            marked.append((measure.get('number'), note.findtext('staff'),
                           note.findtext('pitch/step'), note.findtext('pitch/octave'),
                           note.findtext('notations/technical/string'),
                           note.findtext('notations/technical/fret'),
                           note.findtext('type')))
assert marked == [('1', '1', 'A', '2', None, None, 'quarter'),
                  ('1', '2', 'A', '2', '6', '3', 'quarter'),
                  ('2', '1', 'E', '3', None, None, 'quarter'),
                  ('2', '2', 'E', '3', '4', '0', 'quarter')], marked
assert not parts[1].findall("measure/note/notations/ornaments"), 'vibrato leaked into track 2'
print('MusicXML 3.1 XSD and vibrato ornament start markings verified')
