#!/usr/bin/env python3
"""Validate let-ring semantics and MusicXML 3.1 schema independently."""
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
assert len(parts) == 2
for part_index, part in enumerate(parts):
    marked = []
    for measure in part.findall('measure'):
        for note in measure.findall('note'):
            if note.find("notations/tied[@type='let-ring']") is not None:
                marked.append((measure.get('number'), note.findtext('staff'), note.findtext('pitch/step'),
                               note.findtext('notations/technical/string'), note.findtext('notations/technical/fret')))
    if part_index == 0:
        assert marked == [('1', '1', 'C', None, None), ('1', '2', 'C', '5', '2'),
                          ('2', '1', 'A', None, None), ('2', '2', 'A', '6', '3')], marked
    else:
        assert not marked, marked
print('MusicXML 3.1 XSD and final-segment let-ring markings verified')
