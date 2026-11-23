#!/usr/bin/env python3
"""Validate pull-off marks and MusicXML 3.1 schema (test-only)."""
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
    for note in measure.findall('note'):
        technical = note.find('notations/technical')
        if technical is None:
            continue
        hammer = technical.find('hammer-on')
        pull = technical.find('pull-off')
        if hammer is None and pull is None:
            continue  # only tab string/fret on the second note
        found.append((
            measure.get('number'),
            note.findtext('staff'),
            None if hammer is None else 'hammer-on',
            None if pull is None else 'pull-off',
            note.findtext('pitch/step'),
            technical.findtext('fret'),
        ))
# Capo 2, string 2 (D3 = 50, sounding 52): A3 at fret 5 down to G3 at fret 3.
# The pull-off mark belongs to the first note only; the second note still
# carries its tab string/fret under <technical>.
assert found == [
    ('1', '1', None, 'pull-off', 'A', None),
    ('1', '2', None, 'pull-off', 'A', '5'),
], found
print('MusicXML 3.1 XSD and pull-off technical marks verified')
