#!/usr/bin/env python3
"""Validate arbitrary MusicXML against the bundled MusicXML 3.1 schema."""
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
schema.assertValid(etree.parse(sys.argv[1], parser))
print('MusicXML 3.1 XSD valid')
