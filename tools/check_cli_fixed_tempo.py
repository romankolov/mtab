#!/usr/bin/env python3
"""Independent Mido/MusicXML/ASCII check of the CLI's 96 BPM fixture.

Development/CI only. PyGuitarPro reconstruction is checked separately by
check_gp5_cli_ir.py. The fixture is synthetic and does not use a model.
"""
import json
import sys
import xml.etree.ElementTree as ET
import mido

midi_path, xml_path, ascii_path, ir_path = sys.argv[1:]
with open(ir_path, encoding='utf-8') as source:
    doc = json.load(source)
assert doc['meta']['tempo_map']['changes'] == [{'time': 0.0, 'bpm': 96.0}]
midi = mido.MidiFile(midi_path)
assert midi.type == 1 and midi.ticks_per_beat == 480
notes, tempos, meters = [], [], []
for track in midi.tracks:
    tick = 0
    for message in track:
        tick += message.time
        if message.type == 'set_tempo':
            tempos.append((tick, message.tempo))
        elif message.type == 'time_signature':
            meters.append((tick, message.numerator, message.denominator))
        elif message.type in ('note_on', 'note_off'):
            notes.append((tick, message.type, message.note, message.velocity))
assert tempos == [(0, 625000)], tempos
assert meters == [(0, 4, 4)], meters
assert sorted(notes) == sorted([(0, 'note_on', 40, 81), (240, 'note_off', 40, 0),
                                (480, 'note_on', 45, 104), (960, 'note_off', 45, 0)]), notes
root = ET.parse(xml_path).getroot()
assert root.findall('.//sound') and all(node.get('tempo') == '96' for node in root.findall('.//sound'))
assert root.findall('.//per-minute') and all(node.text == '96' for node in root.findall('.//per-minute'))
with open(ascii_path, encoding='utf-8') as source:
    assert '96 BPM | 4/4 | 1/32 grid' in source.read()
print('Mido, MusicXML, ASCII and IR agree at 96 BPM; MIDI note times and velocities match')
