#!/usr/bin/env python3
"""Independently check GP5 note-effect flag bits and payload alignment."""
import sys
import guitarpro

song = guitarpro.parse(sys.argv[1])
notes = [n for measure in song.tracks[0].measures for beat in measure.voices[0].beats for n in beat.notes]
assert len(notes) == 8, len(notes)
effects = [n.effect for n in notes]
assert effects[0].staccato and not effects[0].palmMute
assert effects[1].palmMute and not effects[1].staccato
assert effects[2].tremoloPicking and effects[2].tremoloPicking.duration.value == 16
assert effects[3].slides == [guitarpro.SlideType.shiftSlideTo]
assert isinstance(effects[4].harmonic, guitarpro.NaturalHarmonic)
assert effects[5].trill and effects[5].trill.fret == 4 and effects[5].trill.duration.value == 32
assert effects[6].vibrato
for effect in effects[2:7]:
    assert not effect.staccato and not effect.palmMute
combined = effects[7]
assert combined.palmMute and combined.vibrato and not combined.staccato
assert combined.slides == [guitarpro.SlideType.legatoSlideTo]
assert isinstance(combined.harmonic, guitarpro.NaturalHarmonic)
assert combined.tremoloPicking.duration.value == 8
assert combined.trill.fret == 6 and combined.trill.duration.value == 16
print('PyGuitarPro verified all GP5 second-byte note effect bits and payload alignment')
