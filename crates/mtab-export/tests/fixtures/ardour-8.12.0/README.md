# Ardour imported-source fixture

`bend-import.mid` is the MIDI source written inside an Ardour 8.12.0 session
when importing the repository's `mtab-export` `midi_bend_fixture` example.
The fixture is synthetic, contains no third-party soundfont or rendered audio,
and is included as reproducible test data. SHA-256:
`85ebb02bb6482ac09beea44bd6e8e471384aa79344db583f665ceaee52eee1fa`.

The generated input MIDI SHA-256 was
`38aba1ec3a054d3c9c58a52fe3162ae1fd28a597ece6114d6905140c973c1997`.
Details, observed RPN loss, and limitations are documented in
[`benches/m4-ardour-midi.md`](../../../../../benches/m4-ardour-midi.md).
This is the saved import source, **not an Ardour MIDI-export round trip**.
