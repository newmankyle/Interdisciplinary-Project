from music21 import *

#allBach = corpus.search('bach')

score = converter.parseFile(r'.\Bach_BWV849-02_001_20090916-SMD.mid')
#p = score.parse()

partStream = score.parts.stream()

for n in score.flat.notes:
    if(n.isChord):
        print( "Midi: " + str(n.pitches) + " Offset: " + str(n.offset) + " EndTime: " + str(n.offset + n.duration.quarterLength))
    else:
        print( "Midi: " + str(n.pitch.midi) + " Offset: " + str(n.offset) + " EndTime: " + str(n.offset + n.duration.quarterLength))
    
    
# WORKS so far. Current problem. For some reason we have chord objects. We don't want chord objects.
# Currently we can tell when a chord object has been introduced, however we don't want that. Must find a way to break down chords
# into notes before printing.