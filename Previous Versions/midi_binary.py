from music21 import *
# from music21 import converter
# from music21 import note
# from music21 import stream
# from music21 import timespans

def main():
    score = converter.parseFile(r'.\Bach_BWV849-02_001_20090916-SMD.mid')
    # scoreTree = score.asTimespans()
    # scoreTree = timespans.streamToTimespanTree(score, flatten=true, classList=(note.NOte, chord.Chord))
    #print(scoreTree)
    

    #saParts = sa.parts.stream()

    # for n in scoreTree:
        # print(n.note)
    #can't access notes directly because of the way music21 implements their score-note hierarchy...
    #https://stackoverflow.com/questions/36647054/music21-getting-all-notes-with-durations
    #look up stream class stream.secondsMap
    #look up timespans
    # for n in score.pitches: 
        # print("Note: " + str(n.midi))
        
    # print(score)    
    for n in score.flat.notes: 
        print("Note: " + str(n.pitch.name))
    
if __name__ == "__main__":
    main()