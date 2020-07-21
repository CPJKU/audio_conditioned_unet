## Videos
Videos of our best model trying to track 5 pieces from the test set.
On the left you will see the score image together with the prediction of the model. (Note that the repetitions are not considered in this work)
To the right we visualize the last two seconds of audio. The prediction itself will be based on more than this excerpt (see. Section 3).

- Johann Sebastian Bach - BMV 988 v09: medium pace, tracker has no problem

- Johan Friedrich Franz Burgmüller - L'Arabesque: medium fast pace, tracker makes some errors as shown in Figure 1 in the paper

- Carl Czerny - Op. 821 Eight-Measure Exercise: fast pace, tracker has a harder time to follow

- Wolfgang Amadeus Mozart - KV331 Var 1: medium fast pace, minor confusions of similar notes

- Traditioner af Swenska Folk-Dansar - 3rd Part nr. 22: medium pace, confusion of similar note patterns transposed by half a step in staff 4 and 5. Possibly due to downscaling the score, however a higher resolution increases the overall train time and does not significantly improve the results (see Section 4.1)


