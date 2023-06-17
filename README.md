# Seer - TF2 Port

Seer is a project of mine focused around AI assisted anti-cheats
This is the TF2 version (Not ready!)

## Why
I saw the work done by MegaScatterBomb (https://github.com/megascatterbomb/MegaAntiCheat), and decided to port and open-source my Seer anti-cheat, originally developed for Minecraft.

## I wanna run this NOW!
Please Don't! Seer is a massive machine learning model, I've attempted to scale it down in the TF2 version
Seer utilizes two different binary classifiers which are massive! (1 for player looking and another for player movement)
I'm not sure how MegaScatterBomb wants things done and I'm not sure about TF2 demos either, so use this as a GUIDE and not a ready-to-go solution!

If you choose to run this, Good luck! Training takes days to hours AND uses 10-11gb of memory!

## Notes
Some of the Demo parsing I stole from (MegaAntiCheat)[https://github.com/megascatterbomb/MegaAntiCheat], Sorry!

I wasn't sure how to handle the Steam ID, so if Seer were to be trained, it would expect the Steam ID format to be the same as in the demo files.

## Usage
Seer expects at least 1 command line argument which is 
`--demo=` another after the `=` should be a path to a demo file, it will then open the demo file and then attempt to find cheaters! (this works without training too, but will not work accuratly).

Seer also accepts `--train` which will make Seer train its models, Seer will not train models if it has to create the folders required!.
Seer expects labels in a .toml format (an example can be found in training_data/labels/label1.toml ).