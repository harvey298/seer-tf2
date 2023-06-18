# Seer - TF2 Port

Seer is a project of mine focused around AI assisted anti-cheats
This is the TF2 version (Not ready!)

I aim with Seer TF2 to get it as close to the Minecraft version as possible with new features and optimisations.
Seer-Minecraft will remain private though.

## Why
I saw the work done by [MegaScatterBomb](https://github.com/megascatterbomb), and decided to port and open-source my Seer anti-cheat, originally developed for Minecraft.
This is still a proof-of-concept

## I wanna run this NOW!
Please Don't! Seer is a massive machine learning model, I've attempted to scale it down in the TF2 version
Seer utilizes two different binary classifiers which are massive! (1 for player looking and another for player movement)
I'm not sure how MegaScatterBomb wants things done and I'm not sure about TF2 demos either, so use this as a GUIDE and not a ready-to-go solution!

It is runable in its current state, its not very accurate with its current dataset, but it works for now

## Notes
Some of the Demo parsing I stole from [MegaAntiCheat](https://github.com/megascatterbomb/MegaAntiCheat), Sorry!


## .replay files
Seer will generate .replay files to be able to quickly recreate data required (as parsing demo files is expensive), Replays will generate a new UUID for each player.
if a player appears in 2 demo files, the replay files that will create will contain 2 different UUIDs

Replay files contain almost all the information in a `label.toml`, but `label.toml` are required to allow Seer to use them

## Usage
Seer expects at least 1 command line argument which is 
`--demo=` another after the `=` should be a path to a demo file, it will then open the demo file and then attempt to find cheaters! (this works without training too, but will not work accuratly).

Seer also accepts `--train` which will make Seer train its models, Seer will not train models if it has to create the folders required!.
Seer expects labels in a .toml format (an example can be found in `examples/label.toml` ).