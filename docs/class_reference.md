# Species reference

## Where the labels come from

The Watkins release used here ships an explicit, unambiguous `class_id`
-> `display_name` / `scientific_name` mapping as part of the dataset
itself. `watkins.data.CLASS_INFO` is a direct copy of that mapping (with
two small corrections, noted below) -- no inference needed.

Source: the [Watkins Marine Mammal Sound
Database](https://cis.whoi.edu/science/B/whalesounds/) (Woods Hole
Oceanographic Institution), via the `ivangtorre/watkins-marine-mammal-full-cuts`
Hugging Face re-release (WHOI's own site has been down; this release
recovered the audio from Internet Archive snapshots). See the README for
full citations.

Two metadata corrections made to the upstream `scientific_name` field,
from published species facts, not guesswork: `HarbourSeal` (id 45) was
copied from `BottlenoseDolphin`'s row upstream (`Tursiops truncatus`) --
corrected to `Phoca vitulina`. `WestIndianManatee` (id 4) was blank
upstream -- filled in as `Trichechus manatus`.

## Severe, real long-tail imbalance

54 species, 15,248 clips total, ranging from 2,647 clips (killer whale)
down to a single clip (harbour seal). This is a genuine property of a
~70-year historical archive built opportunistically (some species were
simply recorded far more often than others, often because a specific
individual -- e.g. the captive orca Keiko, who alone accounts for most of
the killer whale clips -- was recorded repeatedly), not a preprocessing
artifact.

`tapes` below is the number of distinct original recordings each species
was cut from (`record_number`'s first 6 characters -- see `data.py`).
Species with few tapes have little acoustic variety to learn from
regardless of clip count, and species with exactly 1 tape have no
independent recording to hold out for val/test at all (`watkins.data.
build_split` puts those entirely in train -- see `Split.
train_only_classes`).

| class_id | species | clips | tapes |
|---:|---|---:|---:|
| 22 | KillerWhale | 2647 | 65 |
| 51 | SpermWhale | 1379 | 63 |
| 39 | Long_FinnedPilotWhale | 1104 | 18 |
| 53 | PantropicalSpottedDolphin | 1025 | 10 |
| 36 | CommonDolphin | 884 | 9 |
| 1 | StripedDolphin | 681 | 10 |
| 23 | Short_Finned(Pacific)PilotWhale | 607 | 13 |
| 12 | HumpbackWhale | 604 | 14 |
| 0 | Fin_FinbackWhale | 580 | 41 |
| 5 | White_sidedDolphin | 560 | 7 |
| 46 | SpinnerDolphin | 524 | 12 |
| 26 | FalseKillerWhale | 508 | 4 |
| 19 | NorthernRightWhale | 486 | 22 |
| 43 | BowheadWhale | 406 | 6 |
| 6 | Grampus_Risso'sDolphin | 359 | 8 |
| 40 | ClymeneDolphin | 328 | 3 |
| 21 | Walrus | 273 | 11 |
| 20 | AtlanticSpottedDolphin | 244 | 3 |
| 41 | Fraser'sDolphin | 199 | 4 |
| 16 | White_beakedDolphin | 196 | 9 |
| 14 | BottlenoseDolphin | 183 | 7 |
| 47 | Beluga_WhiteWhale | 150 | 12 |
| 13 | RossSeal | 149 | 9 |
| 10 | MelonHeadedWhale | 148 | 2 |
| 11 | WeddellSeal | 133 | 11 |
| 17 | Rough_ToothedDolphin | 98 | 2 |
| 3 | Narwhal | 86 | 6 |
| 4 | WestIndianManatee | 78 | 4 |
| 8 | LongBeaked(Pacific)CommonDolphin | 69 | 2 |
| 32 | BeardedSeal | 63 | 7 |
| 9 | Dall'sPorpoise | 52 | 1 |
| 28 | SouthernRightWhale | 49 | 2 |
| 30 | HarpSeal | 47 | 3 |
| 18 | HarborPorpoise | 46 | 3 |
| 27 | RingedSeal | 46 | 5 |
| 25 | RibbonSeal | 45 | 5 |
| 37 | GrayWhale | 36 | 3 |
| 29 | Boutu_AmazonRiverDolphin | 30 | 2 |
| 52 | DuskyDolphin | 27 | 1 |
| 34 | MinkeWhale | 24 | 3 |
| 35 | SpottedSeal | 22 | 3 |
| 42 | LeopardSeal | 15 | 2 |
| 24 | Heaviside'sDolphin | 14 | 1 |
| 7 | TucuxiDolphin | 12 | 1 |
| 48 | GraySeal | 7 | 1 |
| 15 | StellerSeaLion | 6 | 1 |
| 33 | IrawaddyDolphin | 5 | 1 |
| 31 | JuanFernandezFurSeal | 4 | 1 |
| 2 | SeaOtter | 2 | 1 |
| 38 | FinlessPorpoise | 2 | 1 |
| 49 | NewZealandFurSeal | 2 | 1 |
| 44 | HoodedSeal | 2 | 1 |
| 45 | HarbourSeal | 1 | 1 |
| 50 | Commerson'sDolphin | 1 | 1 |

14 species (the last 14 rows above) have exactly 1 tape and are therefore
train-only in every split this project builds. Realistic expectations for
this project's models: near-perfect recall on killer whale and sperm
whale is not impressive (they dominate the training set by clip count and
tape count both); a model that also does reasonably on, say, walrus (273
clips, 11 tapes -- genuinely varied) is doing something more interesting
than one that doesn't.

## Recording heterogeneity

Beyond class imbalance, the raw archive is heterogeneous: 47 distinct
native sample rates (320Hz-192kHz) and clip durations
from 16ms to over 24 minutes, reflecting seven decades of different
recording equipment. `watkins.prepare_data` resamples everything to a
common 16kHz and caps stored duration at 60s; `watkins.data.
WatkinsDataset` further crops/pads every clip to a fixed 3.0s window at
load time. See `features.py`'s module docstring for the consequence worth
remembering throughout this project: resampling to 16kHz throws away
ultrasonic content (echolocation clicks, high whistles) that's often the
most diagnostic signal for odontocetes (toothed whales, dolphins,
porpoises) specifically -- baleen whale calls mostly survive the cut fine.
