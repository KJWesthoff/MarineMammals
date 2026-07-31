# Next steps: where to take this project further

This curriculum covers spectrogram preprocessing, CNN vs. transformer
architectures, transfer learning, severe class imbalance, and noise
robustness -- most of the standard toolkit for a bioacoustic
classification problem. Real research (and real deployed monitoring
systems) go further in a few directions. Roughly ordered from "do this
next" to "significant additional project":

## Close to what's already built

- **Full AST fine-tuning at scale.** `configs/ast_finetune.yaml` is
  wired up but deliberately scoped down for CPU. On a real GPU (a free
  Colab/Kaggle T4 is enough for AST-base), unfreeze the whole backbone,
  use the full dataset, and compare against the linear-probe numbers in
  notebook 04.
- **Trim AST's input length.** The pretrained checkpoint expects
  ~10.24s inputs; our clips are 3.0s, so roughly 70% of every input is
  zero-padding, wasting positional embedding range. Reduce
  `ASTFeatureExtractor`'s `max_length` and interpolate the pretrained
  positional embeddings down to match (HuggingFace's AST implementation
  supports resizing positional embeddings) -- a genuinely useful
  architecture-adaptation exercise.
- **A higher-sample-rate variant for odontocetes.** This project
  resamples everything to 16kHz (see `docs/class_reference.md`), which
  discards echolocation clicks and high whistles above 8kHz -- often the
  most diagnostic content for dolphins/porpoises specifically. Re-run
  `watkins.prepare_data` with a higher `SAMPLE_RATE` (e.g. 48kHz) and
  compare per-species F1 for odontocete species before/after -- a direct,
  measurable test of how much that design choice actually costs.
- **Better noise realism.** `watkins/augment.py` supports white and pink
  synthetic noise. Real ambient ocean noise has more structure -- try
  mixing in real clips from a *different* species as an interferer
  (via `augment.mix_at_snr` directly) to simulate a busier acoustic
  scene with multiple calling animals, which is closer to how real
  hydrophone recordings actually look.

## Bigger extensions

- **Raw-waveform models.** Every model here consumes a spectrogram.
  Architectures like SincNet or wav2vec 2.0 learn directly from the raw
  waveform, with the first layer effectively learning its own
  filterbank rather than using a fixed STFT. Comparing a raw-waveform
  model against the fixed log-mel front end used throughout this project
  is a natural extension of the "compare architectures" theme -- and
  would sidestep the 16kHz information loss noted above entirely.
- **Self-supervised pretraining on unlabeled hydrophone audio.** AST and
  the CNN backbones here transfer from AudioSet/ImageNet -- neither is
  underwater-domain audio. A self-supervised pretraining pass (masked
  spectrogram modeling, contrastive learning) on unlabeled bioacoustic
  audio before fine-tuning on Watkins' labels could close some of that
  domain gap.
- **Use the metadata this project currently ignores.** The source
  dataset carries `observation_date`, `location` (with lat/lon for many
  recordings), and partial `animal` metadata (individual ID, sex, age --
  populated for a minority of rows, e.g. the captive orca "Keiko"
  appears by name). Stratifying evaluation by decade or ocean region, or
  building an individual-identification task for the subset with animal
  IDs, are both realistic bioacoustics research questions this project
  doesn't currently touch.
- **Few-shot / rare-species classification.** 14 species in this dataset
  have a single source recording; several more have single-digit clip
  counts. Standard supervised training (what this project does) can't do
  much for them. Few-shot learning approaches (prototypical networks,
  fine-tuning a frozen embedding on a handful of examples) are the more
  appropriate tool, and this dataset's long tail is a realistic testbed
  for exactly that.
- **Multi-label / multi-species scenes.** Watkins clips are single-source
  by construction (curated "best of" cuts). Real hydrophone recordings
  often capture multiple animals -- sometimes multiple species --
  simultaneously. Synthetically mixing two labeled clips at varying SNR
  and training a multi-label classifier (or a source-separation front
  end) is a realistic next problem.
- **Streaming / online inference.** Every model here classifies a fixed
  3-second window. A deployed monitoring buoy needs to run continuously
  on a stream, decide when enough evidence has accumulated, and handle
  long silent/no-animal-present periods gracefully.
- **Explainability.** Grad-CAM (for the CNNs) or attention-rollout (for
  AST) applied to a spectrogram input can highlight which time-frequency
  regions drove a prediction -- compare that against the tonal structure
  visible in the LOFARgram view from notebook 01, and check whether the
  model is actually attending to the animal's call or to some
  recording-artifact shortcut (tape hiss level, background hum) instead.

## A meta-exercise worth doing regardless of direction

Whatever you build next, think carefully about how you split data before
trusting any accuracy number. `watkins.data.build_split` groups clips by
original recording (`tape_id`) specifically so a model can't partly
"recognize the recording" instead of "recognizing the species" -- but any
new data you add, any new augmentation you apply before splitting, or any
feature you cache across a split boundary can quietly reintroduce that
same problem. Treat "what's my train/test independence story" as a
standing question for every new experiment in this project.
