# dir with ESC50 data
esc50_path = 'data/esc50'

runs_path = 'results'
# sub-epoch (batch-level) progress bar display
disable_bat_pbar = False
#disable_bat_pbar = True

# do not change this block
n_classes = 50
folds = 5
test_folds = [1, 2, 3, 4, 5]
# use only first fold for internal testing
#test_folds = [1]

# sampling rate for waves
sr = 44100
n_mels = 128
n_steps = 431
hop_length = 512

model_constructor = "AudioCNN(n_mels=config.n_mels, n_steps=config.n_steps, n_classes=config.n_classes)"

# ###TRAINING
# ratio to split off from training data
val_size = .2  # could be changed
device_id = 0
# in Colab to avoid Warning
num_workers = 2
num_workers = 0
# for local Windows or Linux machine
# num_workers = 6#16
persistent_workers = True
persistent_workers = False
epochs = 75         # statt 50
batch_size = 64     # schneller bei guter GPU
#epochs = 1
# early stopping after epochs with no improvement
patience = 10       # mehr Geduld bei EarlyStopping
lr = 5e-4          # statt 1e-3
weight_decay = 5e-4  # optional leicht reduzieren
step_size = 4
gamma = 0.75       # etwas konservativer decay
warm_epochs = 10
step_size = 4

# ### TESTING
# model checkpoints loaded for testing
test_checkpoints = ['terminal.pt']  # ['terminal.pt', 'best_val_loss.pt']
# experiment folder used for testing (result from cross validation training)
#test_experiment = 'results/2025-04-07-00-00'
test_experiment = 'results/2025-06-01-02-07'
