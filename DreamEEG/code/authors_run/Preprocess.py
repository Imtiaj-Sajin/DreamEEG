"""
EEG Signal Processing Script

 Read BDF, extract events at original sampling rate
 Optionally downsample and check event timing alignment
 Bandpass filter, standard montage, average reference
 ICA (Picard extended) + ICLabel automatic artifact rejection
 Interpolate bad channels, narrow-band filter
 Epoch, baseline correct, crop
 Concatenate all trials and save as .npy

"""

import mne
import numpy as np
from mne_icalabel import label_components
import copy
from typing import Dict, Tuple, Optional
import os
from pyprep import NoisyChannels

def read_bdf_and_extract_events(bdf_file_path: str) -> Tuple[mne.io.Raw, np.ndarray]:
    """Load BDF and extract events at original sampling rate"""
    raw = mne.io.read_raw_bdf(bdf_file_path, preload=True)
    events = extract_events_universal(raw)
    return raw, events


def extract_events_universal(raw: mne.io.Raw) -> np.ndarray:
    """Extract discrete events from Status channel"""
    status_data = raw.get_data(picks=['Status'])[0]
    unique_vals, counts = np.unique(status_data, return_counts=True)
    baseline_value = unique_vals[np.argmax(counts)]
    event_indices = np.where(np.isin(status_data, [1, 2, 3]))[0]
    events = []
    for idx in event_indices:
        prev_val = status_data[idx - 1] if idx > 0 else baseline_value
        next_val = status_data[idx + 1] if idx < len(status_data) - 1 else baseline_value
        if prev_val == baseline_value and next_val == baseline_value:
            events.append([idx, 0, int(status_data[idx])])
    return np.array(events)


def build_mne_events(events_arr: np.ndarray) -> np.ndarray:
    """Convert event array to MNE format"""
    mne_events = []
    for event in events_arr:
        sample_idx = int(event[0])
        event_type = int(event[2])
        mne_events.append([sample_idx, 0, event_type])
    return np.array(mne_events, dtype=int)


def preprocess_eeg_data(raw: mne.io.Raw,
                        low_freq: float = 1.0,
                        high_freq: float = 100.0,
                        montage_name: str = 'standard_1020',
                        reference: str = "average") -> mne.io.Raw:
    """Montage, bad channel detection, bandpass filter, re-reference"""
    print("Applying standard montage...")
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage)
    raw.info["bads"] = [] 
    noise_res=NoisyChannels(raw)
    noise_res.find_all_bads()
    raw.info["bads"] = noise_res.get_bads()
    print(f"Bad channels detected: {raw.info['bads']}")
    print(f"Applying bandpass filter ({low_freq}-{high_freq} Hz) before ICA...")
    raw.notch_filter(freqs=50)
    raw.filter(
        l_freq=low_freq,
        h_freq=high_freq,
        method='fir',
        fir_window='hamming',
        phase='zero-double',
        pad='edge',
        verbose=False
    )
    print(f"Applying {reference} reference...")
    if reference == "average":
        raw = raw.set_eeg_reference(ref_channels="average")
    else:
        raw = raw.set_eeg_reference(ref_channels=reference)
    return raw


def remove_artifacts_ica(raw: mne.io.Raw,
                         n_components: float = 0.999999,
                         low_freq: float = 4.0,
                         high_freq: float = 80.0,
                         max_iter: int = 300,
                         random_state: int = 42) -> Tuple[mne.io.Raw, mne.preprocessing.ICA]:
    """ICA (Picard extended) + ICLabel artifact rejection, then interpolate bad channels and narrow-band filter"""
    print("Fitting ICA for artifact removal (picard extended)...")
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method='picard',
        fit_params=dict(ortho=False, extended=True),
        random_state=random_state,
        max_iter=max_iter,
        verbose=False
    )
    ica.fit(raw)
    print("Labeling ICA components...")
    ica_labels = label_components(raw, ica, method="iclabel")
    bad_labels = ica_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(bad_labels)
                   if label not in ["brain", "other"]]
    print(f"Excluding {len(exclude_idx)} non-brain components: {exclude_idx}")
    cleaned_raw = ica.apply(raw, exclude=exclude_idx)
    print(f"Post-ICA bandpass filter ({low_freq}-{high_freq} Hz)")
    
    if cleaned_raw.info["bads"]:
                print(f"Interpolating bad channels: {cleaned_raw.info['bads']}")
                cleaned_raw.interpolate_bads(reset_bads=True, verbose=False)
    else:
                print("No bad channels to interpolate.")
    cleaned_raw.filter(
        l_freq=low_freq,
        h_freq=high_freq,
        method='fir',
        fir_window='hamming',
        phase='zero-double',
        pad='edge',
        verbose=False
    )
    return cleaned_raw, ica


def split_raw_by_custom_events(raw: mne.io.Raw,
                               mne_events: np.ndarray,
                               tmin: float = -0.2,
                               tmax: float = 4.0,
                               baseline: Tuple[float, float] = (-0.2, 0)) -> Tuple[dict, dict, mne.Epochs]:
    """Epoch data based on events, baseline correct, then crop"""
    if mne_events.shape[1] != 3:
        raise ValueError(f"Event array should be in (n_events, 3) format, got {mne_events.shape}")
    if not np.issubdtype(mne_events.dtype, np.integer):
        print(f"Warning: Event array data type is {mne_events.dtype}, converting to integer")
        mne_events = mne_events.astype(int)
    event_times = mne_events[:, 0]
    raw_n_times = len(raw.times)
    valid_mask = (event_times >= 0) & (event_times < raw_n_times)
    valid_events = mne_events[valid_mask]
    invalid_count = len(mne_events) - len(valid_events)
    if invalid_count > 0:
        print(f"Warning: {invalid_count} events are outside raw data time range, filtered out")
        mne_events = valid_events
    event_id = {
        "dog": 1,
        "bird": 2,
        "fish": 3,
    }

    print(f"Event ID mapping used: {event_id}")
    epochs = mne.Epochs(
        raw,
        events=mne_events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose=False,
        on_missing='warn'
    )
    print(f"\nEvents actually included in Epochs: {epochs.event_id}")
    print(f"Total valid trials after epoching: {len(epochs)}")
    epochs.crop(tmin=0)
    split_data = {}
    for type_name, type_id in event_id.items():
        if type_name in epochs.event_id:
            split_data[type_name] = epochs[type_name].copy()
            print(f"\nEvent type {type_name} (ID: {type_id}):")
            print(f"  - Number of segments: {len(split_data[type_name])}")
            print(f"  - Data shape: {split_data[type_name].get_data().shape}")
        else:
            print(f"Warning: Event type {type_name} (ID: {type_id}) missing in Epochs")
            split_data[type_name] = None
    return split_data, event_id, epochs


def full_preprocessing_pipeline(bdf_file_path: str,
                                low_freq: float = 4.0,
                                high_freq: float = 80.0,
                                montage_name: str = 'standard_1020',
                                reference: str = "average",
                                ica_n_components: float = 0.999999,
                                downsample_freq: Optional[float] = None) -> Tuple[mne.io.Raw, np.ndarray, dict, dict, mne.Epochs]:
    """Run complete preprocessing from BDF to epoched data"""
    print("===== Start Full EEG Preprocessing Pipeline =====")
    print("\n[Step 1] Read BDF & extract events on original high-sampling raw")
    eeg_raw_orig, _ = read_bdf_and_extract_events(bdf_file_path)
    orig_sfreq = eeg_raw_orig.info["sfreq"]
    print(f"Original sampling rate: {orig_sfreq} Hz")
    events_original = extract_events_universal(eeg_raw_orig)
    print(f"Original events count: {len(events_original)}")
    orig_event_seconds = events_original[:, 0] / orig_sfreq
    events_arr = events_original
    eeg_raw = eeg_raw_orig.copy()
    data = eeg_raw.get_data()
    data *= 1e-6
    eeg_raw._data = data
    if downsample_freq is not None and downsample_freq < orig_sfreq:
        print(f"\n[Step 2] Resample raw {orig_sfreq}Hz → {downsample_freq}Hz, auto sync events")
        eeg_raw, events_arr = eeg_raw.resample(
            sfreq=downsample_freq,
            events=events_original,
            verbose=False
        )
        new_sfreq = eeg_raw.info["sfreq"]
        new_event_seconds = events_arr[:, 0] / new_sfreq
        time_diff = np.abs(orig_event_seconds - new_event_seconds)
        max_time_err = np.max(time_diff)
        mean_time_err = np.mean(time_diff)
        print("\n===== Resample Event Time Alignment Check =====")
        print(f"Max time offset between original & resampled events: {max_time_err:.6f} s")
        print(f"Mean time offset between original & resampled events: {mean_time_err:.6f} s")
        if max_time_err < 0.005:
            print(" Event timing match perfectly after resample, no misalignment!")
        else:
            print(" WARNING: Large timing offset between events, check data!")
        print("==============================================\n")
    else:
        new_sfreq = orig_sfreq
        print("\n[Step 2] Skip downsampling, keep original sampling rate")
    print(f"Sampling rate after resample: {new_sfreq} Hz")
    print(f"Events count after resample: {len(events_arr)}")
    print("\n[Step 3] Montage + Average Reference + 1-100Hz Bandpass")
    eeg_raw = preprocess_eeg_data(
        raw=eeg_raw,
        low_freq=1.0,
        high_freq=100.0,
        montage_name=montage_name,
        reference=reference
    )
    print("\n[Step 4] ICA artifact removal + post-ICA filter")
    cleaned_raw, ica = remove_artifacts_ica(
        eeg_raw,
        n_components=ica_n_components,
        low_freq=low_freq,
        high_freq=high_freq
    )
    print("\n[Step 5] Build MNE standard event array")
    mne_events = build_mne_events(events_arr=events_arr)
    print(f"MNE events array shape: {mne_events.shape}")
    print("\n[Step 6] Epoching based on synced events")
    split_data, event_id, epochs = split_raw_by_custom_events(copy.copy(cleaned_raw), mne_events)
    print("\n===== Preprocessing Pipeline Finished Successfully =====")
    return cleaned_raw, mne_events, split_data, event_id, epochs


def generate_data_label(input_data):
    """Merge all categories into (n_trials, n_channels, n_times) and label array"""
    all_data = []
    all_labels_id = []
    for label_name, epochs_data in input_data.items():
        if epochs_data is not None:
            data = epochs_data.get_data()
            labels_id = epochs_data.events[:, 2]
            all_data.append(data)
            all_labels_id.append(labels_id)
    all_data = np.concatenate(all_data, axis=0)
    all_labels_id = np.concatenate(all_labels_id, axis=0)
    return all_data, all_labels_id


def extract_core_name(file_path: str) -> str:
    """Extract filename without extension"""
    basename = os.path.basename(file_path)
    return basename.replace(".bdf", "")



if __name__ == "__main__":
    # # List of BDF files (example paths)
    bdf_file_path = [
        r"g:/codes/Ass/ai msc course/DreamEEG/data/sub-09/sub-09/ses-01/eeg/sub-09_ses-01_task-AVI_eeg.bdf",
        # ... add remaining subjects
    ]

    target_downsample = 250

    for file_path in bdf_file_path:
        print(f"\n==================== Processing {file_path} ====================")
        try:
            cleaned_raw, mne_events, split_data, event_id, epochs = full_preprocessing_pipeline(
                bdf_file_path=file_path,
                low_freq=4.0,
                high_freq=80.0,
                montage_name='standard_1020',
                reference="average",
                ica_n_components=0.999999,
                downsample_freq=target_downsample
            )

            data, label = generate_data_label(split_data)
            train_data_epoch = np.array(data)[:,:32,:1000]
            label = np.array(label)
            print(f"\nFinal epoch data shape: {train_data_epoch.shape}")

            # Save to directory
            core_name = extract_core_name(file_path)
            save_dir = "ani_s1_signal_sub_newset_del_ch"
            os.makedirs(save_dir, exist_ok=True)
            data_name = os.path.join(save_dir, f"{core_name}_data.npy")
            label_name = os.path.join(save_dir, f"{core_name}_label.npy")
            np.save(data_name, train_data_epoch)
            np.save(label_name, label)
            print(f"Saved data: {data_name}")
            print(f"Saved label: {label_name}")

        except Exception as e:
            print(f"Error processing {file_path}, skip. Detail: {str(e)}")
            continue