"""
EEGNet Multi-class Training Script
- Load .npy preprocessed data
- Sliding window data augmentation
- 5-fold stratified cross-validation
- Output per-fold and average accuracy, F1, etc.
"""

import copy
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score


def classify_files_by_filename_substring(root_dir, target_substring):
    """Recursively search directories and classify files with specified substrings in the file name by extension"""
    classified_files = {}
    if not os.path.isdir(root_dir):
        raise ValueError(f"Invalid directory: {root_dir}")
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if target_substring.lower() in filename.lower():
                file_path = os.path.join(dirpath, filename)
                _, ext = os.path.splitext(filename)
                ext = ext[1:].lower() or "no_extension"
                classified_files.setdefault(ext, []).append(file_path)
    return classified_files


def sort_file_name(string_list, reverse=False):
    return sorted(string_list, reverse=reverse)


# -------------------------- EEGNet 模型 --------------------------
class EEGNetModel(nn.Module):
    def __init__(self, chans=32, classes=5, time_points=1000, temp_kernel=25,
                 f1=8, f2=16, d=2, pk1=16, pk2=8, dropout_rate=0.5, max_norm1=1, max_norm2=1):
        super(EEGNetModel, self).__init__()
        linear_size = (time_points // (pk1 * pk2)) * f2
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, temp_kernel), padding='same', bias=False),
            nn.BatchNorm2d(f1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, pk1)),
            nn.Dropout(dropout_rate)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(d * f1, f2, (1, 16), groups=f2, bias=False, padding='same'),
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2)),
            nn.Dropout(dropout_rate)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(linear_size, classes)
        self.max_norm1 = max_norm1
        self.max_norm2 = max_norm2

    def _apply_max_norm(self):
        for name, param in self.block2[0].named_parameters():
            if 'weight' in name:
                param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=self.max_norm1)
        for name, param in self.fc.named_parameters():
            if 'weight' in name:
                param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=self.max_norm2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def generate_sliding_windows(trials_data, trials_labels, window_sec, stride_sec, sfreq):
    """Generate sliding windows along the time dimension for data augmentation"""
    window_size = int(round(window_sec * sfreq))
    stride = int(round(stride_sec * sfreq))
    if window_size <= 0 or stride <= 0:
        raise ValueError("Window size and stride must be positive")
    n_trials, _, chans, n_timepoints = trials_data.shape
    if window_size > n_timepoints:
        raise ValueError(f"Window points({window_size})exceed trial length({n_timepoints})")
    windows_data = []
    windows_labels = []
    for i in range(n_trials):
        trial_data = trials_data[i]
        trial_label = trials_labels[i]
        max_start = n_timepoints - window_size
        for start in range(0, max_start + 1, stride):
            end = start + window_size
            window = trial_data[:, :, start:end]
            windows_data.append(window)
            windows_labels.append(trial_label)
    windows_data = torch.stack(windows_data)
    windows_labels = torch.tensor(windows_labels, dtype=torch.long)
    return windows_data, windows_labels


class TrainModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train_model(self, model_class=EEGNetModel, dataset=None, model_kwargs=None,
                    learning_rate=0.0001, batch_size=32, epochs=500, n_splits=5,
                    weight_decay=0.09, seed=42, selection_metric='accuracy',
                    early_stopping_patience=100, lr_scheduler_patience=30,
                    window_size=None, stride=None, freqs=None):
        self.set_seed(seed)
        model_kwargs = model_kwargs or {}
        all_trials_data, all_trials_labels = dataset.tensors
        n_trials = len(all_trials_data)
        n_timepoints = all_trials_data.shape[-1]
        if window_size is None:
            window_size = n_timepoints / freqs
        if stride is None:
            stride = window_size
        print(f"\n Sliding window config: window={window_size}s, stride={stride}s, 采样率={freqs}Hz")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        all_fold_results = {
            'best_val_acc': [], 'best_val_loss': [],
            'best_val_f1': [], 'best_val_precision': [], 'best_val_recall': []
        }
        best_overall_acc = 0.0
        best_overall_loss = float('inf')
        best_model_state = None

        labels_np = all_trials_labels.cpu().numpy()
        for fold, (train_idx, val_idx) in enumerate(skf.split(range(n_trials), labels_np)):
            print(f"\n{'='*50}")
            print(f"Fold {fold+1}/{n_splits}")
            print('='*50)

            train_data = all_trials_data[train_idx]
            train_labels = all_trials_labels[train_idx]
            val_data = all_trials_data[val_idx]
            val_labels = all_trials_labels[val_idx]

            mean = train_data.mean(dim=(0, 3), keepdim=True)
            std = train_data.std(dim=(0, 3), keepdim=True) + 1e-8
            train_data = (train_data - mean) / std
            val_data = (val_data - mean) / std

            train_win, train_win_labels = generate_sliding_windows(
                train_data, train_labels, window_size, stride, freqs)
            val_win, val_win_labels = generate_sliding_windows(
                val_data, val_labels, window_size, stride, freqs)

            train_loader = DataLoader(TensorDataset(train_win, train_win_labels),
                                      batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(val_win, val_win_labels),
                                    batch_size=batch_size, shuffle=False)
            print(f"Training windows: {len(train_win)} | Validation window: {len(val_win)}")

            time_len = val_win.shape[-1]
            ch_num = val_win.shape[2]
            class_num = torch.unique(train_win_labels).numel()
            model_kwargs.update({'chans': ch_num, 'classes': class_num, 'time_points': time_len})
            model = model_class(**model_kwargs).to(self.device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=lr_scheduler_patience, verbose=True)

            best_val_acc = 0.0
            best_val_loss = float('inf')
            best_f1 = best_precision = best_recall = 0.0
            early_stop_cnt = 0

            for epoch in range(epochs):
                model.train()
                train_loss, train_correct, train_total = 0.0, 0, 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * inputs.size(0)
                    _, pred = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += (pred == labels).sum().item()

                model.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0
                all_val_labels, all_val_preds = [], []
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * inputs.size(0)
                        _, pred = torch.max(outputs, 1)
                        val_total += labels.size(0)
                        val_correct += (pred == labels).sum().item()
                        all_val_labels.extend(labels.cpu().numpy())
                        all_val_preds.extend(pred.cpu().numpy())

                epoch_val_acc = val_correct / val_total
                epoch_val_loss = val_loss / len(val_loader.dataset)
                val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
                val_precision = precision_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
                val_recall = recall_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)

                if epoch % 30 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | "
                          f"Train Acc: {train_correct/train_total*100:.2f}% | "
                          f"Val Acc: {epoch_val_acc*100:.2f}% | F1: {val_f1:.4f}")

                # 更新最佳
                if epoch_val_acc > best_val_acc:
                    best_val_acc = epoch_val_acc
                    best_val_loss = epoch_val_loss
                    best_f1 = val_f1
                    best_precision = val_precision
                    best_recall = val_recall
                    early_stop_cnt = 0
                    if (selection_metric == 'accuracy' and epoch_val_acc > best_overall_acc) or \
                       (selection_metric == 'loss' and epoch_val_loss < best_overall_loss):
                        best_overall_acc = epoch_val_acc
                        best_overall_loss = epoch_val_loss
                        best_model_state = copy.deepcopy(model.state_dict())
                else:
                    early_stop_cnt += 1
                # if early_stop_cnt >= early_stopping_patience: break

                scheduler.step(epoch_val_acc)

            all_fold_results['best_val_acc'].append(best_val_acc)
            all_fold_results['best_val_loss'].append(best_val_loss)
            all_fold_results['best_val_f1'].append(best_f1)
            all_fold_results['best_val_precision'].append(best_precision)
            all_fold_results['best_val_recall'].append(best_recall)
            print(f"Fold {fold+1} best: Acc={best_val_acc*100:.2f}% | F1={best_f1:.4f}")

        # 汇总
        avg_acc = np.mean(all_fold_results['best_val_acc'])
        std_acc = np.std(all_fold_results['best_val_acc'])
        avg_f1 = np.mean(all_fold_results['best_val_f1'])
        std_f1 = np.std(all_fold_results['best_val_f1'])
        print(f"\n{'='*50}")
        print("5-fold Cross-validation Results")
        print(f"Average Accuracy: {avg_acc*100:.2f}% ± {std_acc*100:.2f}%")
        print(f"Average F1:     {avg_f1:.4f} ± {std_f1:.4f}")
        print(f"Overall Best Accuracy: {best_overall_acc*100:.2f}%")

        best_model = model_class(**model_kwargs).to(self.device)
        best_model.load_state_dict(best_model_state)
        return best_model, all_fold_results


# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    data_list = [r"..\ani_s1_signal_sub_newset_del_ch",
                 ]
    trainer = TrainModel()

    for folder_path in data_list:
        try:
            data_files = sort_file_name(
                classify_files_by_filename_substring(folder_path, "data").get("npy", []))
            label_files = sort_file_name(
                classify_files_by_filename_substring(folder_path, "label").get("npy", []))
        except Exception as e:
            print(f"Failed to read files ({folder_path}): {e}")
            continue

        print(f"\n{folder_path}: Found  {len(data_files)} subjects")

        for i, (d_path, l_path) in enumerate(zip(data_files, label_files)):
            print(f"\n{'='*60}")
            print(f"Subject: {os.path.basename(d_path)}")
            print(f"{'='*60}")

            data = np.load(d_path)
            label = np.load(l_path)
            print(f"Data shape: {data.shape}")

            le = LabelEncoder()
            y_enc = le.fit_transform(label)

            X = torch.Tensor(data).unsqueeze(1)          # (n_trials, 1, chans, time)
            y = torch.LongTensor(y_enc)
            dataset = TensorDataset(X, y)

            _, results = trainer.train_model(
                model_class=EEGNetModel,
                dataset=dataset,
                learning_rate=0.001,
                batch_size=64,
                epochs=1500,
                n_splits=5,
                weight_decay=0.09,
                stride=None,
                window_size=None,
                freqs=250
            )

    print("\n Program finished")