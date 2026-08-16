"""
EEG Multi-class Classification Script
=================================================
- Load preprocessed .npy data
- Feature extraction: Multi-band CSP , global stats , frequency features
- Classifier: KNN with grid search
- 5-fold stratified cross-validation, per-subject best results output
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, ParameterGrid
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
import warnings
import mne
from mne.decoding import CSP
from mne.filter import filter_data
from scipy.stats import skew, kurtosis
from joblib import Parallel, delayed
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
mne.set_log_level("WARNING")


class MultiBandCSP(BaseEstimator, TransformerMixin):
    """Extract CSP features from multiple frequency bands and concatenate"""
    def __init__(self, bands, sfreq, n_components=4, n_jobs=1, reg="oas", log=True, norm_trace=True):
        self.bands = bands
        self.sfreq = sfreq
        self.n_components = n_components
        self.n_jobs = n_jobs
        self.reg = reg
        self.log = log
        self.norm_trace = norm_trace

    def _make_csp(self):
        """Create a CSP object"""
        return CSP(
            n_components=self.n_components,
            reg=self.reg,
            log=self.log,
            norm_trace=self.norm_trace
        )

    def _validate_params(self, X):
        """Check input dimensions and band validity"""
        if X.ndim != 3:
            raise ValueError(f"X must be 3D, got {X.ndim}D")
        nyquist = self.sfreq / 2
        for l_freq, h_freq in self.bands:
            if l_freq >= h_freq:
                raise ValueError(f"Invalid band ({l_freq}, {h_freq})")
            if h_freq >= nyquist:
                raise ValueError(f"{h_freq}Hz exceeds Nyquist {nyquist}Hz")

    def _filter(self, X, band):
        """Bandpass filter the data"""
        l_freq, h_freq = band
        return filter_data(X, self.sfreq, l_freq, h_freq, verbose=False)

    def _fit_band(self, X, y, band):
        """Fit a CSP model for a single band"""
        csp = self._make_csp()
        csp.fit(self._filter(X, band), y)
        return csp

    def _transform_band(self, X, csp, band):
        """Apply a trained CSP model to a single band"""
        return csp.transform(self._filter(X, band))

    def fit(self, X, y):
        """Fit CSP models for all bands in parallel"""
        self._validate_params(X)
        self.csp_models_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_band)(X, y, band) for band in self.bands
        )
        return self

    def fit_transform(self, X, y):
        self._validate_params(X)
        features = []
        self.csp_models_ = []
        for band in self.bands:
            X_filtered = self._filter(X, band)
            csp = self._make_csp()
            csp.fit(X_filtered, y)
            self.csp_models_.append(csp)
            features.append(csp.transform(X_filtered))
        return np.hstack(features)

    def transform(self, X):
        check_is_fitted(self)
        features = Parallel(n_jobs=self.n_jobs)(
            delayed(self._transform_band)(X, csp, band)
            for csp, band in zip(self.csp_models_, self.bands)
        )
        return np.hstack(features)


class EEGStatisticalFeatures_tf(BaseEstimator, TransformerMixin):
    """Extract global statistics"""
    def __init__(
            self,
            sfreq: float = 250.0,
            use_global_stats: bool = True,
            use_sliding_window: bool = False,
            window_size: int = 50,
            step_size: int = 25,
            use_freq_features: bool = True,
            freq_bands: dict = None,
            psd_method: str = 'welch',
            psd_params: dict = None
    ):
        self.sfreq = sfreq
        self.use_global_stats = use_global_stats
        self.use_sliding_window = use_sliding_window
        self.window_size = window_size
        self.step_size = step_size
        self.use_freq_features = use_freq_features
        self.freq_bands = freq_bands if freq_bands is not None else {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 80)
        }
        self.psd_method = psd_method
        self.psd_params = psd_params if psd_params is not None else {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_trials, n_channels, n_times = X.shape
        feat_list = []

        # 1. 全局时域统计量（全向量化）
        if self.use_global_stats:
            mean_feat = np.mean(X, axis=-1)
            std_feat = np.std(X, axis=-1)
            skew_feat = skew(X, axis=-1)
            kurt_feat = kurtosis(X, axis=-1)
            global_feat = np.stack([mean_feat, std_feat, skew_feat, kurt_feat], axis=-1)
            feat_list.append(global_feat.reshape(n_trials, -1))

        # 2. 滑动窗口统计（向量化窗口）
        if self.use_sliding_window:
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(X, window_shape=self.window_size, axis=-1)[:, :, ::self.step_size, :]
            win_mean = np.mean(windows, axis=-1)
            win_std = np.std(windows, axis=-1)
            win_feat = np.concatenate([win_mean, win_std], axis=-1)
            feat_list.append(win_feat.reshape(n_trials, -1))

        # 3. 频域特征（批量计算PSD）
        if self.use_freq_features:
            from scipy.signal import welch, periodogram
            nperseg = self.psd_params.get('nperseg', min(256, n_times))
            noverlap = self.psd_params.get('noverlap', nperseg // 2)

            if self.psd_method == 'welch':
                freqs, psd = welch(X, fs=self.sfreq, nperseg=nperseg, noverlap=noverlap, axis=-1)
            else:
                freqs, psd = periodogram(X, fs=self.sfreq, axis=-1)

            total_power = np.sum(psd, axis=-1, keepdims=True) + 1e-8
            freq_feat_parts = []

            for fmin, fmax in self.freq_bands.values():
                band_mask = np.logical_and(freqs >= fmin, freqs <= fmax)
                band_power = np.sum(psd[..., band_mask], axis=-1, keepdims=True)
                rel_power = band_power / total_power
                freq_feat_parts.append(band_power)
                freq_feat_parts.append(rel_power)

            spectral_centroid = np.sum(freqs * psd, axis=-1, keepdims=True) / total_power
            mean_freq = spectral_centroid
            spectral_var = np.sum(((freqs[None, None, :] - mean_freq) ** 2) * psd, axis=-1, keepdims=True) / total_power
            spectral_kurt = np.sum(((freqs[None, None, :] - mean_freq) ** 4) * psd, axis=-1, keepdims=True) / (
                        total_power * (spectral_var ** 2 + 1e-8))

            freq_feat_parts.extend([spectral_centroid, spectral_kurt])
            freq_feat = np.concatenate(freq_feat_parts, axis=-1)
            feat_list.append(freq_feat.reshape(n_trials, -1))

        return np.concatenate(feat_list, axis=1).astype(np.float32)


def sort_file_name(string_list, reverse=False):
    """Sort a list of file names alphabetically"""
    return sorted(string_list, reverse=reverse)


def classify_files_by_filename_substring(root_dir, target_substring):
    """Recursively search and group files containing a substring by extension"""
    classified_files = {}
    if not os.path.isdir(root_dir):
        raise ValueError(f"Not a valid directory: {root_dir}")
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if target_substring.lower() in filename.lower():
                file_path = os.path.join(dirpath, filename)
                _, ext = os.path.splitext(filename)
                ext = ext[1:].lower() if ext else "no_extension"
                if ext not in classified_files:
                    classified_files[ext] = []
                classified_files[ext].append(file_path)
    for ext in classified_files:
        classified_files[ext].sort()
    return classified_files

def build_pipeline(n_comp, current_bands, n_channels=24, classifier_name='knn', classifier_params=None):
    """Assemble feature extraction"""
    if classifier_params is None:
        if classifier_name == 'knn':
            classifier_params = {'n_neighbors': 5, 'weights': 'distance', 'metric': 'euclidean'}

    if classifier_name == 'knn':
        clf = KNeighborsClassifier(**classifier_params)
    else:
        raise ValueError(f"Unsupported: {classifier_name}")

    feature_union = FeatureUnion([
        ('fbcsp', MultiBandCSP(
            n_components=n_comp,
            sfreq=250,
            bands=current_bands,
            reg="oas",
            n_jobs=1,
            log=True,
            norm_trace=True
        )),
        ('stats', EEGStatisticalFeatures_tf(
            use_sliding_window=True,
            window_size=50,
            step_size=25,
        )),
    ])

    pipeline = Pipeline([
        ('features', feature_union),
        ('scaler', RobustScaler()),
        ('selector', SelectKBest(f_classif, k=100)),
        ('clf', clf),
    ])
    return pipeline

def main():
    FIXED_N_COMP = 4
    current_bands = [(4, 8), (8, 13), (13, 30), (30, 45), (45, 80)]
    data_list = ["ani_s1_signal_sub_newset_del_ch"]

    # KNN hyperparameter grid
    param_grids = {
        'knn': {
            'n_neighbors': [1, 3, 5, 7],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'cosine']
        }
    }

    all_classifiers = ['knn']
    classifier_param_list = {}
    for clf_name in all_classifiers:
        classifier_param_list[clf_name] = list(ParameterGrid(param_grids[clf_name]))
        print(f"{clf_name.upper()}: {len(classifier_param_list[clf_name])} parameter combinations")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for dataset_name in data_list:
        folder_path = dataset_name

        print("\n" + "=" * 80)
        print(f"Processing dataset: {dataset_name}")
        print("=" * 80)

        try:
            save_path_data = classify_files_by_filename_substring(folder_path, target_substring="data")
            save_path_label = classify_files_by_filename_substring(folder_path, target_substring="label")
            data_files = sort_file_name(save_path_data["npy"])
            label_files = sort_file_name(save_path_label["npy"])
        except Exception as e:
            print(f"skip {folder_path}，File loading error: {e}")
            continue

        print(f"\n Found {len(data_files)} subject files")

        # 逐个被试处理
        for sub_idx, (d_path, l_path) in enumerate(zip(data_files, label_files), 1):
            sub_name = Path(d_path).stem.split('_')[0]

            print(f"\n{'=' * 60}")
            print(f" Subject [{sub_idx}/{len(data_files)}]: {sub_name}")
            print(f"{'=' * 60}")

            try:
                X = np.load(d_path)
                y = np.load(l_path)
                if X.shape[-1] == 1125:
                    X = X[:, :, :1001]
            except Exception as e:
                print(f"Data loading failed: {e}")
                continue

            print(f" Data shape: {X.shape} | Labels: {np.unique(y)}")

            le = LabelEncoder()
            y_enc = le.fit_transform(y)

            clf_name = 'knn'
            params_list = classifier_param_list[clf_name]
            all_rows = []

            print(f"\n Evaluating all {clf_name.upper()} parameter combinations...")

            for idx, params in enumerate(params_list, 1):
                clf_model = build_pipeline(
                    n_comp=FIXED_N_COMP,
                    current_bands=current_bands,
                    n_channels=24,
                    classifier_name=clf_name,
                    classifier_params=params
                )

                scoring = {
                    'accuracy': 'accuracy',
                    'f1': 'f1_weighted',
                    'precision': 'precision_weighted',
                    'recall': 'recall_weighted',
                }

                cv_results = cross_validate(
                    clf_model, X, y_enc,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=False,
                    n_jobs=1
                )

                row = {
                    'Parameter ': idx,
                    **params,
                    'Accuracy': cv_results['test_accuracy'].mean(),
                    'F1': cv_results['test_f1'].mean(),
                    'Precision': cv_results['test_precision'].mean(),
                    'Recall': cv_results['test_recall'].mean(),
                }
                all_rows.append(row)

            df_all = pd.DataFrame(all_rows)
            df_all = df_all.sort_values('Accuracy', ascending=False).reset_index(drop=True)
            best_row = df_all.iloc[0]

            print(f"\n   Evaluation done | Best result:")
            print(f"      Accuracy: {best_row['Accuracy']:.4f}")
            print(f"      F1: {best_row['F1']:.4f}")
            print(f"      Precision: {best_row['Precision']:.4f}")
            print(f"      Recall: {best_row['Recall']:.4f}")
            print(
                f"      Best params: n_neighbors={best_row['n_neighbors']}, weights={best_row['weights']}, metric={best_row['metric']}")

    print("\n" + "=" * 80)
    print("All subjects processed！")
    print("=" * 80)


if __name__ == "__main__":
    main()