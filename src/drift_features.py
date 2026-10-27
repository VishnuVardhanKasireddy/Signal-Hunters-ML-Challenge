import numpy as np
import pandas as pd


DRIFT_RATES = np.arange(-2.0, 2.01, 0.5, dtype=np.float32)


def drift_features(X, batch_size=128):
    """
    Extract strongest straight-line features from six panels.

    Input:
        X: (N, 6, 16, 128), uint8

    Output:
        DataFrame with per-panel track features.
    """
    n = len(X)
    rows = np.arange(16, dtype=np.int32)
    starts = np.arange(128, dtype=np.int32)

    results = []

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)

        # Keep zero pixels marked as missing.
        raw = X[batch_start:batch_end].astype(np.float32)
        valid = raw != 0
        raw[~valid] = np.nan

        # Robustly center and scale each panel.
        median = np.nanmedian(raw, axis=(2, 3), keepdims=True)
        centered = raw - median
        scale = np.nanpercentile(
            centered, 75, axis=(2, 3), keepdims=True
        ) - np.nanpercentile(
            centered, 25, axis=(2, 3), keepdims=True
        )
        scale = np.where(scale > 1e-6, scale, 1.0)

        z = centered / scale
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        batch_n = len(raw)
        features = {}

        for p in range(6):
            best_score = np.full(batch_n, -np.inf, dtype=np.float32)
            best_start = np.zeros(batch_n, dtype=np.float32)
            best_slope = np.zeros(batch_n, dtype=np.float32)
            best_mean = np.zeros(batch_n, dtype=np.float32)
            best_coverage = np.zeros(batch_n, dtype=np.float32)

            panel = z[:, p]
            panel_valid = valid[:, p]

            for slope in DRIFT_RATES:
                # One candidate track per starting frequency.
                freq = np.rint(
                    starts[None, :] + slope * rows[:, None]
                ).astype(np.int32)

                in_bounds = (freq >= 0) & (freq < 128)
                safe_freq = np.clip(freq, 0, 127)

                # Gather all 128 candidate tracks simultaneously.
                values = panel[:, rows[:, None], safe_freq]
                masks = panel_valid[:, rows[:, None], safe_freq]
                masks = masks & in_bounds[None, :, :]

                count = masks.sum(axis=1)
                total = (values * masks).sum(axis=1)

                # Reward coherent positive power across the track.
                scores = total / np.sqrt(np.maximum(count, 1))
                scores[count < 8] = -np.inf

                candidate_idx = np.argmax(scores, axis=1)
                candidate_score = scores[
                    np.arange(batch_n), candidate_idx
                ]
                candidate_count = count[
                    np.arange(batch_n), candidate_idx
                ]

                improve = candidate_score > best_score

                best_score[improve] = candidate_score[improve]
                best_start[improve] = candidate_idx[improve]
                best_slope[improve] = slope

                best_mean[improve] = (
                    total[
                        np.arange(batch_n), candidate_idx
                    ][improve]
                    / np.maximum(candidate_count[improve], 1)
                )

                best_coverage[improve] = candidate_count[improve] / 16.0

            # Avoid infinities for panels with no valid track.
            best_score[~np.isfinite(best_score)] = 0.0

            features[f'drift_p{p}_score'] = best_score
            features[f'drift_p{p}_start'] = best_start
            features[f'drift_p{p}_slope'] = best_slope
            features[f'drift_p{p}_mean'] = best_mean
            features[f'drift_p{p}_coverage'] = best_coverage

        results.append(pd.DataFrame(features))

        print(f"Processed {batch_end}/{n}")

    return pd.concat(results, ignore_index=True)


DRIFT_RATES_V2 = np.arange(-2.0, 2.01, 0.5, dtype=np.float32)

def drift_features_v2(X, batch_size=128):
    output = []
    rows = np.arange(16, dtype=np.int32)
    starts = np.arange(128, dtype=np.int32)

    for batch_start in range(0, len(X), batch_size):
        raw = X[batch_start:batch_start + batch_size].astype(np.float32)
        valid = raw != 0
        raw[~valid] = np.nan

        # Estimate and remove each panel's frequency-dependent background.
        channel_bg = np.nanmedian(raw, axis=2, keepdims=True)
        residual = raw - channel_bg

        # Robustly scale each panel.
        center = np.nanmedian(residual, axis=(2, 3), keepdims=True)
        residual -= center
        q25 = np.nanpercentile(residual, 25, axis=(2, 3), keepdims=True)
        q75 = np.nanpercentile(residual, 75, axis=(2, 3), keepdims=True)
        scale = q75 - q25
        scale = np.where(scale > 1e-6, scale, 1.0)

        z = np.nan_to_num(
            residual / scale,
            nan=0.0, posinf=0.0, neginf=0.0
        )

        features = {}
        n = len(raw)

        for p in range(6):
            panel = z[:, p]
            panel_valid = valid[:, p]

            best_score = np.zeros(n, dtype=np.float32)
            best_start = np.zeros(n, dtype=np.float32)
            best_slope = np.zeros(n, dtype=np.float32)
            best_coverage = np.zeros(n, dtype=np.float32)

            for slope in DRIFT_RATES_V2:
                freq = np.rint(
                    starts[None, :] + slope * rows[:, None]
                ).astype(np.int32)

                in_bounds = (freq >= 0) & (freq < 128)
                safe_freq = np.clip(freq, 0, 127)

                values = panel[:, rows[:, None], safe_freq]
                masks = panel_valid[:, rows[:, None], safe_freq]
                masks &= in_bounds[None, :, :]

                count = masks.sum(axis=1)
                total = (values * masks).sum(axis=1)

                # Penalize tracks with insufficient valid pixels.
                scores = total / np.sqrt(np.maximum(count, 1))
                scores[count < 8] = -np.inf

                candidate_idx = np.argmax(scores, axis=1)
                candidate_score = scores[
                    np.arange(n), candidate_idx
                ]
                candidate_count = count[
                    np.arange(n), candidate_idx
                ]

                improve = candidate_score > best_score
                best_score[improve] = candidate_score[improve]
                best_start[improve] = candidate_idx[improve]
                best_slope[improve] = slope
                best_coverage[improve] = (
                    candidate_count[improve] / 16.0
                )

            features[f"v2_drift_p{p}_score"] = best_score
            features[f"v2_drift_p{p}_start"] = best_start
            features[f"v2_drift_p{p}_slope"] = best_slope
            features[f"v2_drift_p{p}_coverage"] = best_coverage

        output.append(pd.DataFrame(features))
        print(f"Processed {min(batch_start + batch_size, len(X))}/{len(X)}")

    return pd.concat(output, ignore_index=True)