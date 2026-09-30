
import math
from collections import Counter
import numpy as np


AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
TOKENS = AA_ORDER + "X"
TOKEN_INDEX = {token: index for index, token in enumerate(TOKENS)}


TOKEN_INDEX["-"] = TOKEN_INDEX["X"]
FEATURE_SCHEMA_VERSION = "shared-padding-unknown-v1"




RAW_PROPERTY_TABLES = {
    "hydropathy_z": dict(zip(AA_ORDER, [
        1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
        1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3,
    ])),
    "volume_z": dict(zip(AA_ORDER, [
        88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7, 168.6, 166.7,
        162.9, 114.1, 112.7, 143.8, 173.4, 89.0, 116.1, 140.0, 227.8, 193.6,
    ])),
    "polarity_z": dict(zip(AA_ORDER, [
        8.1, 5.5, 13.0, 12.3, 5.2, 9.0, 10.4, 5.2, 11.3, 4.9,
        5.7, 11.6, 8.0, 10.5, 10.5, 9.2, 8.6, 5.9, 5.4, 6.2,
    ])),
    "flexibility_z": dict(zip(AA_ORDER, [
        0.357, 0.346, 0.511, 0.497, 0.314, 0.544, 0.323, 0.462, 0.466, 0.365,
        0.295, 0.463, 0.509, 0.493, 0.529, 0.507, 0.444, 0.386, 0.305, 0.420,
    ])),
    "helix_z": dict(zip(AA_ORDER, [
        1.42, 0.70, 1.01, 1.51, 1.13, 0.57, 1.00, 1.08, 1.16, 1.21,
        1.45, 0.67, 0.57, 1.11, 0.98, 0.77, 0.83, 1.06, 1.08, 0.69,
    ])),
    "sheet_z": dict(zip(AA_ORDER, [
        0.83, 1.19, 0.54, 0.37, 1.38, 0.75, 0.87, 1.60, 0.74, 1.30,
        1.05, 0.89, 0.55, 1.10, 0.93, 0.75, 1.19, 1.70, 1.37, 1.47,
    ])),
    "turn_z": dict(zip(AA_ORDER, [
        0.66, 1.19, 1.46, 0.74, 0.60, 1.56, 0.95, 0.47, 1.01, 0.59,
        0.60, 1.56, 1.52, 0.98, 0.95, 1.43, 0.96, 0.50, 0.96, 1.14,
    ])),
}


def _zscore_table(table):
    values = np.asarray([table[aa] for aa in AA_ORDER], dtype=np.float32)
    standard_deviation = float(values.std()) or 1.0
    mean = float(values.mean())
    return {aa: (float(table[aa]) - mean) / standard_deviation for aa in AA_ORDER}


PROPERTY_TABLES = {name: _zscore_table(table) for name, table in RAW_PROPERTY_TABLES.items()}
CHARGE = {aa: 0.0 for aa in AA_ORDER}
CHARGE.update({"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1})
DISORDER_PROMOTING = {aa: float(aa in "EKRQSPG") for aa in AA_ORDER}

ONEHOT_CHANNELS = tuple(f"onehot_{token}" for token in AA_ORDER) + (
    "onehot_X_or_padding",
)
LOCAL_CHANNELS = tuple(PROPERTY_TABLES) + ("charge", "disorder_promoting")
CONTEXT_CHANNELS = (
    "context_hydropathy_mean",
    "context_hydropathy_std",
    "context_charge_mean",
    "context_disorder_fraction",
    "context_glycine_fraction",
    "context_proline_fraction",
    "context_entropy_normalized",
)


def normalize_encoding_name(name):

    normalized = str(name).strip().lower().replace("_", "-")
    aliases = {
        "onehot": "onehot",
        "one-hot": "onehot",
        "physchem": "physchem-v1",
        "physchem-v1": "physchem-v1",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported encoding: {name!r}; use onehot or physchem-v1.")
    return aliases[normalized]


def channel_names(encoding):
    mode = normalize_encoding_name(encoding)
    if mode == "onehot":
        return ONEHOT_CHANNELS
    return ONEHOT_CHANNELS + LOCAL_CHANNELS + CONTEXT_CHANNELS


def _normalized_entropy(sequence):
    canonical = [aa for aa in sequence if aa in AA_ORDER]
    if not canonical:
        return 0.0
    counts = Counter(canonical)
    probabilities = np.asarray(list(counts.values()), dtype=np.float32) / len(canonical)
    entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    return entropy / math.log2(len(AA_ORDER))


def _sanitize_sequence(sequence):
    return [symbol if symbol in TOKEN_INDEX else "X" for symbol in sequence.upper()]


def encode_sequences(sequences, encoding="physchem-v1"):

    mode = normalize_encoding_name(encoding)
    if not sequences:
        raise ValueError("No sequences are available for encoding.")
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError("All sequences must be non-empty and have the same length.")

    sequence_length = lengths.pop()
    names = channel_names(mode)
    output = np.zeros((len(sequences), sequence_length, len(names)), dtype=np.float32)

    property_offset = len(ONEHOT_CHANNELS)
    context_offset = property_offset + len(LOCAL_CHANNELS)

    for sample_index, raw_sequence in enumerate(sequences):
        sequence = _sanitize_sequence(raw_sequence)
        canonical = [aa for aa in sequence if aa in AA_ORDER]

        hydropathy = [PROPERTY_TABLES["hydropathy_z"][aa] for aa in canonical]
        charges = [CHARGE[aa] for aa in canonical]
        disorder = [DISORDER_PROMOTING[aa] for aa in canonical]
        denominator = float(len(canonical)) if canonical else 1.0
        context = np.asarray([
            float(np.mean(hydropathy)) if hydropathy else 0.0,
            float(np.std(hydropathy)) if hydropathy else 0.0,
            float(np.mean(charges)) if charges else 0.0,
            float(np.mean(disorder)) if disorder else 0.0,
            sum(aa == "G" for aa in canonical) / denominator,
            sum(aa == "P" for aa in canonical) / denominator,
            _normalized_entropy(canonical),
        ], dtype=np.float32)

        for position, residue in enumerate(sequence):
            output[sample_index, position, TOKEN_INDEX[residue]] = 1.0
            if mode == "onehot":
                continue
            if residue in AA_ORDER:
                for property_index, property_name in enumerate(PROPERTY_TABLES):
                    output[sample_index, position, property_offset + property_index] = (
                        PROPERTY_TABLES[property_name][residue]
                    )
                output[sample_index, position, property_offset + len(PROPERTY_TABLES)] = CHARGE[residue]
                output[sample_index, position, property_offset + len(PROPERTY_TABLES) + 1] = (
                    DISORDER_PROMOTING[residue]
                )
            output[sample_index, position, context_offset:] = context

    if not np.isfinite(output).all():
        raise ValueError("The encoded tensor contains NaN or infinite values.")
    return output
