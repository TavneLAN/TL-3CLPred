
from pathlib import Path
from typing import FrozenSet, NamedTuple


CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


class ProteinRecord(NamedTuple):

    identifier: str
    sequence: str
    positive_sites: FrozenSet[int]


class SiteWindow(NamedTuple):

    protein_id: str
    position0: int
    residue: str
    sequence: str
    label: int

    @property
    def position1(self) -> int:

        return self.position0 + 1


def _decode_marked_sequence(raw_sequence, identifier):

    residues = []
    positives = set()

    for symbol in raw_sequence.upper():
        if symbol.isspace():
            continue
        if symbol == "#":
            if not residues:
                raise ValueError(f"{identifier}: '#' has no preceding amino acid to label.")
            positives.add(len(residues) - 1)
            continue
        if symbol == "*":
            continue
        if not symbol.isalpha():
            raise ValueError(f"{identifier}: sequence contains unsupported character {symbol!r}.")
        residues.append(symbol if symbol in CANONICAL_AA else "X")

    if not residues:
        raise ValueError(f"{identifier}: FASTA record has no sequence.")
    return "".join(residues), frozenset(positives)


def read_marked_fasta(path):

    fasta_path = Path(path)
    if not fasta_path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    records = []
    seen_ids = set()
    current_id = None
    sequence_lines = []

    def finish_record() -> None:
        nonlocal current_id, sequence_lines
        if current_id is None:
            return
        sequence, positives = _decode_marked_sequence("".join(sequence_lines), current_id)
        records.append(ProteinRecord(current_id, sequence, positives))
        sequence_lines = []

    with fasta_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish_record()
                identifier = line[1:].strip()
                if not identifier:
                    raise ValueError(f"{fasta_path}:{line_number}: FASTA ID cannot be empty.")
                if identifier in seen_ids:
                    raise ValueError(f"{fasta_path}:{line_number}: duplicate FASTA ID: {identifier}")
                seen_ids.add(identifier)
                current_id = identifier
            else:
                if current_id is None:
                    raise ValueError(f"{fasta_path}:{line_number}: sequence appears before the first FASTA header.")
                sequence_lines.append(line)

    finish_record()
    if not records:
        raise ValueError(f"{fasta_path}: no FASTA records found.")
    return records


def parse_focus_residues(value):

    parts = value.split(",") if isinstance(value, str) else list(value)
    residues = []
    for part in parts:
        residue = str(part).strip().upper()
        if not residue:
            continue
        if len(residue) != 1 or residue not in CANONICAL_AA:
            raise ValueError(f"focus residue must be a standard one-letter amino acid code; got {part!r}")
        if residue not in residues:
            residues.append(residue)
    if not residues:
        raise ValueError("At least one focus residue is required.")
    return tuple(residues)


def iter_site_windows(records, focus_residues, radius=15, pad_symbol="-"):

    if radius < 1:
        raise ValueError("radius must be at least 1.")
    if len(pad_symbol) != 1:
        raise ValueError("pad_symbol must be a single character.")

    focus = frozenset(parse_focus_residues(focus_residues))
    expected_length = 2 * radius
    left_offsets = range(-(radius - 1), 1)
    right_offsets = range(1, radius + 1)

    for record in records:
        for position, residue in enumerate(record.sequence):
            if residue not in focus:
                continue

            chars = []

            for offset in list(left_offsets) + list(right_offsets):
                index = position + offset
                chars.append(record.sequence[index] if 0 <= index < len(record.sequence) else pad_symbol)
            window = "".join(chars)
            if len(window) != expected_length:
                raise AssertionError("Internal error: candidate window length is incorrect.")

            yield SiteWindow(
                protein_id=record.identifier,
                position0=position,
                residue=residue,
                sequence=window,
                label=int(position in record.positive_sites),
            )


def build_site_windows(records, focus_residues, radius=15):

    examples = list(iter_site_windows(records, focus_residues, radius=radius))
    if not examples:
        focus_text = ",".join(parse_focus_residues(focus_residues))
        raise ValueError(f"No candidate sites with center residue {focus_text} were found in the data.")
    return examples
