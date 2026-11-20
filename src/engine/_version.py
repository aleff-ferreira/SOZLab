"""Single source of truth for the SOZLab version and algorithm revisions."""

__version__ = "2.1.0"

#: Bump an entry when the *numerical definition* of that quantity changes. These are
#: folded into cache keys and written to provenance so old cached results are never
#: reused after a definition changes.
ALGORITHM_VERSIONS = {
    "soz": "2",  # vectorised zone evaluation, PBC-safe COM/COG probes
    "residence": "2",  # midpoint-rule durations, censoring, survival correlation function
    "density": "2",  # solute-centred grid, per-frame wrap + superposition, bulk normalisation
    "extraction": "2",  # identity-verified frame extraction
}

#: Version of the exported file layout. Additive changes only: v2.1 keeps every v2.0 column.
SCHEMA_VERSION = "2.1"
