# Setup shared across Buildomat CI builds.
#
# This file contains environment variables shared across Buildomat CI jobs.

# Color the output for easier readability.
export CARGO_TERM_COLOR=always

# Fail on warnings.
export RUSTFLAGS="-D warnings"

# We only build once, so there's no need to incur the overhead of incremental
# compilation.
export CARGO_INCREMENTAL=0
