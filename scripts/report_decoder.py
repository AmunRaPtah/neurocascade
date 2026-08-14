"""P7.2's model comparison, written out per endpoint.

    python scripts/report_decoder.py > results/decoder_comparison.txt

One block per behavioural endpoint: every graded single-region series that
speaks to it, the four families' held-out errors, the verdict and its
reason. Endpoints with no usable series print their searched-and-empty
record instead, which is the majority of them and is the result rather than
a gap in the report.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade import decoder          # noqa: E402
from neurocascade import endpoints        # noqa: E402


def main():
    print("P7.2: is a nonlinear decoder better than P7.1's linear readout?")
    print("=" * 72)
    print()
    print("Criterion: leave-one-out cross-validation over the drive levels")
    print("of a graded SINGLE-REGION manipulation, comparing four families.")
    print("A nonlinearity is adopted only where the non-monotone family")
    print("predicts held-out levels better than both monotone families, its")
    print("turning point lies inside the tested range, and the measured")
    print("means themselves reverse.")
    print()
    print("Series in the library: %d" % len(decoder.SERIES))
    adopted = decoder.adopted_nonlinearities()
    print("Nonlinearities adopted: %d" % len(adopted))
    for (endpoint_key, acronym), term in sorted(adopted.items()):
        print("  %-22s %-6s %s (licensed by %s)"
              % (endpoint_key, acronym, term.shape, term.licensed_by))
    print()

    for verdict in decoder.all_endpoint_verdicts():
        print()
        print(verdict.report())
        if not verdict.series_verdicts:
            record = decoder.SEARCHED_AND_EMPTY.get(verdict.endpoint_key)
            if record:
                for line in record.strip().split("\n"):
                    print("    " + line)
        print()

    print()
    print("=" * 72)
    print("The three cases P7.1 named for P7.2, and what happened to each")
    print("=" * 72)
    print(decoder.P71_TARGETS.strip())

    print()
    print("=" * 72)
    print("Rejection reasons, in full")
    print("=" * 72)
    for name in sorted(decoder.REJECTION_REASONS):
        print("  %s:" % name)
        print("    %s" % decoder.REJECTION_REASONS[name])

    print()
    print("=" * 72)
    print("Endpoints, for reference (P7.1)")
    print("=" * 72)
    for key in endpoints.ENDPOINTS:
        ep = endpoints.ENDPOINTS[key]
        print("  %-22s %2d terms, %2d regions"
              % (key, len(ep.terms), len(ep.regions())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
