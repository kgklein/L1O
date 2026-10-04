# L1 Observatory

L1O helps you work with spacecraft near Sun–Earth L1: retrieve spacecraft
positions, compare their configuration at a common time, and fetch magnetic-field
observations for analysis. The Python package and command-line tool are named
`l1obs`.

## Current capabilities

| Capability | Spacecraft | Available interface |
| --- | --- | --- |
| SSCWeb ephemerides and common-time configurations | Wind, ACE, DSCOVR, Aditya-L1, IMAP, SOLAR-1 | Python retrieval and configuration plotting; `l1obs positions` |
| Native-cadence GSE magnetic-field data | Wind, ACE, DSCOVR, IMAP | CDAWeb dataset retrieval with normalized components and archive magnitude |
| Configured plasma datasets | ACE, Wind, DSCOVR, SOHO | Legacy CDAWeb fetch path and hourly timeseries command |

Configured spacecraft do not guarantee archive coverage for a particular date.
Plasma retrieval currently expects a legacy CDAWeb response format; the complete
hourly timeseries command can fail with current clients. Use direct magnetic
retrieval for magnetic-only work. See [Input and Output](io.md) for these limits
and export requirements.

## Using L1O

- [Getting Started](getting-started.md): install the package and retrieve your first products.
- [Input and Output](io.md): choose the existing Python calls or CLI commands.
- [Data Products](data-products.md): understand columns, units, coordinates, cadence, and missing data.
