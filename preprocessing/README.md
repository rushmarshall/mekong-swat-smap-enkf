# SMAP preprocessing (R)

The four R scripts in `R/` document how SMAP granules were extracted onto the model's
subbasins and bias-corrected for the Mekong application. They produce the subbasin series
the coupled model reads (`data/smap_subbasin_daily/smap_<product>_subbasin_daily.csv`, format
in [../docs/03_data.md](../docs/03_data.md)).

| Script | Does |
|---|---|
| `04_1_dataPreprocessing_smExtraction_smap_knownShp_multi.r` | extracts 9 km SMAP over the subbasin polygons |
| `04_2_dataPreprocessing_smExtraction_smap1km_knownShp_multi.r` | the same for the 1 km downscaled product |
| `06_1_swatdaInput_smBiascorrection.r` | bias-corrects 9 km retrievals toward the model climatology |
| `06_2_swatdaInput_smBiascorrection1km.r` | the same for 1 km |

They come from the `smap_enkf_swat` workflow of Le et al. (2022) and are kept as written,
including hard-coded paths from their original environment (`source('C:/Users/...')`) and
shapefile locations, which must be edited before use. They are not tested here, and nothing
in `src/mekong_da/` imports or derives from them. The Python equivalent of the bias
correction is `mekong_da.biascorrect`, which is tested; the basin-mean extraction for the
offline filter is `scripts/extract_basin_mean.py`.

## Licence

GPL-3.0, text in `R/LICENSE`, upstream README in `R/UPSTREAM_README.md`. This directory is
licensed separately from the rest of the repository; see [../NOTICE.md](../NOTICE.md).
