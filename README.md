# PatternDiff
## PatternFasion Dataset

First, it is required to download three datasets: [DressCode](https://github.com/aimagelab/dress-code), [VITON-HD](https://github.com/shadow2496/VITON-HD) and [StreetTryOn](https://github.com/cuiaiyu/street-tryon-benchmark). Note that only the upper body category data from the DressCode dataset is needed, and the dataset shall be renamed as dresscode. All other annotation data should be downloaded from this link. The data format of the dataset is shown below.
```
PatternFasion/
   --dresscode/
      --pattern/
      --texture/
      --dresscode.json
      --test_dresscode.text
      --train_dresscode.text
   --vitonhd/
      --pattern/
      --texture/
      --vitonhd.json
      --test_vitonhd.text
      --train_vitonhd.text
   --streettryon/
      --pattern/
      --texture/
      --streettryon.json
      --test_streettryon.text
      --train_streettryon.text
```
