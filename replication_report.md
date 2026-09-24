# Replication report (gate G4)

`tsresample` against the recorded canonical R experiment (jpmsilva1/ts-resampling-replication @ b1c7e57), per `Blueprint/docs/REPLICATION.md` §4. Learner: `lm` (the strict learner, §4.2b); 50 Monte Carlo splits per dataset; `r_quirks=True`; DS12/DS13 and DS23/DS24 on complete cases; DS12/DS13 excluded from R1-R3 (§4.2b, open item DS13). Datasets: 24. Per-iteration equality with R is not expected (different RNG streams and split positions) and is not tested.

| Assertion | Result | Tolerance | Verdict |
|---|---|---|---|
| R1 ranking (Spearman, lm) | mean 0.885, min 0.539 | ≥ 0.7 mean, ≥ 0.6 each | **FAIL** |
| R2 direction | 96.2% of 159 | ≥ 85 % | **pass** |
| R3 median in recorded IQR | 95.0% of 220 | ≥ 75 % | **pass** |
| R4 row counts | 10800/10800 exact | exact | **pass** |
| R5 Wilcoxon conclusion | see below | same sign and significance | **pass** |

## Diagnosis of R1 failures

- **DS19**: ρ = 0.539. Recorded strategy means span 0.0277 while the standard error of one mean is 0.0106, so the ranking of the nine strategies is dominated by split noise; R5 and R2 still hold there. DS19 is also the only dataset with ADR-0014's φ = (1, 0, 0) metric residual.

## R5: one-sided Wilcoxon over datasets, mean F1φ(strategy) − F1φ(baseline)

| Strategy | ours: median Δ > 0, p | recorded: median Δ > 0, p | match |
|---|---|---|---|
| UNDERB | True, 7.48e-05 | True, 7.48e-05 | yes |
| UNDERT | True, 6.35e-05 | True, 5.38e-05 | yes |
| UNDERTPhi | True, 7.48e-05 | True, 2.66e-05 | yes |
| OVERB | True, 1.83e-05 | True, 3.19e-05 | yes |
| OVERT | True, 5.38e-05 | True, 5.38e-05 | yes |
| OVERTPhi | True, 5.38e-05 | True, 5.38e-05 | yes |
| SMOTEB | True, 4.54e-05 | True, 4.54e-05 | yes |
| SMOTET | True, 6.35e-05 | True, 5.38e-05 | yes |
| SMOTETPhi | True, 7.48e-05 | True, 6.35e-05 | yes |

## R1: per-dataset ranking correlation

| Dataset | Spearman ρ |
|---|---|
| DS01 | 0.964 |
| DS02 | 0.939 |
| DS03 | 0.988 |
| DS04 | 0.939 |
| DS05 | 0.707 |
| DS06 | 0.952 |
| DS07 | 0.794 |
| DS08 | 0.988 |
| DS09 | 0.782 |
| DS10 | 0.976 |
| DS11 | 0.976 |
| DS14 | 0.927 |
| DS15 | 0.903 |
| DS16 | 0.891 |
| DS17 | 0.964 |
| DS18 | 0.879 |
| DS19 | 0.539 |
| DS20 | 0.867 |
| DS21 | 0.927 |
| DS22 | 0.988 |
| DS23 | 0.770 |
| DS24 | 0.806 |

## Mean F1φ, ours vs recorded (lm)

| Dataset | Workflow | ours | recorded |
|---|---|---|---|
| DS01 | baseline | 0.0000 | 0.0000 |
| DS01 | UNDERB | 0.5371 | 0.5385 |
| DS01 | UNDERT | 0.5331 | 0.5304 |
| DS01 | UNDERTPhi | 0.5285 | 0.5285 |
| DS01 | OVERB | 0.5411 | 0.5443 |
| DS01 | OVERT | 0.5459 | 0.5502 |
| DS01 | OVERTPhi | 0.5458 | 0.5493 |
| DS01 | SMOTEB | 0.5156 | 0.5227 |
| DS01 | SMOTET | 0.5163 | 0.5146 |
| DS01 | SMOTETPhi | 0.5098 | 0.5150 |
| DS02 | baseline | 0.0000 | 0.0000 |
| DS02 | UNDERB | 0.5995 | 0.5883 |
| DS02 | UNDERT | 0.6006 | 0.5937 |
| DS02 | UNDERTPhi | 0.5837 | 0.5752 |
| DS02 | OVERB | 0.5931 | 0.5903 |
| DS02 | OVERT | 0.5828 | 0.5781 |
| DS02 | OVERTPhi | 0.5822 | 0.5799 |
| DS02 | SMOTEB | 0.5734 | 0.5675 |
| DS02 | SMOTET | 0.5537 | 0.5514 |
| DS02 | SMOTETPhi | 0.5571 | 0.5524 |
| DS03 | baseline | 0.0716 | 0.0858 |
| DS03 | UNDERB | 0.6124 | 0.6142 |
| DS03 | UNDERT | 0.6182 | 0.6053 |
| DS03 | UNDERTPhi | 0.6073 | 0.6017 |
| DS03 | OVERB | 0.6358 | 0.6295 |
| DS03 | OVERT | 0.6498 | 0.6487 |
| DS03 | OVERTPhi | 0.6516 | 0.6488 |
| DS03 | SMOTEB | 0.5996 | 0.5923 |
| DS03 | SMOTET | 0.5740 | 0.5729 |
| DS03 | SMOTETPhi | 0.5763 | 0.5783 |
| DS04 | baseline | 0.5851 | 0.5929 |
| DS04 | UNDERB | 0.5537 | 0.5508 |
| DS04 | UNDERT | 0.5584 | 0.5572 |
| DS04 | UNDERTPhi | 0.5429 | 0.5414 |
| DS04 | OVERB | 0.5694 | 0.5680 |
| DS04 | OVERT | 0.5639 | 0.5617 |
| DS04 | OVERTPhi | 0.5624 | 0.5601 |
| DS04 | SMOTEB | 0.5452 | 0.5395 |
| DS04 | SMOTET | 0.5524 | 0.5534 |
| DS04 | SMOTETPhi | 0.5385 | 0.5418 |
| DS05 | baseline | 0.2295 | 0.3459 |
| DS05 | UNDERB | 0.4427 | 0.4863 |
| DS05 | UNDERT | 0.4426 | 0.4863 |
| DS05 | UNDERTPhi | 0.4454 | 0.4908 |
| DS05 | OVERB | 0.4386 | 0.4806 |
| DS05 | OVERT | 0.4334 | 0.4775 |
| DS05 | OVERTPhi | 0.4334 | 0.4775 |
| DS05 | SMOTEB | 0.4580 | 0.4952 |
| DS05 | SMOTET | 0.4333 | 0.4900 |
| DS05 | SMOTETPhi | 0.4351 | 0.4906 |
| DS06 | baseline | 0.0000 | 0.0000 |
| DS06 | UNDERB | 0.4303 | 0.4302 |
| DS06 | UNDERT | 0.4302 | 0.4301 |
| DS06 | UNDERTPhi | 0.4380 | 0.4375 |
| DS06 | OVERB | 0.4314 | 0.4301 |
| DS06 | OVERT | 0.4334 | 0.4335 |
| DS06 | OVERTPhi | 0.4344 | 0.4362 |
| DS06 | SMOTEB | 0.4478 | 0.4469 |
| DS06 | SMOTET | 0.4460 | 0.4466 |
| DS06 | SMOTETPhi | 0.4469 | 0.4480 |
| DS07 | baseline | 0.5352 | 0.5304 |
| DS07 | UNDERB | 0.5370 | 0.5369 |
| DS07 | UNDERT | 0.5357 | 0.5371 |
| DS07 | UNDERTPhi | 0.5383 | 0.5402 |
| DS07 | OVERB | 0.5347 | 0.5374 |
| DS07 | OVERT | 0.5353 | 0.5360 |
| DS07 | OVERTPhi | 0.5356 | 0.5365 |
| DS07 | SMOTEB | 0.5465 | 0.5487 |
| DS07 | SMOTET | 0.5514 | 0.5538 |
| DS07 | SMOTETPhi | 0.5521 | 0.5537 |
| DS08 | baseline | 0.4983 | 0.4978 |
| DS08 | UNDERB | 0.4897 | 0.4893 |
| DS08 | UNDERT | 0.4915 | 0.4900 |
| DS08 | UNDERTPhi | 0.4936 | 0.4929 |
| DS08 | OVERB | 0.4761 | 0.4750 |
| DS08 | OVERT | 0.4798 | 0.4786 |
| DS08 | OVERTPhi | 0.4793 | 0.4778 |
| DS08 | SMOTEB | 0.4731 | 0.4717 |
| DS08 | SMOTET | 0.4876 | 0.4865 |
| DS08 | SMOTETPhi | 0.4784 | 0.4780 |
| DS09 | baseline | 0.2476 | 0.2476 |
| DS09 | UNDERB | 0.2329 | 0.2344 |
| DS09 | UNDERT | 0.2335 | 0.2383 |
| DS09 | UNDERTPhi | 0.2348 | 0.2358 |
| DS09 | OVERB | 0.2324 | 0.2376 |
| DS09 | OVERT | 0.2314 | 0.2354 |
| DS09 | OVERTPhi | 0.2327 | 0.2322 |
| DS09 | SMOTEB | 0.2396 | 0.2380 |
| DS09 | SMOTET | 0.2407 | 0.2478 |
| DS09 | SMOTETPhi | 0.2434 | 0.2439 |
| DS10 | baseline | 0.0000 | 0.0000 |
| DS10 | UNDERB | 0.6521 | 0.6516 |
| DS10 | UNDERT | 0.6530 | 0.6485 |
| DS10 | UNDERTPhi | 0.6500 | 0.6468 |
| DS10 | OVERB | 0.6666 | 0.6664 |
| DS10 | OVERT | 0.6672 | 0.6672 |
| DS10 | OVERTPhi | 0.6672 | 0.6672 |
| DS10 | SMOTEB | 0.6653 | 0.6642 |
| DS10 | SMOTET | 0.6463 | 0.6454 |
| DS10 | SMOTETPhi | 0.6466 | 0.6432 |
| DS11 | baseline | 0.2141 | 0.1864 |
| DS11 | UNDERB | 0.4379 | 0.4411 |
| DS11 | UNDERT | 0.4435 | 0.4420 |
| DS11 | UNDERTPhi | 0.4381 | 0.4397 |
| DS11 | OVERB | 0.4419 | 0.4422 |
| DS11 | OVERT | 0.4501 | 0.4492 |
| DS11 | OVERTPhi | 0.4501 | 0.4492 |
| DS11 | SMOTEB | 0.4693 | 0.4674 |
| DS11 | SMOTET | 0.4628 | 0.4573 |
| DS11 | SMOTETPhi | 0.4583 | 0.4554 |
| DS12 | baseline | 0.0000 | 0.0000 |
| DS12 | UNDERB | 0.5351 | 0.5670 |
| DS12 | UNDERT | 0.5392 | 0.5680 |
| DS12 | UNDERTPhi | 0.5419 | 0.5748 |
| DS12 | OVERB | 0.5490 | 0.5629 |
| DS12 | OVERT | 0.5458 | 0.5756 |
| DS12 | OVERTPhi | 0.5447 | 0.5747 |
| DS12 | SMOTEB | 0.5356 | 0.5722 |
| DS12 | SMOTET | 0.5305 | 0.5591 |
| DS12 | SMOTETPhi | 0.5314 | 0.5571 |
| DS13 | baseline | 0.3615 | 0.3532 |
| DS13 | UNDERB | 0.6094 | 0.6189 |
| DS13 | UNDERT | 0.6102 | 0.6175 |
| DS13 | UNDERTPhi | 0.6118 | 0.6106 |
| DS13 | OVERB | 0.6091 | 0.6171 |
| DS13 | OVERT | 0.6115 | 0.6265 |
| DS13 | OVERTPhi | 0.6121 | 0.6262 |
| DS13 | SMOTEB | 0.5974 | 0.5928 |
| DS13 | SMOTET | 0.5916 | 0.5870 |
| DS13 | SMOTETPhi | 0.5870 | 0.5838 |
| DS14 | baseline | 0.0000 | 0.0000 |
| DS14 | UNDERB | 0.3200 | 0.3211 |
| DS14 | UNDERT | 0.3248 | 0.3371 |
| DS14 | UNDERTPhi | 0.3400 | 0.3616 |
| DS14 | OVERB | 0.2920 | 0.3449 |
| DS14 | OVERT | 0.3530 | 0.3669 |
| DS14 | OVERTPhi | 0.3658 | 0.3711 |
| DS14 | SMOTEB | 0.3666 | 0.3777 |
| DS14 | SMOTET | 0.3444 | 0.3517 |
| DS14 | SMOTETPhi | 0.3444 | 0.3495 |
| DS15 | baseline | 0.0000 | 0.0000 |
| DS15 | UNDERB | 0.2890 | 0.3057 |
| DS15 | UNDERT | 0.3415 | 0.3418 |
| DS15 | UNDERTPhi | 0.3504 | 0.3486 |
| DS15 | OVERB | 0.2843 | 0.2957 |
| DS15 | OVERT | 0.2904 | 0.2714 |
| DS15 | OVERTPhi | 0.2851 | 0.2780 |
| DS15 | SMOTEB | 0.3576 | 0.3603 |
| DS15 | SMOTET | 0.3393 | 0.3372 |
| DS15 | SMOTETPhi | 0.3451 | 0.3415 |
| DS16 | baseline | 0.0000 | 0.0000 |
| DS16 | UNDERB | 0.3508 | 0.3597 |
| DS16 | UNDERT | 0.3506 | 0.3607 |
| DS16 | UNDERTPhi | 0.3675 | 0.3668 |
| DS16 | OVERB | 0.3422 | 0.3430 |
| DS16 | OVERT | 0.3273 | 0.3282 |
| DS16 | OVERTPhi | 0.3269 | 0.3266 |
| DS16 | SMOTEB | 0.3672 | 0.3642 |
| DS16 | SMOTET | 0.3647 | 0.3606 |
| DS16 | SMOTETPhi | 0.3664 | 0.3556 |
| DS17 | baseline | 0.0000 | 0.0000 |
| DS17 | UNDERB | 0.2900 | 0.2906 |
| DS17 | UNDERT | 0.3386 | 0.2906 |
| DS17 | UNDERTPhi | 0.3517 | 0.3468 |
| DS17 | OVERB | 0.3114 | 0.2724 |
| DS17 | OVERT | 0.2205 | 0.2017 |
| DS17 | OVERTPhi | 0.2544 | 0.2077 |
| DS17 | SMOTEB | 0.3671 | 0.3581 |
| DS17 | SMOTET | 0.3544 | 0.3563 |
| DS17 | SMOTETPhi | 0.3543 | 0.3542 |
| DS18 | baseline | 0.0000 | 0.0000 |
| DS18 | UNDERB | 0.2575 | 0.2926 |
| DS18 | UNDERT | 0.2981 | 0.2941 |
| DS18 | UNDERTPhi | 0.2861 | 0.2465 |
| DS18 | OVERB | 0.2631 | 0.2089 |
| DS18 | OVERT | 0.2895 | 0.2883 |
| DS18 | OVERTPhi | 0.2913 | 0.3086 |
| DS18 | SMOTEB | 0.3613 | 0.3508 |
| DS18 | SMOTET | 0.3418 | 0.3519 |
| DS18 | SMOTETPhi | 0.3492 | 0.3532 |
| DS19 | baseline | 0.0000 | 0.0000 |
| DS19 | UNDERB | 0.3695 | 0.3301 |
| DS19 | UNDERT | 0.3859 | 0.3528 |
| DS19 | UNDERTPhi | 0.3724 | 0.3524 |
| DS19 | OVERB | 0.3807 | 0.3579 |
| DS19 | OVERT | 0.3910 | 0.3516 |
| DS19 | OVERTPhi | 0.3897 | 0.3530 |
| DS19 | SMOTEB | 0.3852 | 0.3465 |
| DS19 | SMOTET | 0.3883 | 0.3516 |
| DS19 | SMOTETPhi | 0.3805 | 0.3367 |
| DS20 | baseline | 0.0000 | 0.0000 |
| DS20 | UNDERB | 0.3513 | 0.3513 |
| DS20 | UNDERT | 0.3536 | 0.3485 |
| DS20 | UNDERTPhi | 0.3412 | 0.3349 |
| DS20 | OVERB | 0.3348 | 0.3470 |
| DS20 | OVERT | 0.3192 | 0.3240 |
| DS20 | OVERTPhi | 0.3212 | 0.3137 |
| DS20 | SMOTEB | 0.3466 | 0.3514 |
| DS20 | SMOTET | 0.3531 | 0.3569 |
| DS20 | SMOTETPhi | 0.3606 | 0.3534 |
| DS21 | baseline | 0.9579 | 0.9793 |
| DS21 | UNDERB | 0.9616 | 0.9824 |
| DS21 | UNDERT | 0.9616 | 0.9826 |
| DS21 | UNDERTPhi | 0.9616 | 0.9823 |
| DS21 | OVERB | 0.9617 | 0.9826 |
| DS21 | OVERT | 0.9617 | 0.9825 |
| DS21 | OVERTPhi | 0.9617 | 0.9825 |
| DS21 | SMOTEB | 0.9617 | 0.9826 |
| DS21 | SMOTET | 0.9611 | 0.9821 |
| DS21 | SMOTETPhi | 0.9614 | 0.9822 |
| DS22 | baseline | 0.6403 | 0.6189 |
| DS22 | UNDERB | 0.5664 | 0.5594 |
| DS22 | UNDERT | 0.5648 | 0.5602 |
| DS22 | UNDERTPhi | 0.5477 | 0.5361 |
| DS22 | OVERB | 0.6160 | 0.5844 |
| DS22 | OVERT | 0.6108 | 0.5740 |
| DS22 | OVERTPhi | 0.6093 | 0.5708 |
| DS22 | SMOTEB | 0.6192 | 0.5932 |
| DS22 | SMOTET | 0.4466 | 0.4318 |
| DS22 | SMOTETPhi | 0.4397 | 0.4266 |
| DS23 | baseline | 0.3621 | 0.3793 |
| DS23 | UNDERB | 0.3219 | 0.3600 |
| DS23 | UNDERT | 0.3295 | 0.3785 |
| DS23 | UNDERTPhi | 0.3407 | 0.3922 |
| DS23 | OVERB | 0.3868 | 0.4060 |
| DS23 | OVERT | 0.3558 | 0.3886 |
| DS23 | OVERTPhi | 0.3558 | 0.3886 |
| DS23 | SMOTEB | 0.3557 | 0.3883 |
| DS23 | SMOTET | 0.3113 | 0.3624 |
| DS23 | SMOTETPhi | 0.3151 | 0.3705 |
| DS24 | baseline | 0.7597 | 0.8002 |
| DS24 | UNDERB | 0.7936 | 0.8205 |
| DS24 | UNDERT | 0.7934 | 0.8191 |
| DS24 | UNDERTPhi | 0.8019 | 0.8337 |
| DS24 | OVERB | 0.8009 | 0.8226 |
| DS24 | OVERT | 0.7915 | 0.8169 |
| DS24 | OVERTPhi | 0.7911 | 0.8170 |
| DS24 | SMOTEB | 0.7995 | 0.8210 |
| DS24 | SMOTET | 0.7864 | 0.8110 |
| DS24 | SMOTETPhi | 0.7941 | 0.8108 |
