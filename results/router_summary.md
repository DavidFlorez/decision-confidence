# Part 2 LLM-Router Summary (`CARROT-LLM-Routing/SPROUT-o3mini`, n=500 test, 100 fit)

## Baselines (Llama-3.2-3B -> OpenAI o3-mini)
- Always Small (`Llama-3.2-3B`): Quality 64.5% [60.6, 68.5], Cost $0.057/1k
- Always Frontier (`o3-mini`): Quality 92.8% [90.7, 94.7], Cost $5.168/1k
- Oracle Upper Bound: Quality 93.8% [91.9, 95.6], Cost $3.001/1k, Route Acc 100.0% [100.0, 100.0]
- Prompt-Length Heuristic: AUROC 0.520 [0.469, 0.570], APGR 0.544 [0.500, 0.586]

## Model Comparison (n=500 test)
- Drex Max v1: AUROC 0.583 [0.533, 0.632], APGR 0.458 [0.416, 0.504], PGR@50%Cost 46.9% [39.6, 54.6], tau=4.649, theta_raw=0.759
- d1-2B:       AUROC 0.615 [0.565, 0.662], APGR 0.416 [0.377, 0.461], PGR@50%Cost 36.1% [29.3, 42.9], tau=2.471, theta_raw=0.609
- Delta (Drex - d1): AUROC -0.032 [-0.065, 0.000], APGR 0.042 [0.014, 0.070]

## Threshold Transfer & 100-Label Tuning (Routing Accuracy / Cost / Quality)
- Drex Max v1 | argmax_050        : RouteAcc=55.0% [50.6, 59.2], Esc=69.8%, SysQual=85.1%, Cost=$4.33/1k
- Drex Max v1 | nace_080          : RouteAcc=57.2% [52.6, 61.4], Esc=50.0%, SysQual=82.6%, Cost=$3.54/1k
- Drex Max v1 | fit100_raw_theta  : RouteAcc=56.0% [51.8, 60.2], Esc=54.4%, SysQual=82.8%, Cost=$3.70/1k
- Drex Max v1 | fit100_temp_theta : RouteAcc=56.0% [51.8, 60.2], Esc=54.4%, SysQual=82.8%, Cost=$3.70/1k
- d1-2B | argmax_050        : RouteAcc=59.4% [55.0, 63.8], Esc=45.0%, SysQual=79.7%, Cost=$3.34/1k
- d1-2B | nace_080          : RouteAcc=54.8% [50.4, 58.8], Esc=14.4%, SysQual=69.6%, Cost=$1.45/1k
- d1-2B | fit100_raw_theta  : RouteAcc=57.2% [52.8, 61.4], Esc=32.4%, SysQual=74.9%, Cost=$2.69/1k
- d1-2B | fit100_temp_theta : RouteAcc=57.2% [52.8, 61.4], Esc=32.4%, SysQual=74.9%, Cost=$2.69/1k
- Cross-swap: Drex@d1_theta RouteAcc=54.8% [50.6, 59.0], d1@Drex_theta RouteAcc=55.4% [51.0, 59.4]

## Part 1 Decision-Model Cascade (`d1-2B` Local -> `Drex Max v1` GPU, n=464)
- all_d1            : Local=100.0% (464/464), Combined Acc=76.1% [72.2, 79.7]
- d1_raw_ge_093     : Local=4.5% (21/464), Combined Acc=84.9% [81.5, 87.9]
- d1_temp_ge_093    : Local=18.8% (87/464), Combined Acc=84.9% [81.5, 87.9]
- d1_temp_fit95prec : Local=25.4% (118/464), Combined Acc=84.7% [81.2, 87.7]
- d1_temp_ge_0859   : Local=34.7% (161/464), Combined Acc=84.3% [80.8, 87.3]
- all_drex          : Local=0.0% (0/464), Combined Acc=84.9% [81.5, 87.9]
