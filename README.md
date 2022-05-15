## KHU 2022-1학기 캡스톤 디자인1 
### [2pm] Tri-Former
-------------
### Folder Description

> ./Autoformer_original 
: 기존 autoformer code

> ./Add trend auto_correlation 
: 기존 autoformer + trend의 auto correlation block 3개 추가

> ./Tri-Former_ver1 
: 새로운 Decomposition block을 적용하여
seasonal, trend, residual의 3가지 input data로 나눴고,
trend와 residual에도 autocorrelation block 추가함.

> ./Tri-Former_ver2
: 새로운 Decomposition block 적용하여
seasonal, trend, residual의 3가지 input data로 나눴고,
residual에만 autocorrelation block 추가함. (현재 Error 있음.)
