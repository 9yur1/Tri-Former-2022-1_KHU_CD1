## KHU 2022-1학기 캡스톤 디자인1 
### [2pm] Tri-Former
-------------
### Folder Description

> __./Autoformer_original__ 
: 기존 autoformer code

> __./Add trend auto_correlation__ 
: 기존 autoformer + trend의 auto correlation block 3개 추가

> __./Tri-Former_ver1__
: 새로운 Decomposition block을 적용하여
seasonal, trend, residual의 3가지 input data로 나눴고,
trend와 residual에도 autocorrelation block 추가함.

> __./Tri-Former_ver2__
: 새로운 Decomposition block 적용하여
seasonal, trend, residual의 3가지 input data로 나눴고,
residual에만 autocorrelation block 추가함. (현재 Error 있음.)
