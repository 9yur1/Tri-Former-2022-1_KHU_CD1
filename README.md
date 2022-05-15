## KHU 2022-1학기 캡스톤 디자인1 
### [2pm] Tri-Former
-------------
### Folder Description

> __./Autoformer_original__ 
: 기존 autoformer code

> __./Add trend auto_correlation__ 
: 기존 autoformer + `trend`의 auto correlation block 3개 추가

> __./Tri-Former_ver1__
: `New Decomposition block` 적용
seasonal, trend, residual의 3가지 input data로 나눴고,
`trend`, `residual`에 `Auto-correlation block` 추가. (현재 Error 있음.)

> __./Tri-Former_ver2__
: `New Decomposition block` 적용
seasonal, trend, residual의 3가지 input data로 나눴고,
`residual`에만 `Auto-correlation block` 추가함. (현재 Error 있음.)
