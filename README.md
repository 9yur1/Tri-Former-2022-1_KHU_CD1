# Tri-Former-2022-1_KHU_CD1
## KHU 2022-1학기 캡스톤 디자인1 [2pm] Tri-Former

#### No autocorrelation ver

>* 'tri_decomp' : 사용 trend, seasonal, residual 분리
>* 'auto-correlation block' : 활용안함 (기존 autoformer처럼 seasonal에는 적용)

#### with autocorrelation ver

>* 'tri_decomp' : 사용 trend, seasonal, residual 분리
>* 'auto-correlation block' : trend에만 적용 (기존 autoformer처럼 seasonal에는 적용)

#### full residual ver

>* 'tri_decomp' : 사용 trend, seasonal, residual 분리
>* 'auto-correlation block' : 활용안함 (기존 autoformer처럼 seasonal에는 적용)
>* residual은 디코더 통과 없이 정답값으로 학습